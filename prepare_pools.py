"""Train fixed classifiers and regenerate the public evaluation pools."""

import argparse
import csv
import hashlib
import io
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
UCI = 'https://archive.ics.uci.edu/static/public/'
SOURCES = {
    'optdigits': ('80/optical+recognition+of+handwritten+digits.zip', ('optdigits.tra', 'optdigits.tes'), ',', -1),
    'pendigits': ('81/pen+based+recognition+of+handwritten+digits.zip', ('pendigits.tra', 'pendigits.tes'), ',', -1),
    'satimage': ('146/statlog+landsat+satellite.zip', ('sat.trn', 'sat.tst'), None, -1),
    'letter': ('59/letter+recognition.zip', ('letter-recognition.data',), ',', 0),
    'dry_bean': ('602/dry+bean+dataset.zip', None, None, None),
    'har': ('240/human+activity+recognition+using+smartphones.zip', None, None, None),
}


def read_numeric_archive(archive, files, delimiter, label_column):
    entries = {Path(name).name: name for name in archive.namelist()}
    matrices = [np.genfromtxt(io.BytesIO(archive.read(entries[file])), delimiter=delimiter, dtype=str)
                for file in files]
    matrix = np.vstack(matrices)
    return np.delete(matrix, label_column, axis=1).astype(float), matrix[:, label_column]


def read_bean(archive):
    name = next(name for name in archive.namelist() if name.endswith('/Dry_Bean_Dataset.arff'))
    lines = archive.read(name).decode('utf-8').splitlines()
    data_start = next(i for i, line in enumerate(lines) if line.strip().lower() == '@data')
    values = list(csv.reader(line for line in lines[data_start + 1:] if line.strip() and not line.startswith('%')))
    matrix = np.array(values)
    return matrix[:, :-1].astype(float), matrix[:, -1]


def read_har(archive):
    inner = next(name for name in archive.namelist() if name.endswith('/UCI HAR Dataset.zip') or name == 'UCI HAR Dataset.zip')
    with ZipFile(io.BytesIO(archive.read(inner))) as dataset:
        def read(name):
            member = next(member for member in dataset.namelist() if member.endswith('/' + name))
            return np.loadtxt(io.BytesIO(dataset.read(member)))

        train = read('train/X_train.txt')
        test = read('test/X_test.txt')
        labels = np.concatenate((read('train/y_train.txt'), read('test/y_test.txt')))
        people_train = read('train/subject_train.txt')
        people_test = read('test/subject_test.txt')
        if np.intersect1d(people_train, people_test).size:
            raise ValueError('HAR participants overlap between train and test')
    return np.concatenate((train, test)), labels, np.arange(len(train)), np.arange(len(train), len(labels))


def fit_models(features, labels, train, test, seed):
    models = {
        'logistic': make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000)),
        'forest': RandomForestClassifier(n_estimators=160, random_state=seed, n_jobs=1),
        'rbf_svm': make_pipeline(StandardScaler(), SVC(C=10, gamma='scale')),
        'extra_trees': ExtraTreesClassifier(n_estimators=160, random_state=seed, n_jobs=1),
        'naive_bayes': GaussianNB(),
    }
    predictions = []
    with threadpool_limits(limits=1):
        for model in models.values():
            model.fit(features[train], labels[train])
            predictions.append(model.predict(features[test]))
    return np.column_stack(predictions).astype(np.int16), np.array(list(models))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=ROOT / 'raw')
    parser.add_argument('--output', type=Path, default=ROOT / 'pools')
    parser.add_argument('--one', choices=SOURCES)
    args = parser.parse_args()
    args.raw.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'source_checksums.csv').open(newline='') as handle:
        checksums = {row['dataset']: row['sha256'] for row in csv.DictReader(handle)}
    for name, (url, files, delimiter, label_column) in SOURCES.items():
        if args.one and name != args.one:
            continue
        raw = args.raw / f'{name}.zip'
        if not raw.exists():
            with urlopen(UCI + url, timeout=120) as response:
                raw.write_bytes(response.read())
        if hashlib.sha256(raw.read_bytes()).hexdigest() != checksums[name]:
            raise ValueError(f'Source SHA-256 mismatch for {name}')
        seed = 20261116
        supplied_split = name == 'har' or (files and len(files) == 2)
        with ZipFile(raw) as archive:
            if name == 'har':
                features, labels, train, test = read_har(archive)
            else:
                features, labels = read_bean(archive) if name == 'dry_bean' else read_numeric_archive(
                    archive, files, delimiter, label_column)
                if supplied_split:
                    boundary = len(read_numeric_archive(archive, files[:1], delimiter, label_column)[1])
                    train, test = np.arange(boundary), np.arange(boundary, len(labels))
        _, labels = np.unique(labels, return_inverse=True)
        _, groups = np.unique(features, axis=0, return_inverse=True)
        removed = 0
        if supplied_split:
            repeats = np.isin(groups[test], groups[train])
            removed = int(np.count_nonzero(repeats))
            test = test[~repeats]
        else:
            split = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
            train, test = next(split.split(features, labels, groups))
        if np.intersect1d(groups[train], groups[test]).size:
            raise ValueError(f'Train and test share a feature vector in {name}')
        predictions, names = fit_models(features, labels, train, test, seed)
        np.savez_compressed(args.output / f'{name}.npz', labels=labels[test].astype(np.int16),
                            predictions=predictions, models=names, test_indices=test, train_indices=train,
                            test_duplicates_removed=np.array(removed))
        print(f'{name}: {len(train)} train, {len(test)} test, {len(names)} models, '
              f'{removed} exact train matches removed from official test rows', flush=True)


if __name__ == '__main__':
    main()
