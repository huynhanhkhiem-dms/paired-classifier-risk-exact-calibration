"""Evaluate all certificates on matched label audits of frozen prediction pools."""

import argparse
import csv
import itertools
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

from paired_exact import (binomial_sign_pvalue, conditional_pvalue, reject_marginal,
                          reject_profile)
from recalibration import reject_calibrated


ROOT = Path(__file__).resolve().parent
ALPHA = Fraction(1, 20)
BUDGETS = (20, 40, 80, 160, 320)
PANELS = (3, 5)
DATASETS = ('optdigits', 'pendigits', 'satimage', 'letter', 'dry_bean', 'har')
METHODS = ('two_sided_marginal', 'one_sided_marginal', 'binomial_sign', 'finite_conditional',
           'calibrated_sign', 'full_profile', 'compatible_profile', 'pilot_profile')


def design(predictions):
    disagreement = predictions != predictions[:, [0]]
    union = np.flatnonzero(disagreement.any(axis=1))
    frames = np.count_nonzero(predictions[:, :, None] != predictions[:, None, :], axis=0)
    return union, frames


def ordered_comparisons(predictions, labels, candidate, frames):
    correct = predictions == labels[:, None]
    for rival in range(predictions.shape[1]):
        if candidate == rival:
            continue
        a = int(np.count_nonzero(correct[:, candidate] & ~correct[:, rival]))
        b = int(np.count_nonzero(correct[:, rival] & ~correct[:, candidate]))
        in_frame = int(np.count_nonzero(predictions[:, candidate] != predictions[:, rival]))
        yield int(frames[candidate, rival]), in_frame, a, b, rival


def strict_leader(predictions, labels):
    wins = (predictions == labels[:, None]).sum(axis=0)
    winners = np.flatnonzero(wins == wins.max())
    return int(winners[0]) if len(winners) == 1 else -1


def certain_advantage(comparisons):
    return all(a - b > M - n for M, n, a, b, _ in comparisons)


def one_stage(predictions, labels, frames):
    k = predictions.shape[1]
    candidate = strict_leader(predictions, labels)
    if candidate < 0:
        return -1, dict.fromkeys(METHODS[:-1], False)
    level = ALPHA / k
    interval_tail = ALPHA / (4 * (k * (k - 1) // 2))
    comparisons = sorted(ordered_comparisons(predictions, labels, candidate, frames),
                         key=lambda row: (row[2] - row[3]) / max(1, row[1]))
    if certain_advantage(comparisons):
        return candidate, dict.fromkeys(METHODS[:-1], True)
    decisions = {
        'two_sided_marginal': all(reject_marginal(M, n, a, b, 0, interval_tail) for M, n, a, b, _ in comparisons),
        'one_sided_marginal': all(reject_marginal(M, n, a, b, 0, level / 2) for M, n, a, b, _ in comparisons),
        'binomial_sign': all(binomial_sign_pvalue(a, b) <= level for _, _, a, b, _ in comparisons),
        'finite_conditional': all(conditional_pvalue(M, n, a, b) <= level for M, n, a, b, _ in comparisons),
        'calibrated_sign': all(reject_calibrated(M, n, a, b, level) for M, n, a, b, _ in comparisons),
        'full_profile': all(reject_profile(M, n, a, b, 0, level, False) for M, n, a, b, _ in comparisons),
        'compatible_profile': all(reject_profile(M, n, a, b, 0, level) for M, n, a, b, _ in comparisons),
    }
    return candidate, decisions


def two_stage(predictions, labels, frames, prefix):
    leader = strict_leader(predictions, labels)
    if leader >= 0 and certain_advantage(ordered_comparisons(predictions, labels, leader, frames)):
        return leader, True
    candidate = int(np.argmax((predictions[:prefix] == labels[:prefix, None]).sum(axis=0)))
    early = list(ordered_comparisons(predictions[:prefix], labels[:prefix], candidate, frames))
    later = list(ordered_comparisons(predictions[prefix:], labels[prefix:], candidate, frames))
    checks = [(M - n0, n1, a1, b1, b0 - a0)
              for (M, n0, a0, b0, _), (_, n1, a1, b1, _) in zip(early, later)]
    checks.sort(key=lambda row: (row[2] - row[3] - row[4]) / max(1, row[1]))
    return candidate, all(reject_profile(M, n, a, b, null, ALPHA) for M, n, a, b, null in checks)


def assess_audit(predictions, labels, frames):
    candidate, decisions = one_stage(predictions, labels, frames)
    selected = {method: candidate if issued else -1 for method, issued in decisions.items()}
    prefix = min(len(labels), max(1, len(labels) // 5))
    candidate, issued = two_stage(predictions, labels, frames, prefix)
    selected['pilot_profile'] = candidate if issued else -1
    return selected


def pool_rows(name, pool_file, repetitions):
    with np.load(pool_file, allow_pickle=False) as pool:
        labels = pool['labels']
        predictions = pool['predictions']
        model_names = pool['models']
    union, frames = design(predictions)
    actual = (predictions == labels[:, None]).sum(axis=0)
    truths = {}
    for k in PANELS:
        best = np.flatnonzero(actual[:k] == actual[:k].max())
        scores = np.sort(actual[:k])
        truths[k] = (int(best[0]) if len(best) == 1 else -1, int(scores[-1] - scores[-2]))
    for budget in BUDGETS:
        start = time.monotonic()
        rng = np.random.default_rng(np.random.SeedSequence([20261116, DATASETS.index(name), budget]))
        for repetition in range(repetitions):
            size = min(budget, len(union))
            sample = rng.choice(union, size, replace=False, shuffle=True)
            measured = labels[sample]
            for k in PANELS:
                truth, gap = truths[k]
                for method, selected in assess_audit(predictions[sample, :k], measured, frames[:k, :k]).items():
                    yield {
                        'dataset': name, 'models': k, 'pool': len(labels), 'union': len(union),
                        'true_best': model_names[truth] if truth >= 0 else '', 'true_gap_count': gap,
                        'budget': budget, 'repetition': repetition, 'labels': size, 'method': method,
                        'selected': model_names[selected] if selected >= 0 else '',
                        'wrong': int(selected >= 0 and selected != truth),
                    }
        print(f'{name} budget {budget}: {repetitions} draws in {time.monotonic() - start:.2f}s', flush=True)


def write_rows(path, rows):
    iterator = iter(rows)
    first = next(iterator)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(first), lineterminator='\n')
        writer.writeheader()
        writer.writerows(itertools.chain((first,), iterator))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pools', type=Path, default=ROOT / 'pools')
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'audit_trials.csv')
    parser.add_argument('--repetitions', type=int, default=160)
    parser.add_argument('--dataset', choices=DATASETS)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error('At least one repetition is required')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    names = (args.dataset,) if args.dataset else DATASETS
    write_rows(args.output, (row for name in names
                            for row in pool_rows(name, args.pools / f'{name}.npz', args.repetitions)))


if __name__ == '__main__':
    main()
