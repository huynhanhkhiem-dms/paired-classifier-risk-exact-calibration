"""Build the manuscript's reported values, tables, and power figure."""

import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from run_experiments import DATASETS, METHODS, PANELS


ROOT = Path(__file__).resolve().parent
DATA_LABELS = dict(zip(DATASETS, ('Optdigits', 'Pendigits', 'Satimage', 'Letter', 'Dry Bean', 'HAR')))
METHOD_LABELS = dict(zip(METHODS, ('Two-sided marginal', 'One-sided marginal', 'Binomial sign',
                                 'Finite conditional', 'Recalibrated sign', 'Unrestricted profile',
                                 'Compatible profile', 'Pilot-selected profile')))


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def audit_summary(results):
    counts = defaultdict(Counter)
    tasks = {}
    decisions = defaultdict(dict)
    total_labels = {}
    for row in read_csv(results / 'audit_trials.csv'):
        task = row['dataset']
        k = int(row['models'])
        budget = int(row['budget'])
        method = row['method']
        draw = task, budget, int(row['repetition'])
        panel = (*draw, k)
        if method in decisions[panel]:
            raise ValueError(f'Duplicate audit row: {panel}, {method}')
        decisions[panel][method] = row['selected']
        size = int(row['labels'])
        if draw in total_labels and total_labels[draw] != size:
            raise ValueError(f'Mismatched label count at {draw}')
        total_labels[draw] = size
        tasks[task, k] = tuple(int(row[key]) for key in ('pool', 'union', 'true_gap_count'))
        issued = bool(row['selected'])
        wrong = int(row['wrong'])
        correct = int(issued) - wrong
        if wrong != int(issued and row['selected'] != row['true_best']):
            raise ValueError(f'Wrong-decision field inconsistent at {panel}')
        for key in ((method,), (method, k), (method, task, k, budget)):
            counts[key].update(draws=1, correct=correct, wrong=wrong, abstain=not issued)
    for panel, choices in decisions.items():
        if set(choices) != set(METHODS):
            raise ValueError(f'Missing method at {panel}')
        if choices['finite_conditional'] and choices['finite_conditional'] != choices['calibrated_sign']:
            raise ValueError(f'Recalibration lost an earlier decision at {panel}')
    return tasks, counts, len(total_labels), len(decisions)


def command(name, body):
    return '\\newcommand{\\' + name + '}{' + str(body) + '}'


def table_command(name, rows):
    return command(name, '%\n' + '\n'.join('  ' + ' & '.join(map(str, r)) + r' \\' for r in rows) + '\n')


def formatted_int(value):
    return f'{int(value):,}'


def build_tex(results, output):
    tasks, counts, draws, panels = audit_summary(results)
    power = {(int(r['margin']), int(r['excess']), int(r['neutral']), int(r['sample']), r['method']):
             Fraction(r['exact_fraction']) for r in read_csv(results / 'exact_power.csv')}
    labels = {(int(r['margin']), int(r['excess']), int(r['neutral']), r['method']):
              r['first_label_count'] for r in read_csv(results / 'label_requirements.csv')}
    sizes = {int(r['sample']): r for r in read_csv(results / 'exact_size.csv')}
    timing = read_csv(results / 'calibration_times.csv')
    sim = {(int(r['true_gap_count']), int(r['budget']), int(r['models']), r['method']):
           r for r in read_csv(results / 'matched_panels.csv')}
    tasks_table = []
    for name in DATASETS:
        n, u, gap3 = tasks[name, 3]
        gap5 = tasks[name, 5][2]
        tasks_table.append((DATA_LABELS[name], formatted_int(n), formatted_int(u), gap3, gap5))
    audit_table = []
    for method in METHODS:
        all_tasks = counts[method,]
        audit_table.append((METHOD_LABELS[method], formatted_int(counts[method, 3]['correct']),
                            formatted_int(counts[method, 5]['correct']), formatted_int(all_tasks['wrong']),
                            formatted_int(all_tasks['abstain'])))
    timing_table = [(formatted_int(r['frame']), r['sample'], f"{float(r['conditional_cpu_ms']):.4f}",
                     f"{float(r['sparse_cpu_ms']):.2f}", f"{float(r['dense_cpu_ms']):.2f}",
                     f"{float(r['dense_over_sparse']):.1f}") for r in timing]
    outputs = [table_command('PoolRows', tasks_table), table_command('AuditRows', audit_table),
               table_command('TimingRows', timing_table)]
    for name, value in (('LabelDraws', draws), ('PanelAudits', panels), ('AuditsPerPanel', panels // 2),
                        ('CondCorrect', counts['finite_conditional',]['correct']),
                        ('CalCorrect', counts['calibrated_sign',]['correct']),
                        ('CondWrong', counts['finite_conditional',]['wrong']),
                        ('CalWrong', counts['calibrated_sign',]['wrong']),
                        ('PilotCorrect', counts['pilot_profile',]['correct']),
                        ('PilotWrong', counts['pilot_profile',]['wrong']),
                        ('TrialRepetitions', counts['calibrated_sign', DATASETS[0], PANELS[0], 20]['draws'])):
        outputs.append(command(name, formatted_int(value)))
    ratios = [float(r['dense_over_sparse']) for r in timing]
    outputs.append(command('TimingRange', f'{min(ratios):.1f}--{max(ratios):.1f}'))
    for name, key in (
        ('CondPower', (0, 16, 64, 16, 'finite_conditional')),
        ('CalPower', (0, 16, 64, 16, 'calibrated_sign')),
        ('CondNearPower', (0, 2, 40, 64, 'finite_conditional')),
        ('CalNearPower', (0, 2, 40, 64, 'calibrated_sign')),
        ('FullMarginPower', (4, 8, 40, 64, 'full_profile')),
        ('CompatMarginPower', (4, 8, 40, 64, 'compatible_profile')),
        ('MarginalMarginPower', (4, 8, 40, 64, 'one_sided_marginal')),
    ):
        outputs.append(command(name, f'{100 * float(power[key]):.2f}'))
    for name, method in (('CondLabels', 'finite_conditional'), ('CalLabels', 'calibrated_sign')):
        outputs.append(command(name, labels[0, 16, 64, method]))
    for name, key in (('CondSize', 'conditional_size'), ('CalSize', 'calibrated_size')):
        outputs.append(command(name, f'{100 * float(Fraction(sizes[16][key])):.2f}'))
    for name, k in zip(('SimThree', 'SimFive', 'SimEight'), (3, 5, 8)):
        outputs.append(command(name, sim[24, 160, k, 'calibrated_sign']['correct']))
    outputs.append(command('SimRepetitions', sim[24, 160, 3, 'calibrated_sign']['draws']))
    (output / 'Results.tex').write_text('\n\n'.join(outputs) + '\n')


def draw_power(results, output):
    rows = read_csv(results / 'exact_power.csv')
    styles = {
        'binomial_sign': ('#9a4319', ':', '^'),
        'finite_conditional': ('#2456a0', '--', 'o'),
        'calibrated_sign': ('#087267', '-', 'D'),
        'one_sided_marginal': ('#5d626e', ':', '^'),
        'full_profile': ('#783f91', '--', 'o'),
        'compatible_profile': ('#087267', '-', 'D'),
    }
    panels = [
        (0, 16, 40, ('binomial_sign', 'finite_conditional', 'calibrated_sign'), r'Zero margin, $\Gamma=16$'),
        (4, 8, 40, ('one_sided_marginal', 'full_profile', 'compatible_profile'), r'Margin $d=4$, $\Gamma=12$'),
    ]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.0, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    figure, axes = plt.subplots(1, 2, figsize=(6.6, 3.45), sharey=True)
    figure.subplots_adjust(left=.115, right=.985, top=.88, bottom=.335, wspace=.19)
    for axis, (margin, excess, neutral, methods, title) in zip(axes, panels):
        for method in methods:
            xs, ys = zip(*sorted((int(r['sample']), float(r['probability'])) for r in rows
                                 if (int(r['margin']), int(r['excess']), int(r['neutral']), r['method'])
                                 == (margin, excess, neutral, method)))
            color, dash, marker = styles[method]
            axis.plot(xs, ys, label=METHOD_LABELS[method], color=color, linestyle=dash,
                      lw=1.35, marker=marker, markevery=(7, 13), markersize=4, fillstyle='none')
        axis.set_title(title, fontsize=10, pad=8)
        axis.set_xlim(1, 82)
        axis.set_ylim(-.02, 1.02)
        axis.set_xticks((20, 40, 60, 80))
        axis.set_yticks((0, .25, .5, .75, 1))
        axis.grid(axis='y', alpha=.3, lw=.5)
        axis.legend(loc='upper center', bbox_to_anchor=(.53, -.25), fontsize=9.6,
                    frameon=False, ncol=1, handlelength=2.5, borderaxespad=0.)
    axes[0].set_ylabel('Rejection probability')
    figure.text(.55, .215, 'Number of sampled labels', ha='center', va='center', fontsize=9.8)
    figure.savefig(output / 'Fig1.pdf', metadata={'CreationDate': None, 'ModDate': None})
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'results')
    parser.add_argument('--output', type=Path, default=ROOT / 'manuscript_assets')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    build_tex(args.results, args.output)
    draw_power(args.results, args.output)


if __name__ == '__main__':
    main()
