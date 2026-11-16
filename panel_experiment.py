"""Vary shortlist size on matched simulated audits of the same finite pool."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from run_experiments import METHODS, assess_audit, write_rows


ROOT = Path(__file__).resolve().parent
FRAME = 240
MODELS = (3, 5, 8)
BUDGETS = (32, 80, 160)
GAPS = (0, 2, 8, 24)


def panel_rows(repetitions):
    for gap in GAPS:
        counts = [84 + gap // 2, 84 - gap // 2] + [12] * 6
        labels = np.repeat(np.arange(8), counts)
        truth = 0 if gap > 0 else -1
        assert len(labels) == FRAME
        for budget in BUDGETS:
            rng = np.random.default_rng(np.random.SeedSequence([20261116, FRAME, gap, budget]))
            totals = Counter()
            for _ in range(repetitions):
                draw = labels[rng.choice(FRAME, budget, replace=False, shuffle=True)]
                for k in MODELS:
                    predictions = np.broadcast_to(np.arange(k), (budget, k))
                    frames = FRAME * (1 - np.eye(k, dtype=np.int64))
                    for method, selected in assess_audit(predictions, draw, frames).items():
                        totals[k, method, 'issued'] += selected >= 0
                        totals[k, method, 'wrong'] += selected >= 0 and selected != truth
            for k in MODELS:
                for method in METHODS:
                    issued, wrong = (totals[k, method, key] for key in ('issued', 'wrong'))
                    yield {'frame': FRAME, 'true_gap_count': gap, 'budget': budget,
                           'models': k, 'draws': repetitions, 'method': method,
                           'correct': issued - wrong, 'wrong': wrong,
                           'abstained': repetitions - issued}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repetitions', type=int, default=800)
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'matched_panels.csv')
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error('At least one repetition is required')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_rows(args.output, panel_rows(args.repetitions))


if __name__ == '__main__':
    main()
