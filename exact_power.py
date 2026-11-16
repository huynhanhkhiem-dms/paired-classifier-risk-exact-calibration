"""Enumerate exact rejection probabilities and label counts at a fixed power target."""

import argparse
from fractions import Fraction
from functools import lru_cache
from math import comb
from pathlib import Path

from paired_exact import (binomial_sign_pvalue, conditional_pvalue, reject_marginal,
                          reject_profile)
from recalibration import balanced_counts
from run_experiments import ALPHA, write_rows
from verify_exact import sample_counts


ROOT = Path(__file__).resolve().parent
FRAME = 80
NEUTRAL = (0, 40, 64)
MARGINS = (0, 4, 8)
EXCESS = {0: (0, 2, 8, 16), 4: (0, 2, 8), 8: (0, 2, 8)}


@lru_cache(maxsize=None)
def calibrated_cutoff(n):
    scores = sorted({conditional_pvalue(FRAME, n, a, b)
                     for a in range(n + 1) for b in range(n - a + 1)})
    target = comb(FRAME, n) * ALPHA.numerator // ALPHA.denominator
    lo, hi = -1, len(scores)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if all(count <= target for count in balanced_counts(FRAME, n, scores[mid])):
            lo = mid
        else:
            hi = mid
    return scores[lo] if lo >= 0 else Fraction(-1)


@lru_cache(maxsize=200000)
def rejected(n, a, b, margin):
    certain = a - b - (FRAME - n) > margin
    decisions = {
        'one_sided_marginal': certain or reject_marginal(FRAME, n, a, b, margin, ALPHA / 2),
        'full_profile': certain or reject_profile(FRAME, n, a, b, margin, ALPHA, False),
        'compatible_profile': certain or reject_profile(FRAME, n, a, b, margin, ALPHA),
    }
    if margin == 0:
        p = conditional_pvalue(FRAME, n, a, b)
        decisions.update(binomial_sign=certain or binomial_sign_pvalue(a, b) <= ALPHA,
                         finite_conditional=p <= ALPHA,
                         calibrated_sign=p <= calibrated_cutoff(n))
    return tuple(method for method, decision in decisions.items() if decision)


def power_rows():
    for n in range(1, FRAME + 1):
        denominator = comb(FRAME, n)
        for margin in MARGINS:
            methods = ['one_sided_marginal', 'full_profile', 'compatible_profile']
            if margin == 0:
                methods.extend(['binomial_sign', 'finite_conditional', 'calibrated_sign'])
            for excess in EXCESS[margin]:
                total = margin + excess
                for C in NEUTRAL:
                    A = (FRAME - C + total) // 2
                    B = FRAME - C - A
                    assert A - B == total
                    rejection_counts = dict.fromkeys(methods, 0)
                    for (a, b), mass in sample_counts(A, B, C, n).items():
                        for method in rejected(n, a, b, margin):
                            rejection_counts[method] += mass
                    for method, count in rejection_counts.items():
                        value = Fraction(count, denominator)
                        yield {'frame': FRAME, 'positive': A, 'negative': B, 'neutral': C,
                               'margin': margin, 'excess': excess, 'sample': n, 'method': method,
                               'probability': format(float(value), '.12f'), 'exact_fraction': str(value)}


def size_rows():
    for n in range(1, FRAME + 1):
        denominator = comb(FRAME, n)
        cutoff = calibrated_cutoff(n)
        base = Fraction(max(balanced_counts(FRAME, n, ALPHA)), denominator)
        calibrated = Fraction(max(balanced_counts(FRAME, n, cutoff)), denominator)
        assert base <= calibrated <= ALPHA
        yield {'frame': FRAME, 'sample': n, 'conditional_size': str(base),
               'calibrated_size': str(calibrated), 'calibrated_score_cutoff': str(cutoff)}


def first_budgets(rows):
    first = {}
    for row in rows:
        if row['excess'] == 0:
            continue
        key = row['margin'], row['excess'], row['neutral'], row['method']
        first.setdefault(key, None)
        if first[key] is None and Fraction(row['exact_fraction']) >= Fraction(1, 2):
            first[key] = row['sample']
    for (margin, excess, neutral, method), sample in sorted(first.items()):
        yield {'frame': FRAME, 'margin': margin, 'excess': excess, 'neutral': neutral,
               'method': method, 'power_target': '1/2', 'first_label_count': sample}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    opts = parser.parse_args()
    opts.output.mkdir(parents=True, exist_ok=True)
    rows = list(power_rows())
    write_rows(opts.output / 'exact_power.csv', rows)
    write_rows(opts.output / 'exact_size.csv', size_rows())
    write_rows(opts.output / 'label_requirements.csv', first_budgets(rows))


if __name__ == '__main__':
    main()
