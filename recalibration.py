"""Unconditional calibration of the finite-population conditional sign score."""

from fractions import Fraction
from functools import lru_cache

from paired_exact import coefficient_rows, conditional_pvalue, denominator


def rejection_frontier(frame, sample, score):
    """Return the largest accepted b at each a, with -1 for an empty row."""
    frontier = []
    b = -1
    for a in range(sample + 1):
        limit = min(a - 1, sample - a)
        b = min(b, limit)
        while b < limit and conditional_pvalue(frame, sample, a, b + 1) <= score:
            b += 1
        frontier.append(b)
    return frontier


def frontier_changes(frontier, sample):
    """Nonzero changes when two zero contrasts become a positive and a negative."""
    changes = []
    for a in range(sample):
        here, after_positive = frontier[a:a + 2]
        if 0 <= here < sample - a:
            changes.append((a, here, sample - 1 - a - here, -1))
        for b in range(here + 1, after_positive + 1):
            changes.append((a, b, sample - 1 - a - b, 1))
    for a in range(sample - 1):
        limit = sample - 2 - a
        before = min(frontier[a], limit)
        after = max(-1, min(frontier[a + 1] - 1, limit))
        direction = 1 if after > before else -1
        for b in range(min(before, after) + 1, max(before, after) + 1):
            changes.append((a, b, sample - 2 - a - b, direction))
    return changes


def balanced_counts(frame, sample, score):
    """Yield exact rejection counts over all balanced populations."""
    if score <= 0 or score >= 1 or sample == frame:
        yield denominator(frame, sample) if score >= 1 else 0
        return
    frontier = rejection_frontier(frame, sample, score)
    changes = frontier_changes(frontier, sample)
    qmax = frame // 2
    plus, _, zero = coefficient_rows(qmax, qmax, frame - 2 * qmax, sample)
    count = 0
    for z in range(min(sample, len(zero) - 1) + 1):
        for a in range(max(0, sample - z - qmax), min(qmax, sample - z) + 1):
            b = sample - z - a
            if b <= frontier[a]:
                count += plus[a] * plus[b] * zero[z]
    yield count
    for q in range(qmax - 1, -1, -1):
        plus, _, zero = coefficient_rows(q, q, frame - 2 * q - 2, sample - 1)
        count -= sum(direction * plus[a] * plus[b] * zero[z] for a, b, z, direction in changes
                     if max(a, b) < len(plus) and z < len(zero))
        yield count


def calibrated_pvalue(frame, sample, a, b):
    """Evaluate the p-value ordered by the conditional sign score."""
    score = conditional_pvalue(frame, sample, a, b)
    return Fraction(max(balanced_counts(frame, sample, score)), denominator(frame, sample))


@lru_cache(maxsize=40000)
def reject_calibrated(frame, sample, a, b, level):
    """Make an integer-level decision without evaluating unnecessary tails."""
    score = conditional_pvalue(frame, sample, a, b)
    if score <= level:
        return True
    threshold = denominator(frame, sample) * level.numerator // level.denominator
    return all(count <= threshold for count in balanced_counts(frame, sample, score))
