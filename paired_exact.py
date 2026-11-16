"""Exact directional tests for ternary contrasts in a simple random sample."""

from functools import lru_cache
from fractions import Fraction
from math import comb


@lru_cache(maxsize=4096)
def binomial_row(size, capacity):
    """Return integer binomial coefficients through the requested capacity."""
    row = [1]
    for j in range(1, min(size, capacity) + 1):
        row.append(row[-1] * (size - j + 1) // j)
    return row


def coefficient_rows(positive, negative, neutral, sample):
    capacity = (sample + 31) // 32 * 32
    return tuple(binomial_row(count, capacity) for count in (positive, negative, neutral))


def mass_count(rows, sample, score):
    """Count subsets of the indicated size with exactly the indicated signed sum."""
    plus, minus, zero = rows
    minimum = max(0, -score, (sample - score - len(zero) + 2) // 2)
    maximum = min(len(minus) - 1, len(plus) - 1 - score, (sample - score) // 2)
    return sum(plus[b + score] * minus[b] * zero[sample - 2 * b - score]
               for b in range(minimum, maximum + 1))


def tail_count(rows, sample, score):
    """Count subsets of the indicated size with signed sum at least score."""
    plus, minus, zero = rows
    count = 0
    for z in range(max(0, sample - len(plus) - len(minus) + 2), min(len(zero) - 1, sample) + 1):
        remaining = sample - z
        lower = max(0, remaining - len(minus) + 1, (remaining + score + 1) // 2)
        upper = min(len(plus) - 1, remaining)
        count += zero[z] * sum(plus[a] * minus[remaining - a] for a in range(lower, upper + 1))
    return count


def boundary_range(frame, sample, a, b, null, compatible=True):
    """Return the boundary total and admitted nuisance interval, or an empty range."""
    if not (0 <= a and 0 <= b and a + b <= sample <= frame):
        raise ValueError('Invalid frame or sample counts')
    score = a - b
    unseen = frame - sample
    if compatible:
        if score - unseen > null:
            return 0, 1, 0
        total = min(null, score + unseen)
        minimum = max(0, a - max(total, 0), b - max(-total, 0))
        maximum = (frame - abs(total) - sample + a + b) // 2
    else:
        if null < -frame:
            return 0, 1, 0
        total = min(frame, null)
        minimum = 0
        maximum = (frame - abs(total)) // 2
    return total, minimum, maximum


def boundary_tails(frame, sample, score, total, minimum, maximum):
    """Yield (q, numerator) in descending order using the adjacent-count identity."""
    if minimum > maximum:
        return
    if score <= -sample:
        count = comb(frame, sample)
        for q in range(maximum, minimum - 1, -1):
            yield q, count
        return
    up, down = max(total, 0), max(-total, 0)
    rows = coefficient_rows(maximum + up, maximum + down, frame - abs(total) - 2 * maximum, sample)
    count = tail_count(rows, sample, score)
    yield maximum, count
    for q in range(maximum - 1, minimum - 1, -1):
        if min(sample, q + up, total + frame - sample) < score:
            return  # All the remaining tails are zero.
        rows = coefficient_rows(q + up, q + down, frame - abs(total) - 2 * q - 2, sample - 1)
        count += mass_count(rows, sample - 1, score) - mass_count(rows, sample - 1, score - 1)
        yield q, count


@lru_cache(maxsize=1024)
def denominator(frame, sample):
    return comb(frame, sample)


def hypergeometric_tail_count(frame, positive, sample, threshold):
    plus, _, rest = coefficient_rows(positive, 0, frame - positive, sample)
    low = max(0, threshold, sample - len(rest) + 1)
    high = min(len(plus) - 1, sample)
    return sum(plus[a] * rest[sample - a] for a in range(low, high + 1))


def profile_pvalue(frame, sample, a, b, null=0, *, compatible=True):
    """Upper-tail p-value for a finite-population signed total at most null."""
    total, low, high = boundary_range(frame, sample, a, b, null, compatible)
    if frame == sample:
        return Fraction(int(a - b <= null), 1)
    count = max((count for _, count in boundary_tails(frame, sample, a - b, total, low, high)), default=0)
    return Fraction(count, denominator(frame, sample))


@lru_cache(maxsize=100000)
def reject_profile(frame, sample, a, b, null, level, compatible=True):
    """Compare integer tail counts to a rational level, with early nonrejection."""
    total, low, high = boundary_range(frame, sample, a, b, null, compatible)
    if frame == sample:
        return a - b > null
    if compatible and a - b + frame - sample <= null:
        return False
    if low > high:
        return True
    bound = denominator(frame, sample) * level.numerator // level.denominator
    for _, count in boundary_tails(frame, sample, a - b, total, low, high):
        if count > bound:
            return False
    return True


@lru_cache(maxsize=100000)
def lower_count(frame, sample, observed, tail):
    """Invert a one-sided hypergeometric upper tail for the success total."""
    lo, hi = observed, frame - sample + observed
    target = denominator(frame, sample) * tail.numerator
    while lo < hi:
        mid = (lo + hi) // 2
        if hypergeometric_tail_count(frame, mid, sample, observed) * tail.denominator >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def reject_marginal(frame, sample, a, b, null, tail):
    lower = lower_count(frame, sample, a, tail)
    upper = frame - lower_count(frame, sample, sample - b, tail)
    return lower - upper > null


@lru_cache(maxsize=100000)
def conditional_pvalue(frame, sample, a, b):
    """Maximize the sign upper tail conditional on the number of nonzero draws."""
    if a <= b:
        return Fraction(1)
    nonzero = a + b
    pairs = (frame - sample + nonzero) // 2
    if pairs < a:
        return Fraction(0)
    count = hypergeometric_tail_count(2 * pairs, pairs, nonzero, a)
    return Fraction(count, denominator(2 * pairs, nonzero))


@lru_cache(maxsize=10000)
def binomial_sign_pvalue(a, b):
    if a <= b:
        return Fraction(1)
    nonzero = a + b
    row = binomial_row(nonzero, nonzero)
    return Fraction(sum(row[a:]), 2 ** nonzero)
