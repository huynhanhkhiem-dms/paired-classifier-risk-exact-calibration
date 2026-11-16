"""Compare CPU time for direct and sparse exact null summation."""

import argparse
from math import isqrt
from pathlib import Path
import platform
from statistics import median
from time import thread_time

from paired_exact import coefficient_rows, conditional_pvalue, denominator
from recalibration import balanced_counts, rejection_frontier
from run_experiments import write_rows


ROOT = Path(__file__).resolve().parent
SIZES = ((200, 40), (1000, 80), (4000, 160))


def processor_name():
    info = Path('/proc/cpuinfo')
    if info.is_file():
        for line in info.read_text().splitlines():
            label, _, value = line.partition(':')
            if label.strip() == 'model name' and value.strip():
                return value.strip()
    return platform.processor() or platform.machine()


def dense_counts(frame, sample, score):
    frontier = rejection_frontier(frame, sample, score)
    for q in range(frame // 2, -1, -1):
        chosen, _, zero = coefficient_rows(q, q, frame - 2 * q, sample)
        total = 0
        for a in range(min(q, sample) + 1):
            low = max(0, sample - a - len(zero) + 1)
            high = min(q, sample - a, frontier[a])
            subtotal = sum(chosen[b] * zero[sample - a - b] for b in range(low, high + 1))
            total += chosen[a] * subtotal
        yield total


def median_time(operation, repeats=5, batch=1):
    elapsed = []
    for _ in range(repeats):
        start = thread_time()
        for _ in range(batch):
            result = operation()
        elapsed.append((thread_time() - start) / batch)
    return result, median(elapsed)


def timing_rows():
    for M, n in SIZES:
        m = 3 * n // 5
        a = m // 2 + max(1, isqrt(m) // 2)
        b = m - a
        conditional = lambda: conditional_pvalue.__wrapped__(M, n, a, b)
        score, single = median_time(conditional, batch=500)
        if not (0 < score < 1):
            raise ValueError('Select a nontrivial score for timing')
        sparse = lambda: max(balanced_counts(M, n, score))
        dense = lambda: max(dense_counts(M, n, score))
        assert sparse() == dense()  # Both runs start with warmed coefficient caches.
        batch = 8 if M == SIZES[0][0] else 1
        x, sparse_time = median_time(sparse, batch=batch)
        y, dense_time = median_time(dense, batch=batch)
        assert x == y
        R = denominator(M, n)
        yield {'frame': M, 'sample': n, 'positive_observed': a, 'negative_observed': b,
               'conditional_p': f'{float(score):.12g}', 'calibrated_p': f'{x / R:.12g}',
               'conditional_cpu_ms': f'{1000 * single:.6f}',
               'sparse_cpu_ms': f'{1000 * sparse_time:.6f}',
               'dense_cpu_ms': f'{1000 * dense_time:.6f}',
               'dense_over_sparse': f'{dense_time / sparse_time:.4f}',
               'python': platform.python_version(), 'platform': platform.platform(),
               'cpu': processor_name()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'calibration_times.csv')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_rows(args.output, timing_rows())


if __name__ == '__main__':
    main()
