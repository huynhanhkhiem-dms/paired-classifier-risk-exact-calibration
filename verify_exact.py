"""Cross-check integer tails, error rates and small multivariate audit worlds."""

import argparse
from collections import Counter
from fractions import Fraction
from itertools import combinations, product
import json
from math import comb
from pathlib import Path

import numpy as np

from paired_exact import binomial_sign_pvalue, conditional_pvalue, profile_pvalue, reject_profile
from recalibration import calibrated_pvalue, reject_calibrated
from run_experiments import ALPHA, design, one_stage, two_stage


ROOT = Path(__file__).resolve().parent


def sample_counts(A, B, C, size):
    return {(a, b): comb(A, a) * comb(B, b) * comb(C, size - a - b)
            for a in range(min(A, size) + 1)
            for b in range(max(0, size - a - C), min(B, size - a) + 1)}


def full_distribution(A, B, C, size):
    cells = sample_counts(A, B, C, size)
    by_score = Counter()
    for (a, b), count in cells.items():
        by_score[a - b] += count
    tail, total = {}, 0
    for score in range(size, -size - 1, -1):
        total += by_score[score]
        tail[score] = total
    return cells, tail


def brute_conditional(frame, size, a, b):
    m = a + b
    values = []
    for L in range(m, frame - size + m + 1):
        for A in range(a, min(L - b, L // 2) + 1):
            B = L - A
            total = sum(comb(A, i) * comb(B, m - i)
                        for i in range(max(a, m - B), min(m, A) + 1))
            values.append(Fraction(total, comb(L, m)))
    return max(values, default=Fraction(0))


def full_null_calibration(frame, size, scores):
    thresholds = sorted(set(scores.values()))
    lookup = {p: i for i, p in enumerate(thresholds)}
    worst = [0] * len(thresholds)
    for A in range(frame // 2 + 1):
        for B in range(A, frame - A + 1):
            frequencies = [0] * len(thresholds)
            for observed, weight in sample_counts(A, B, frame - A - B, size).items():
                frequencies[lookup[scores[observed]]] += weight
            cumulative = 0
            for i, mass in enumerate(frequencies):
                cumulative += mass
                worst[i] = max(worst[i], cumulative)
    total = comb(frame, size)
    return {p: Fraction(worst[i], total) for i, p in enumerate(thresholds)}


def verify_calibration(max_frame):
    checked = 0
    frames = [(m, n) for m in range(max_frame + 1) for n in range(m + 1)]
    frames.extend([(21, 11), (38, 20), (80, 16), (80, 40), (80, 64)])
    for M, n in frames:
        scores = {(a, b): conditional_pvalue(M, n, a, b)
                  for a in range(n + 1) for b in range(n - a + 1)}
        direct = full_null_calibration(M, n, scores)
        for (a, b), p in scores.items():
            if a + b < n:
                assert scores[a + 1, b] <= p
            if b:
                assert scores[a, b - 1] <= p
            want = direct[p]
            assert want <= p
            assert calibrated_pvalue(M, n, a, b) == want
            for level in (ALPHA, ALPHA / 3, ALPHA / 5):
                assert reject_calibrated(M, n, a, b, level) == (want <= level)
            checked += 1
    return {'frame_sample_configurations': len(frames), 'observations': checked,
            'large_frame_sample_pairs': frames[-5:]}


def verify_pairs(max_frame):
    states = 0
    size_checks = 0
    max_size = Fraction(0)
    dense_witness = None
    compatible_dense_witness = None
    for frame in range(max_frame + 1):
        for size in range(frame + 1):
            normalizer = comb(frame, size)
            histograms = {(A, B): full_distribution(A, B, frame - A - B, size)
                          for A in range(frame + 1) for B in range(frame - A + 1)}
            observations = [(a, b) for a in range(size + 1) for b in range(size - a + 1)]
            conditional = {}
            for a, b in observations:
                got = conditional_pvalue(frame, size, a, b)
                assert got == brute_conditional(frame, size, a, b), (frame, size, a, b, got)
                assert got <= binomial_sign_pvalue(a, b)
                conditional[a, b] = got
            for null in (-2, -1, 0, 1, 2):
                worlds = {key: cell for key, cell in histograms.items() if key[0] - key[1] <= null}
                compatible_p, dense_p, compatible_dense_p = {}, {}, {}
                for a, b in observations:
                    score = a - b
                    all_tails = [tails[score] for _, tails in worlds.values()]
                    compatible_tails = [tails[score] for cells, tails in worlds.values() if (a, b) in cells]
                    expected = Fraction(max(compatible_tails, default=0), normalizer)
                    full = Fraction(max(all_tails, default=0), normalizer)
                    assert expected == profile_pvalue(frame, size, a, b, null), (frame, size, a, b, null)
                    assert full == profile_pvalue(frame, size, a, b, null, compatible=False)
                    assert reject_profile(frame, size, a, b, null, ALPHA) == (expected <= ALPHA)
                    compatible_p[a, b] = expected
                    if null == 0:
                        q = frame // 2
                        dense_p[a, b] = Fraction(histograms[q, q][1][score], normalizer)
                        d = min(0, score + frame - size)
                        feasible = [(A, B, tails[score]) for (A, B), (cells, tails) in worlds.items()
                                    if A - B == d and (a, b) in cells]
                        largest = max(feasible, default=(0, 0, 0))
                        compatible_dense_p[a, b] = Fraction(largest[2], normalizer)
                    states += 1
                for (A, B), (cells, _) in worlds.items():
                    error = sum(w for obs, w in cells.items() if compatible_p[obs] <= ALPHA)
                    size_value = Fraction(error, normalizer)
                    assert size_value <= ALPHA
                    max_size = max(size_value, max_size)
                    size_checks += 1
                    if null != 0:
                        continue
                    assert Fraction(sum(w for obs, w in cells.items() if conditional[obs] <= ALPHA), normalizer) <= ALPHA
                    for case, probabilities in enumerate((dense_p, compatible_dense_p)):
                        witness = (dense_witness, compatible_dense_witness)[case]
                        invalid = Fraction(sum(w for obs, w in cells.items()
                                               if probabilities[obs] <= ALPHA), normalizer)
                        if witness is None and invalid > ALPHA:
                            result = {'frame': frame, 'sample': size, 'positive': A, 'negative': B,
                                      'neutral': frame - A - B, 'size': str(invalid),
                                      'profile_size': str(size_value),
                                      'rejection_cells': [list(obs) for obs in cells if probabilities[obs] <= ALPHA]}
                            if case == 0:
                                dense_witness = result
                            else:
                                compatible_dense_witness = result
    _, tail_sparse = full_distribution(6, 6, 3, 4)
    _, tail_dense = full_distribution(7, 7, 1, 4)
    denom = comb(15, 4)
    assert (tail_dense[3], tail_sparse[3], denom) == (70, 75, 1365)
    assert profile_pvalue(15, 4, 3, 0) == Fraction(75, 1365)
    return {'frame_limit': max_frame, 'profile_observation_null_cases': states,
            'profile_world_size_checks': size_checks, 'maximum_profile_size': str(max_size),
            'dense_null_counterexample': dense_witness,
            'largest_compatible_null_level_violation': compatible_dense_witness,
            'interior_tail_maximum': {'frame': 15, 'sample': 4, 'positive_observed': 3,
                                      'negative_observed': 0, 'largest_q_numerator': 70,
                                      'interior_q_numerator': 75, 'denominator': denom}}


def verify_winners():
    predictions = np.array([[0, 1, 1], [0, 2, 1], [1, 1, 2], [2, 0, 2], [2, 0, 1], [2, 1, 0]])
    union, frames = design(predictions)
    checked, tied, draws, censuses = 0, 0, 0, 0
    max_sizes = Counter()
    for labels_tuple in product(range(3), repeat=len(union)):
        labels = np.array(labels_tuple)
        winds = np.count_nonzero(predictions == labels[:, None], axis=0)
        top = np.flatnonzero(winds == winds.max())
        best = int(top[0]) if len(top) == 1 else -1
        tied += int(best < 0)
        for size in (3, 5, 6):
            single_errors, staged_errors = Counter(), 0
            for subset in combinations(union, size):
                ids = list(subset)
                j, decisions = one_stage(predictions[ids], labels[ids], frames)
                for method, issued in decisions.items():
                    single_errors[method] += int(issued and j != best)
                for pilot in ids:
                    order = [pilot] + [idx for idx in ids if idx != pilot]
                    k, issued = two_stage(predictions[order], labels[order], frames, 1)
                    if size == len(union):
                        assert issued == (best >= 0)
                        assert not issued or k == best
                        censuses += 1
                    staged_errors += int(issued and k != best)
                    draws += 1
            audit_count = comb(len(union), size)
            single_errors['pilot_profile'] = Fraction(staged_errors, audit_count * size)
            for method, count in single_errors.items():
                error = count if method == 'pilot_profile' else Fraction(count, audit_count)
                assert error <= ALPHA, (labels_tuple, method, size, error)
                max_sizes[method] = max(max_sizes[method], error)
            checked += 1
    return {'world_budget_cases': checked, 'label_worlds_with_tied_best': tied,
            'staged_audits_enumerated': draws,
            'pilot_census_orders_verified': censuses,
            'maximum_false_certificate_by_method': {method: str(x) for method, x in max_sizes.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-frame', type=int, default=14)
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'verification.json')
    args = parser.parse_args()
    results = {'paired': verify_pairs(args.max_frame), 'recalibration': verify_calibration(args.max_frame),
               'winner': verify_winners()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(results, indent=2) + '\n'
    args.output.write_text(rendered)
    print(rendered)


if __name__ == '__main__':
    main()
