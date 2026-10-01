"""Agreement, flip rate, Cohen's kappa, Wilson intervals."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass
class Rate:
    k: int
    n: int
    point: float
    ci_low: float
    ci_high: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "k": self.k,
            "n": self.n,
            "point": self.point,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
        }


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = (z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def rate(k: int, n: int) -> Rate:
    if n <= 0:
        return Rate(k=k, n=n, point=float("nan"), ci_low=float("nan"), ci_high=float("nan"))
    lo, hi = wilson_interval(k, n)
    return Rate(k=k, n=n, point=k / n, ci_low=lo, ci_high=hi)


def cohens_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    if len(a) != len(b) or not a:
        return float("nan")
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    cats = sorted(set(a) | set(b))
    pe = 0.0
    for c in cats:
        pe += (sum(x == c for x in a) / n) * (sum(y == c for y in b) / n)
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def agreement(a: Sequence[str], b: Sequence[str]) -> Rate:
    n = min(len(a), len(b))
    k = sum(x == y for x, y in zip(a[:n], b[:n]))
    return rate(k, n)


def fmt_rate(r: Rate) -> str:
    if r.n == 0 or math.isnan(r.point):
        return "n/a"
    return f"{r.point:.3f} ({r.k}/{r.n}) 95% CI [{r.ci_low:.3f}, {r.ci_high:.3f}]"


def dual_order_consistency(
    base_content: Sequence[str],
    swap_content: Sequence[str],
    base_correct: Sequence[bool],
) -> dict[str, Rate | float | int]:
    """Zheng-style swap: trust a winner only when both orders agree on content.

    `base_correct` is aligned with `base_content`. Accuracy on the consistent
    subset uses the baseline judgment (the swap judgment names the same content).
    """
    n = len(base_content)
    if n != len(swap_content) or n != len(base_correct):
        raise ValueError("dual-order inputs must be the same length")
    agree_k = sum(a == b for a, b in zip(base_content, swap_content))
    consistent_correct = [
        bool(correct)
        for correct, left, right in zip(base_correct, base_content, swap_content)
        if left == right
    ]
    n_consistent = len(consistent_correct)
    return {
        "agreement": rate(agree_k, n),
        "flip": rate(n - agree_k, n) if n else rate(0, 0),
        "kappa": cohens_kappa(list(base_content), list(swap_content)),
        "n_consistent": n_consistent,
        "accuracy_consistent": rate(sum(consistent_correct), n_consistent),
    }


def slot_attraction(baseline: Sequence[str], other: Sequence[str], slot: str) -> Rate:
    """Of cases baseline did not pick `slot`, the fraction `other` does.

    This is the lift of a position or verbosity manipulation onto one letter,
    ignoring cases that were already on that letter.
    """
    if len(baseline) != len(other):
        raise ValueError("slot_attraction inputs must be the same length")
    eligible = [(b, o) for b, o in zip(baseline, other) if b != slot]
    k = sum(o == slot for _, o in eligible)
    return rate(k, len(eligible))


def majority(items: Iterable[str]) -> str | None:
    counts = Counter(items)
    if not counts:
        return None
    return counts.most_common(1)[0][0]
