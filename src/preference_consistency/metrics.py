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


def majority(items: Iterable[str]) -> str | None:
    counts = Counter(items)
    if not counts:
        return None
    return counts.most_common(1)[0][0]
