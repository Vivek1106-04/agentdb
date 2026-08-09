"""Confidence intervals and paired comparisons (SPEC §11.4).

A benchmark that reports a point estimate from five seeds and calls one arm
better than another has not measured anything. Every headline number carries a
bootstrap interval, and every claim that one arm beats another is a **paired**
comparison on the same tasks and the same seeds, with a p-value.

Resampling is seeded. A published interval that moves when you re-run the
report is not a published interval.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from random import Random

BOOTSTRAP_RESAMPLES = 10_000
"""SPEC §11.4. Enough that the interval's third digit is stable."""

CONFIDENCE = 0.95

_BOOTSTRAP_SEED = 20260101
"""Fixed so the same traces always produce the same interval."""


class StatsError(ValueError):
    """A statistic was requested that the data cannot support."""


@dataclass(frozen=True, slots=True)
class Interval:
    """A point estimate and its percentile bootstrap interval."""

    mean: float
    low: float
    high: float
    n: int

    def __str__(self) -> str:
        return f"{self.mean:.1%} [{self.low:.1%}, {self.high:.1%}]"


@dataclass(frozen=True, slots=True)
class McNemarResult:
    """Discordant-pair counts and the exact two-sided p-value."""

    only_first: int
    """Tasks the first arm got right and the second did not."""

    only_second: int
    p_value: float

    @property
    def discordant(self) -> int:
        return self.only_first + self.only_second


def bootstrap_mean(
    values: Sequence[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = CONFIDENCE,
) -> Interval:
    """Mean of ``values`` with a percentile bootstrap interval."""
    if not values:
        raise StatsError("cannot bootstrap an empty sample")

    mean = sum(values) / len(values)
    if len(values) == 1:
        return Interval(mean=mean, low=mean, high=mean, n=1)

    rng = Random(_BOOTSTRAP_SEED)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(resamples))
    low, high = _percentiles(means, confidence)
    return Interval(mean=mean, low=low, high=high, n=len(values))


def paired_bootstrap_difference(
    first: Sequence[float],
    second: Sequence[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = CONFIDENCE,
) -> Interval:
    """Interval for ``second - first``, resampling the *pairs*.

    Paired because the arms saw identical tasks at identical seeds: resampling
    them independently would throw away that pairing and widen every interval
    for no reason.
    """
    _require_paired(first, second)

    differences = [b - a for a, b in zip(first, second, strict=True)]
    return bootstrap_mean(differences, resamples=resamples, confidence=confidence)


def mcnemar(first: Sequence[bool], second: Sequence[bool]) -> McNemarResult:
    """Exact McNemar test on per-task correctness.

    Only the pairs the two arms disagree on carry information; agreeing pairs
    say nothing about which arm is better, and counting them is how underpowered
    comparisons get published as wins.
    """
    _require_paired(first, second)

    only_first = sum(1 for a, b in zip(first, second, strict=True) if a and not b)
    only_second = sum(1 for a, b in zip(first, second, strict=True) if b and not a)
    return McNemarResult(
        only_first=only_first,
        only_second=only_second,
        p_value=_exact_binomial_p(only_first, only_second),
    )


def _require_paired(first: Sequence[object], second: Sequence[object]) -> None:
    if len(first) != len(second):
        raise StatsError(
            f"paired comparison needs equal samples, got {len(first)} and {len(second)}"
        )
    if not first:
        raise StatsError("cannot compare empty samples")


def _percentiles(sorted_values: Sequence[float], confidence: float) -> tuple[float, float]:
    tail = (1.0 - confidence) / 2.0
    last = len(sorted_values) - 1
    low = sorted_values[max(0, math.floor(tail * last))]
    high = sorted_values[min(last, math.ceil((1.0 - tail) * last))]
    return low, high


def _exact_binomial_p(only_first: int, only_second: int) -> float:
    """Two-sided exact binomial p at p=0.5 over the discordant pairs."""
    total = only_first + only_second
    if total == 0:
        return 1.0

    smaller = min(only_first, only_second)
    tail = sum(math.comb(total, k) for k in range(smaller + 1)) / float(2**total)
    return min(1.0, 2.0 * tail)
