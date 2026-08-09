"""Statistics the report publishes. Reproducibility is a tested property."""

from __future__ import annotations

import pytest

from agenteval.stats import (
    StatsError,
    bootstrap_mean,
    mcnemar,
    paired_bootstrap_difference,
)

SMALL = 500  # resamples; the default 10,000 is slow to run in a unit test


def test_a_bootstrap_interval_brackets_the_mean() -> None:
    values = [1.0] * 7 + [0.0] * 3

    interval = bootstrap_mean(values, resamples=SMALL)

    assert interval.mean == pytest.approx(0.7)
    assert interval.low <= interval.mean <= interval.high
    assert interval.n == 10


def test_the_interval_is_reproducible() -> None:
    # A published interval that moves when you re-run the report is not published
    values = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]

    assert bootstrap_mean(values, resamples=SMALL) == bootstrap_mean(values, resamples=SMALL)


def test_a_unanimous_sample_has_a_zero_width_interval() -> None:
    interval = bootstrap_mean([1.0] * 5, resamples=SMALL)

    assert (interval.low, interval.mean, interval.high) == (1.0, 1.0, 1.0)


def test_a_single_observation_reports_itself() -> None:
    interval = bootstrap_mean([0.4])

    assert (interval.low, interval.high, interval.n) == (0.4, 0.4, 1)


def test_an_empty_sample_is_refused() -> None:
    with pytest.raises(StatsError, match="cannot bootstrap an empty sample"):
        bootstrap_mean([])


def test_the_interval_renders_as_percentages() -> None:
    assert str(bootstrap_mean([1.0] * 4, resamples=SMALL)) == "100.0% [100.0%, 100.0%]"


def test_a_wider_confidence_level_gives_a_wider_interval() -> None:
    values = [1.0, 0.0] * 10

    narrow = bootstrap_mean(values, resamples=SMALL, confidence=0.50)
    wide = bootstrap_mean(values, resamples=SMALL, confidence=0.99)

    assert wide.high - wide.low >= narrow.high - narrow.low


def test_the_paired_difference_is_signed_second_minus_first() -> None:
    difference = paired_bootstrap_difference([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], resamples=SMALL)

    assert difference.mean == pytest.approx(1.0)


def test_mismatched_samples_cannot_be_paired() -> None:
    with pytest.raises(StatsError, match="needs equal samples"):
        paired_bootstrap_difference([1.0], [1.0, 0.0])


def test_empty_samples_cannot_be_paired() -> None:
    with pytest.raises(StatsError, match="cannot compare empty samples"):
        paired_bootstrap_difference([], [])


def test_mcnemar_counts_only_the_pairs_that_disagree() -> None:
    # Arrange — four agreeing pairs carry no information about which arm is better
    first = [True, True, False, False, True, False]
    second = [True, True, False, False, False, True]

    result = mcnemar(first, second)

    assert result.only_first == 1
    assert result.only_second == 1
    assert result.discordant == 2


def test_total_agreement_cannot_reject_anything() -> None:
    result = mcnemar([True, False, True], [True, False, True])

    assert result.discordant == 0
    assert result.p_value == 1.0


def test_a_lopsided_disagreement_produces_a_small_p_value() -> None:
    # Arrange — ten tasks the second arm fixed and none it broke
    first = [False] * 10
    second = [True] * 10

    result = mcnemar(first, second)

    assert result.only_second == 10
    assert result.p_value < 0.01


def test_an_even_split_produces_a_p_value_of_one() -> None:
    assert mcnemar([True, False], [False, True]).p_value == pytest.approx(1.0)
