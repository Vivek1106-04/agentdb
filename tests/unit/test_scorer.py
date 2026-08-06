"""Every comparison rule in SPEC §11.1 gets a test that states the rule."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from agenteval.scorer import (
    GoldResult,
    has_top_level_order_by,
    result_hash,
    results_match,
    score_attempt,
)
from agenteval.systems.base import Attempt, BlindAttempt, EmittedQuery, ModelSpec, TokenUsage
from agenteval.tasks import Task

GOLD = GoldResult(columns=("engine", "visitors"), rows=(("google", 10), ("bing", 5)))


def _task(gold_sql: str = "SELECT a, b FROM t") -> Task:
    return Task(
        id="t1",
        suite="clickbench_nl",
        engines=("clickhouse",),
        question="q",
        gold_sql=gold_sql,
    )


def _query(
    rows: tuple[tuple[object, ...], ...],
    *,
    columns: tuple[str, ...] = ("a", "b"),
    succeeded: bool = True,
    **kwargs: object,
) -> EmittedQuery:
    return EmittedQuery(
        sql="SELECT ...",
        succeeded=succeeded,
        columns=columns,
        rows=rows,
        **kwargs,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------
# order sensitivity
# --------------------------------------------------------------------------


def test_a_top_level_order_by_makes_the_answer_order_sensitive() -> None:
    assert has_top_level_order_by("SELECT a FROM t ORDER BY a DESC") is True


def test_no_order_by_means_order_does_not_matter() -> None:
    assert has_top_level_order_by("SELECT a FROM t GROUP BY a") is False


def test_an_order_by_inside_a_subquery_does_not_order_the_result() -> None:
    sql = "SELECT * FROM (SELECT a FROM t ORDER BY a) x"
    assert has_top_level_order_by(sql) is False


def test_an_order_by_inside_a_window_function_does_not_order_the_result() -> None:
    sql = "SELECT row_number() OVER (ORDER BY a) FROM t"
    assert has_top_level_order_by(sql) is False


def test_an_order_by_after_a_closed_subquery_still_counts() -> None:
    sql = "SELECT * FROM (SELECT a FROM t) x ORDER BY a"
    assert has_top_level_order_by(sql) is True


def test_unbalanced_parentheses_do_not_wedge_the_scan() -> None:
    assert has_top_level_order_by("SELECT a FROM t) ORDER BY a") is True


def test_order_by_in_a_comment_or_string_is_ignored() -> None:
    assert has_top_level_order_by("SELECT a FROM t -- ORDER BY a") is False
    assert has_top_level_order_by("SELECT a FROM t /* ORDER BY a */") is False
    assert has_top_level_order_by("SELECT 'order by a' FROM t") is False


# --------------------------------------------------------------------------
# result comparison
# --------------------------------------------------------------------------


def test_rows_match_regardless_of_order_when_gold_is_unordered() -> None:
    ok, reason = results_match(GOLD, ("x", "y"), (("bing", 5), ("google", 10)), ordered=False)
    assert ok is True
    assert reason is None


def test_row_order_matters_when_gold_orders_its_result() -> None:
    ok, reason = results_match(GOLD, ("x", "y"), (("bing", 5), ("google", 10)), ordered=True)
    assert ok is False
    assert reason is not None
    assert "position 0" in reason


def test_column_names_are_ignored_but_the_count_is_not() -> None:
    ok, _ = results_match(GOLD, ("totally", "different"), GOLD.rows, ordered=True)
    assert ok is True

    ok, reason = results_match(GOLD, ("only_one",), (("google",), ("bing",)), ordered=True)
    assert ok is False
    assert reason is not None
    assert "column count differs" in reason


def test_duplicate_rows_are_significant() -> None:
    gold = GoldResult(columns=("a",), rows=(("x",), ("x",)))
    ok, reason = results_match(gold, ("a",), (("x",),), ordered=False)
    assert ok is False
    assert reason is not None
    assert "row count differs" in reason


def test_floats_compare_at_relative_tolerance() -> None:
    gold = GoldResult(columns=("v",), rows=((1_000_000.0,),))

    within, _ = results_match(gold, ("v",), ((1_000_000.4,),), ordered=True)
    outside, _ = results_match(gold, ("v",), ((1_000_100.0,),), ordered=True)

    assert within is True
    assert outside is False


def test_integers_and_decimals_compare_as_numbers() -> None:
    gold = GoldResult(columns=("v",), rows=((10,),))
    ok, _ = results_match(gold, ("v",), ((Decimal("10.000000"),),), ordered=True)
    assert ok is True


def test_null_equals_null_and_nothing_else() -> None:
    gold = GoldResult(columns=("v",), rows=((None,),))

    same, _ = results_match(gold, ("v",), ((None,),), ordered=True)
    different, _ = results_match(gold, ("v",), ((0,),), ordered=True)

    assert same is True
    assert different is False


def test_nan_compares_equal_to_nan() -> None:
    gold = GoldResult(columns=("v",), rows=((float("nan"),),))
    ok, _ = results_match(gold, ("v",), ((float("nan"),),), ordered=True)
    assert ok is True


def test_nan_does_not_compare_equal_to_a_number() -> None:
    gold = GoldResult(columns=("v",), rows=((float("nan"),),))
    ok, _ = results_match(gold, ("v",), ((1.0,),), ordered=True)
    assert ok is False


def test_dates_bytes_and_booleans_normalize_before_comparison() -> None:
    gold = GoldResult(
        columns=("d", "ts", "b", "flag"),
        rows=((date(2026, 8, 6), datetime(2026, 8, 6, 12, tzinfo=UTC), b"hi", True),),
    )
    ok, _ = results_match(
        gold,
        ("d", "ts", "b", "flag"),
        (("2026-08-06", "2026-08-06T12:00:00+00:00", "hi", True),),
        ordered=True,
    )
    assert ok is True


def test_mixed_type_rows_can_still_be_sorted_for_multiset_comparison() -> None:
    gold = GoldResult(columns=("v",), rows=((None,), (1,), ("a",), (True,)))
    ok, _ = results_match(gold, ("v",), (("a",), (True,), (None,), (1,)), ordered=False)
    assert ok is True


# --------------------------------------------------------------------------
# gold hashing
# --------------------------------------------------------------------------


def test_hash_is_stable_under_row_order_when_the_answer_is_unordered() -> None:
    a = result_hash(("x", "y"), (("google", 10), ("bing", 5)), ordered=False)
    b = result_hash(("p", "q"), (("bing", 5), ("google", 10)), ordered=False)
    assert a == b


def test_hash_distinguishes_row_order_when_the_answer_is_ordered() -> None:
    a = result_hash(("x",), (("google",), ("bing",)), ordered=True)
    b = result_hash(("x",), (("bing",), ("google",)), ordered=True)
    assert a != b


def test_hash_changes_when_the_data_changes() -> None:
    before = result_hash(("x",), (("google", 10),), ordered=False)
    after = result_hash(("x",), (("google", 11),), ordered=False)
    assert before != after


def test_hash_covers_the_special_float_values() -> None:
    digests = {
        result_hash(("v",), ((value,),), ordered=True)
        for value in (float("nan"), float("inf"), float("-inf"), 1.0, None, True, "1")
    }
    assert len(digests) == 7


def test_hash_is_prefixed_so_the_algorithm_is_visible_in_the_task_file() -> None:
    assert result_hash(("v",), ((1,),), ordered=True).startswith("sha256:")


# --------------------------------------------------------------------------
# scoring an attempt
# --------------------------------------------------------------------------


def test_a_correct_final_query_scores_execution_accuracy() -> None:
    # Arrange
    attempt = BlindAttempt(
        task_id="t1",
        seed=0,
        queries=(_query(GOLD.rows, bytes_read=1_024),),
        tokens=TokenUsage(input_tokens=900, output_tokens=100),
        context_bytes=4_096,
        wall_clock_ms=1_500,
    )

    # Act
    score = score_attempt(_task(), attempt, GOLD)

    # Assert
    assert score.verdict == "correct"
    assert score.execution_accuracy is True
    assert score.accuracy_at_1 is True
    assert score.valid_sql is True
    assert score.retries == 0
    assert score.bytes_read == 1_024
    assert score.input_tokens == 900
    assert score.context_bytes == 4_096
    assert score.reason is None


def test_recovery_after_a_failure_is_accurate_but_not_accurate_at_one() -> None:
    # Arrange — the self-correction loop the report reports on
    attempt = BlindAttempt(
        task_id="t1",
        seed=1,
        queries=(
            _query((), succeeded=False, error_class="syntax", error_text="Unknown identifier"),
            _query(GOLD.rows),
        ),
    )

    # Act
    score = score_attempt(_task(), attempt, GOLD)

    # Assert
    assert score.execution_accuracy is True
    assert score.accuracy_at_1 is False
    assert score.retries == 1


def test_a_query_that_ran_but_returned_the_wrong_rows_is_incorrect_not_an_error() -> None:
    attempt = BlindAttempt(
        task_id="t1",
        seed=0,
        queries=(
            _query(
                (("google", 11), ("bing", 5)),
            ),
        ),
    )

    score = score_attempt(_task(), attempt, GOLD)

    assert score.verdict == "incorrect"
    assert score.valid_sql is True
    assert score.reason is not None
    assert "row mismatch" in score.reason


def test_a_failing_final_query_is_an_execution_error_with_its_class_kept() -> None:
    attempt = BlindAttempt(
        task_id="t1",
        seed=0,
        queries=(
            _query((), succeeded=False, error_class="plan_rejection", error_text="not supported"),
        ),
    )

    score = score_attempt(_task(), attempt, GOLD)

    assert score.verdict == "execution_error"
    assert score.valid_sql is False
    assert score.error_class == "plan_rejection"
    assert score.reason is not None
    assert "plan_rejection" in score.reason


def test_a_failure_without_detail_still_produces_a_readable_reason() -> None:
    attempt = BlindAttempt(
        task_id="t1", seed=0, queries=(_query((), succeeded=False, error_class="timeout"),)
    )
    score = score_attempt(_task(), attempt, GOLD)
    assert score.reason is not None
    assert "no detail" in score.reason


def test_emitting_no_query_at_all_is_its_own_verdict() -> None:
    score = score_attempt(_task(), BlindAttempt(task_id="t1", seed=3), GOLD)

    assert score.verdict == "no_query"
    assert score.execution_accuracy is False
    assert score.accuracy_at_1 is False
    assert score.reason == "the system emitted no query"


def test_order_sensitivity_is_taken_from_the_gold_query_and_recorded() -> None:
    ordered_task = _task("SELECT a, b FROM t ORDER BY b DESC")
    attempt = BlindAttempt(task_id="t1", seed=0, queries=(_query((("bing", 5), ("google", 10))),))

    score = score_attempt(ordered_task, attempt, GOLD)

    assert score.order_sensitive is True
    assert score.execution_accuracy is False


# --------------------------------------------------------------------------
# blindness
# --------------------------------------------------------------------------


def test_the_scorer_never_receives_the_system_that_produced_an_attempt() -> None:
    # Arrange
    attempt = Attempt(
        system="S3_clickhouse_agents",
        task_id="t1",
        seed=0,
        model=ModelSpec(provider="anthropic", name="claude-opus-5"),
        queries=(_query(GOLD.rows),),
    )

    # Act
    blind = attempt.blind()

    # Assert — the field is absent from the type the scorer takes, not merely unread
    assert not hasattr(blind, "system")
    assert not hasattr(blind, "model")
    assert score_attempt(_task(), blind, GOLD).execution_accuracy is True


def test_attempts_are_immutable_and_notes_are_appended_to_a_copy() -> None:
    attempt = Attempt(system="S1", task_id="t1", seed=0)

    annotated = attempt.with_note("model chosen by the service")

    assert attempt.notes == ()
    assert annotated.notes == ("model chosen by the service",)


def test_model_spec_renders_for_the_report_table() -> None:
    assert str(ModelSpec(provider="anthropic", name="claude-opus-5")) == ("anthropic/claude-opus-5")


def test_token_usage_totals() -> None:
    assert TokenUsage(input_tokens=10, output_tokens=5).total == 15


def test_attempt_exposes_first_and_final_query() -> None:
    first = _query((), succeeded=False, error_class="syntax")
    final = _query(GOLD.rows)
    blind = BlindAttempt(task_id="t1", seed=0, queries=(first, final))

    assert blind.first_query is first
    assert blind.final_query is final


def test_an_empty_attempt_has_no_queries_to_expose() -> None:
    blind = BlindAttempt(task_id="t1", seed=0)
    assert blind.first_query is None
    assert blind.final_query is None
    assert blind.retries == 0


@pytest.mark.parametrize("succeeded", [True, False])
def test_only_failed_predecessors_count_as_retries(succeeded: bool) -> None:
    blind = BlindAttempt(
        task_id="t1", seed=0, queries=(_query((), succeeded=succeeded), _query(GOLD.rows))
    )
    assert blind.retries == (0 if succeeded else 1)
