"""Gold drift is the failure that quietly invalidates a text-to-SQL benchmark."""

from __future__ import annotations

from dataclasses import replace

import pytest

from agenteval.gold import GoldCache, GoldError, resolve_gold
from agenteval.scorer import result_hash
from tests.harness_fakes import OK, SYNTAX_ERROR, FakeExecutor, sample_task

GOLD_ROWS = ((99997497,),)
GOLD_HASH = result_hash(("count()",), GOLD_ROWS, ordered=False)


async def test_gold_is_the_result_of_running_the_gold_query() -> None:
    executor = FakeExecutor()

    gold = await resolve_gold(executor, sample_task())

    assert gold.columns == ("count()",)
    assert gold.rows == GOLD_ROWS
    assert executor.executed == ["SELECT count() FROM hits"]


async def test_a_gold_query_that_does_not_run_is_fatal() -> None:
    # Arrange — a broken gold query is a benchmark defect, not a task failure
    executor = FakeExecutor(outcomes=[SYNTAX_ERROR])

    with pytest.raises(GoldError, match="does not run on clickhouse"):
        await resolve_gold(executor, sample_task())


async def test_a_matching_committed_hash_passes() -> None:
    task = replace(sample_task(), gold_result_hash=GOLD_HASH)

    gold = await resolve_gold(FakeExecutor(), task)

    assert gold.rows == GOLD_ROWS


async def test_a_mismatched_committed_hash_stops_the_run() -> None:
    task = replace(sample_task(), gold_result_hash="sha256:stale")

    with pytest.raises(GoldError, match="gold drift on task"):
        await resolve_gold(FakeExecutor(), task)


async def test_order_sensitivity_is_taken_from_the_gold_query() -> None:
    # Arrange — the hash of an ordered result differs from an unordered one, so
    # the tripwire has to agree with the comparison rules
    ordered_sql = "SELECT count() FROM hits ORDER BY 1"
    task = replace(
        sample_task(),
        gold_sql=ordered_sql,
        gold_result_hash=result_hash(("count()",), GOLD_ROWS, ordered=True),
    )

    assert (await resolve_gold(FakeExecutor(), task)).rows == GOLD_ROWS


async def test_the_cache_resolves_each_task_once() -> None:
    # Arrange — every arm must be graded against identical truth
    executor = FakeExecutor()
    cache = GoldCache(executor=executor)
    task = sample_task()

    first = await cache.get(task)
    second = await cache.get(task)

    assert first is second
    assert executor.executed == ["SELECT count() FROM hits"]


async def test_the_cache_keeps_tasks_apart() -> None:
    executor = FakeExecutor(outcomes=[OK, replace(OK, rows=((7,),))])
    cache = GoldCache(executor=executor)

    first = await cache.get(sample_task("a"))
    second = await cache.get(sample_task("b"))

    assert first.rows != second.rows
