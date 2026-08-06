"""The fake adapter is test infrastructure the whole suite leans on; keep it honest."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from agentdb.adapters import (
    Capability,
    ExplainMode,
    Limits,
    RawPlan,
    RelationRef,
    ResultSet,
    SamplePolicy,
    TimeWindow,
    WorkloadEntry,
)
from tests.fakes import FakeAdapter, clickhouse_hits_fixture

HITS = RelationRef(namespace="agentdb", name="hits")


async def test_fixture_models_a_clickbench_shaped_table() -> None:
    # Arrange
    adapter = clickhouse_hits_fixture()

    # Act
    layout = await adapter.physical_layout(HITS)
    detail = await adapter.describe_relation(HITS)

    # Assert
    assert layout.leading_sort_column == "CounterID"
    assert layout.approx_rows == 99_997_497
    assert "URL" in detail.column_names
    assert adapter.supports(Capability.SORT_KEY) is True


async def test_list_relations_filters_by_namespace() -> None:
    adapter = clickhouse_hits_fixture()

    assert len(await adapter.list_relations()) == 1
    assert len(await adapter.list_relations("agentdb")) == 1
    assert await adapter.list_relations("other") == []


async def test_column_profile_records_the_sample_policy_core_used() -> None:
    # Arrange
    adapter = clickhouse_hits_fixture()
    policy = SamplePolicy(fraction=0.01, max_rows=1_000_000, timeout_s=30)

    # Act
    profiles = await adapter.column_profile(HITS, ["SearchEngineID", "missing"], policy)

    # Assert — unknown columns are skipped, and the policy is auditable
    assert [p.name for p in profiles] == ["SearchEngineID"]
    assert adapter.calls_named("column_profile") == [(HITS, ("SearchEngineID", "missing"), policy)]


async def test_explain_echoes_the_sql_and_mode_it_was_asked_for() -> None:
    adapter = clickhouse_hits_fixture()
    adapter.plan = RawPlan(
        engine="clickhouse", mode=ExplainMode.ESTIMATE, sql="", payload="Expression"
    )

    plan = await adapter.explain("SELECT count() FROM hits", ExplainMode.PIPELINE)

    assert plan.sql == "SELECT count() FROM hits"
    assert plan.mode is ExplainMode.PIPELINE


async def test_execute_returns_the_scripted_result() -> None:
    adapter = clickhouse_hits_fixture()
    adapter.result = ResultSet(columns=("n",), rows=((1,),), row_count=1, truncated=False)

    result = await adapter.execute("SELECT 1", Limits(timeout_s=5, max_result_rows=10))

    assert result.rows == ((1,),)


async def test_workload_respects_top_n() -> None:
    adapter = clickhouse_hits_fixture()
    adapter.workload_entries = (
        WorkloadEntry(normalized_sql="SELECT ?", calls=3),
        WorkloadEntry(normalized_sql="SELECT ??", calls=2),
    )
    window = TimeWindow(
        start=datetime(2026, 8, 6, tzinfo=UTC), end=datetime(2026, 8, 7, tzinfo=UTC)
    )

    assert len(await adapter.workload(window, top_n=1)) == 1


async def test_dialect_rules_default_to_a_minimal_stub() -> None:
    adapter = FakeAdapter()

    rules = await adapter.dialect_rules()

    assert rules.engine == "clickhouse"
    assert rules.identifier_quote == "`"


async def test_scripted_dialect_rules_win() -> None:
    rules = await clickhouse_hits_fixture().dialect_rules()

    assert rules.version == "25.9"
    assert "estimate-only" in rules.quirks[0]


async def test_unscripted_plan_and_result_fail_loudly() -> None:
    adapter = FakeAdapter()

    with pytest.raises(AssertionError, match="plan was not scripted"):
        await adapter.explain("SELECT 1", ExplainMode.ESTIMATE)
    with pytest.raises(AssertionError, match="result was not scripted"):
        await adapter.execute("SELECT 1", Limits(timeout_s=5, max_result_rows=10))
