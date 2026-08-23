"""The advice tools' inputs, refusals and evidence gathering (SPEC §13.1, §9).

The recommendations themselves are specified by the advisor's own tests. What is
here is the part the tool layer owns: which engine may answer which question,
where the demand signal comes from, and the promise that nothing is ever
executed — an advice call that altered a table because an agent asked a question
would be the last time anyone gave this server a connection.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from agentdb.adapters import Capability, Engine, ExplainMode, RawPlan, WorkloadEntry
from agentdb.config import Config
from agentdb.server import build_catalog
from tests.fakes import clickhouse_hits_fixture, databricks_tpch_fixture
from tests.server_fakes import (
    clickhouse_catalog,
    databricks_catalog,
    scripted_clickhouse_adapter,
)

CH_SQL = "SELECT count() FROM hits WHERE UserID = 42"
CANDIDATE_SQL = "SELECT count() FROM hits WHERE SearchEngineID = 2"
"""SearchEngineID is outside this fixture's sort key, so a skip index is proposed
for it — UserID is inside the key and correctly earns no candidate at all."""

SHADOW_PLAN = json.dumps(
    {
        "Plan": {
            "Node Type": "ReadFromMergeTree",
            "Description": "agentdb.hits__agentdb_shadow",
            "Indexes": [{"Type": "PrimaryKey", "Initial Granules": 100, "Selected Granules": 5}],
        }
    }
)


@dataclass
class FakeShadowRunner:
    """A write channel that records its statements and returns a pruned plan."""

    engine: Engine = "clickhouse"
    statements: list[str] = field(default_factory=list)

    async def run(self, sql: str) -> None:
        self.statements.append(sql)

    async def explain(self, sql: str, mode: ExplainMode) -> RawPlan:
        return RawPlan(engine=self.engine, mode=mode, sql=sql, payload=SHADOW_PLAN)

    async def list_tables(self, namespace: str) -> Sequence[str]:
        return ()


def shadow_catalog(config: Config, runner: FakeShadowRunner) -> Any:
    """A catalog over the scripted fake, with a write channel behind it."""
    return build_catalog(scripted_clickhouse_adapter(), config, shadow=cast(Any, runner))


DBX_SQL = "SELECT count(*) FROM samples.tpch.lineitem WHERE l_shipdate > '1995-01-01'"


async def test_a_sort_key_proposal_carries_ddl_that_is_never_run() -> None:
    catalog, adapter = clickhouse_catalog()

    response = await catalog.call("advise_sort_key", {"relation": "agentdb.hits", "sql": CH_SQL})

    assert not response.is_error, response.structured
    recommendations = response.structured["recommendations"]
    assert isinstance(recommendations, list) and recommendations
    first = recommendations[0]
    assert isinstance(first, dict)
    assert "ORDER BY" in str(first["ddl"])
    assert [call for call in adapter.calls if call[0] == "execute"] == []


async def test_databricks_advice_on_a_clickhouse_connection_is_refused_by_name() -> None:
    catalog, _ = clickhouse_catalog()

    response = await catalog.call("advise_clustering", {"relation": "agentdb.hits"})

    assert response.is_error
    assert "Databricks-specific" in str(response.structured["error"])
    assert "advise_sort_key" in str(response.structured["suggestion"])


async def test_clickhouse_advice_on_a_warehouse_is_refused_the_same_way() -> None:
    catalog, _ = databricks_catalog()

    response = await catalog.call("advise_indexes", {"relation": "samples.tpch.lineitem"})

    assert response.is_error
    assert "ClickHouse-specific" in str(response.structured["error"])


async def test_the_statistics_tool_reports_the_columns_that_cannot_prune() -> None:
    catalog, _ = databricks_catalog()

    response = await catalog.call(
        "advise_skipping_stats", {"relation": "samples.tpch.lineitem", "sql": DBX_SQL}
    )

    assert not response.is_error, response.structured
    assert response.structured["workload_queries"]


async def test_a_workload_window_past_a_month_is_refused() -> None:
    catalog, _ = clickhouse_catalog()

    response = await catalog.call(
        "advise_indexes", {"relation": "agentdb.hits", "sql": CH_SQL, "hours": 24 * 400}
    )

    assert response.is_error
    assert "rotated" in str(response.structured["suggestion"])


async def test_a_connection_with_no_log_and_no_query_has_nothing_to_advise_from() -> None:
    """Advice from no demand signal at all would be a guess wearing a rationale."""
    adapter = clickhouse_hits_fixture()
    adapter.capabilities = adapter.capabilities - {Capability.WORKLOAD_LOG}
    catalog = build_catalog(adapter)

    response = await catalog.call("advise_indexes", {"relation": "agentdb.hits"})

    assert response.is_error
    assert "cannot read the engine's query log" in str(response.structured["error"])
    assert "'sql'" in str(response.structured["suggestion"])


async def test_the_mined_workload_is_counted_alongside_the_query_at_hand() -> None:
    catalog, adapter = clickhouse_catalog()

    response = await catalog.call("advise_indexes", {"relation": "agentdb.hits", "sql": CH_SQL})

    assert not response.is_error, response.structured
    assert adapter.calls_named("workload"), "the log was read"
    queries = response.structured["workload_queries"]
    assert isinstance(queries, int) and queries >= 1


async def test_only_the_columns_the_workload_filters_on_are_profiled() -> None:
    """Profiling is the expensive half of advising; a scan per unused column is waste."""
    catalog, adapter = clickhouse_catalog()

    await catalog.call("advise_indexes", {"relation": "agentdb.hits", "sql": CH_SQL})

    profiled = adapter.calls_named("column_profile")
    assert profiled
    columns = profiled[0][1]  # type: ignore[index]  # (ref, columns, policy)
    assert "UserID" in columns
    assert "URL" not in columns, "the query never filters on it"


async def test_a_relation_the_demand_signal_never_touches_is_not_profiled() -> None:
    """No filtered column, nothing worth a sampling scan."""
    adapter = clickhouse_hits_fixture()
    adapter.capabilities = adapter.capabilities - {Capability.WORKLOAD_LOG}
    catalog = build_catalog(adapter)

    response = await catalog.call(
        "advise_projection", {"relation": "agentdb.hits", "sql": "SELECT count() FROM hits"}
    )

    assert not response.is_error, response.structured
    assert adapter.calls_named("column_profile") == []


async def test_a_rewrite_is_offered_for_the_query_as_written() -> None:
    catalog, _ = databricks_catalog()

    response = await catalog.call(
        "suggest_rewrite",
        {
            "relation": "samples.tpch.lineitem",
            "sql": "SELECT count(*) FROM lineitem WHERE year(l_shipdate) = 1995",
        },
    )

    assert not response.is_error, response.structured
    recommendations = response.structured["recommendations"]
    assert isinstance(recommendations, list)
    rewritten = [str(item["rewritten_sql"]) for item in recommendations if isinstance(item, dict)]
    assert any("samples.tpch.lineitem" in sql for sql in rewritten)
    assert any("l_shipdate >= '1995-01-01'" in sql for sql in rewritten)


async def test_an_unqualified_relation_is_refused_before_any_engine_call() -> None:
    """The UNQUALIFIED_RELATION hazard, caught one layer earlier than the plan."""
    served = build_catalog(databricks_tpch_fixture())

    response = await served.call("advise_clustering", {"relation": "lineitem"})

    assert response.is_error
    assert "not a fully qualified relation name" in str(response.structured["error"])


# --------------------------------------------------------------------------
# validation: the only path to a measured recommendation
# --------------------------------------------------------------------------


async def test_validation_is_refused_where_the_server_has_no_write_channel() -> None:
    """The read-only role is the boundary; validation cannot borrow it."""
    catalog, _ = clickhouse_catalog()

    response = await catalog.call(
        "advise_indexes", {"relation": "agentdb.hits", "sql": CANDIDATE_SQL, "validate": True}
    )

    assert response.is_error
    assert "writable connection" in str(response.structured["error"])
    assert "read-only role" in str(response.structured["suggestion"])


async def test_validation_needs_the_query_it_is_measuring() -> None:
    """The mined workload says which columns matter; it does not say what to plan."""
    adapter = scripted_clickhouse_adapter()
    adapter.workload_entries = (
        *adapter.workload_entries,
        WorkloadEntry(
            normalized_sql="SELECT count() FROM hits WHERE SearchEngineID = ?",
            calls=900,
            relations=("agentdb.hits",),
            sample_sql="SELECT count() FROM hits WHERE SearchEngineID = 2",
        ),
    )
    catalog = build_catalog(
        adapter, Config(allow_shadow=True), shadow=cast(Any, FakeShadowRunner())
    )

    response = await catalog.call("advise_indexes", {"relation": "agentdb.hits", "validate": True})

    assert response.is_error
    assert "the query to measure" in str(response.structured["error"])


async def test_validation_without_the_opt_in_flag_says_which_flag() -> None:
    catalog = shadow_catalog(Config(), FakeShadowRunner())

    response = await catalog.call(
        "advise_indexes", {"relation": "agentdb.hits", "sql": CANDIDATE_SQL, "validate": True}
    )

    assert response.is_error
    assert "AGENTDB_ALLOW_SHADOW" in str(response.structured["suggestion"])


async def test_a_validated_recommendation_comes_back_measured() -> None:
    runner = FakeShadowRunner()
    catalog = shadow_catalog(Config(allow_shadow=True), runner)

    response = await catalog.call(
        "advise_indexes", {"relation": "agentdb.hits", "sql": CANDIDATE_SQL, "validate": True}
    )

    assert not response.is_error, response.structured
    recommendations = response.structured["recommendations"]
    assert isinstance(recommendations, list) and recommendations
    best = recommendations[0]
    assert isinstance(best, dict)
    assert best["confidence"] == "measured"
    assert isinstance(best["evidence"], dict) and best["evidence"]["source"] == "shadow"
    assert any("DROP TABLE" in statement for statement in runner.statements), "and cleaned up"


async def test_only_the_top_candidate_is_measured() -> None:
    """Every validation copies a sample of the table; on a warehouse that is money."""
    runner = FakeShadowRunner()
    catalog = shadow_catalog(Config(allow_shadow=True), runner)

    response = await catalog.call(
        "advise_indexes",
        {
            "relation": "agentdb.hits",
            "sql": "SELECT count() FROM hits WHERE SearchEngineID = 2 AND Referer = 'x'",
            "validate": True,
        },
    )

    recommendations = response.structured["recommendations"]
    assert isinstance(recommendations, list)
    confidences = [item["confidence"] for item in recommendations if isinstance(item, dict)]
    assert confidences.count("measured") == 1
    assert len([s for s in runner.statements if s.startswith("CREATE TABLE")]) == 1


async def test_the_validate_flag_must_be_a_boolean() -> None:
    catalog, _ = clickhouse_catalog()

    response = await catalog.call(
        "advise_indexes", {"relation": "agentdb.hits", "sql": CH_SQL, "validate": "yes"}
    )

    assert response.is_error
    assert "must be a boolean" in str(response.structured["error"])


async def test_nothing_to_recommend_means_nothing_to_validate() -> None:
    """No candidate, no shadow table: validation is never a reason to write."""
    runner = FakeShadowRunner()
    catalog = shadow_catalog(Config(allow_shadow=True), runner)

    response = await catalog.call(
        "advise_indexes",
        {
            "relation": "agentdb.hits",
            "sql": "SELECT count() FROM hits WHERE CounterID = 62",
            "validate": True,
        },
    )

    assert not response.is_error, response.structured
    assert response.structured["recommendations"] == []
    assert runner.statements == [], "a table nobody needed advice on was never copied"
