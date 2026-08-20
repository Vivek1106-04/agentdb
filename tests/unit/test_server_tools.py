"""What each tool does with the engine's answers, and with the caller's arguments.

The contract tests already prove every response matches its schema. What is left
here is behaviour: the limits ``run_query`` refuses to raise, the level
``grounded_context`` refuses to fake, and the choice ``explain_diff`` refuses to
make when the plans are indistinguishable.
"""

from __future__ import annotations

import json
from typing import Any, cast

from agentdb.adapters import Capability, ExplainMode, Limits, RawPlan
from agentdb.config import Config
from agentdb.server import build_catalog
from tests.fakes import CLICKHOUSE_CAPABILITIES, clickhouse_hits_fixture
from tests.server_fakes import (
    PRUNED_CLICKHOUSE_PLAN,
    clickhouse_catalog,
    databricks_catalog,
)

# --- discovery -------------------------------------------------------------


async def test_namespaces_come_back_fully_qualified_on_a_three_level_engine() -> None:
    """An agent must never have to guess whether a name wants two parts or three."""
    catalog, _ = databricks_catalog()

    response = await catalog.call("list_namespaces", {})

    assert response.structured["namespaces"] == ["samples.tpch"]


async def test_listing_relations_without_a_namespace_asks_the_engine_for_its_default() -> None:
    catalog, adapter = clickhouse_catalog()

    await catalog.call("list_relations", {})

    assert adapter.calls_named("list_relations") == [None]


async def test_columns_are_numbered_from_one_in_schema_order() -> None:
    """The ordinal is what decides whether Delta collects statistics for a column."""
    catalog, _ = databricks_catalog()

    response = await catalog.call("describe_relation", {"relation": "samples.tpch.lineitem"})

    columns = cast(list[dict[str, Any]], response.structured["columns"])
    assert columns[0]["ordinal"] == 1
    assert columns[-1] == {
        "name": "l_audit_note",
        "data_type": "string",
        "is_nullable": True,
        "ordinal": 40,
        "default_expression": None,
        "comment": None,
    }


# --- grounding -------------------------------------------------------------


async def test_profiling_goes_through_a_bounded_sample_policy() -> None:
    """Profiling a hundred-million-row table must never become a full scan."""
    catalog, adapter = clickhouse_catalog()

    await catalog.call("profile_columns", {"relation": "agentdb.hits", "columns": ["UserID"]})

    _, columns, policy = cast(tuple[Any, ...], adapter.calls_named("column_profile")[0])
    assert columns == ("UserID",)
    assert policy.fraction == Config().default_sample_fraction
    assert policy.max_rows == Config().profile_max_rows


async def test_the_default_grounding_level_is_layout() -> None:
    catalog, _ = clickhouse_catalog()

    response = await catalog.call("grounded_context", {"namespace": "agentdb"})

    assert response.structured["level"] == "layout"


async def test_naming_relations_narrows_the_bundle_without_listing_the_namespace() -> None:
    catalog, adapter = clickhouse_catalog()

    await catalog.call(
        "grounded_context", {"namespace": "agentdb", "relations": ["hits"], "level": "schema"}
    )

    assert adapter.calls_named("list_relations") == []


async def test_an_unknown_grounding_level_names_the_ones_that_exist() -> None:
    catalog, _ = clickhouse_catalog()

    response = await catalog.call("grounded_context", {"namespace": "agentdb", "level": "deep"})

    assert response.is_error
    assert response.structured["suggestion"] == "use one of: schema, stats, layout"


async def test_a_level_the_engine_cannot_serve_fails_instead_of_quietly_thinning() -> None:
    """A silently downgraded payload would be measured as something that never ran."""
    adapter = clickhouse_hits_fixture()
    adapter.capabilities = CLICKHOUSE_CAPABILITIES - {Capability.COLUMN_STATS}
    catalog = build_catalog(adapter)

    response = await catalog.call("grounded_context", {"namespace": "agentdb", "level": "stats"})

    assert response.is_error
    assert "column_stats" in str(response.structured["message"])


async def test_reserved_words_are_sorted_so_the_payload_is_byte_stable() -> None:
    catalog, _ = clickhouse_catalog()

    response = await catalog.call("dialect_rules", {})

    assert response.structured["reserved_words"] == ["ORDER", "SAMPLE", "SELECT"]


# --- plan ------------------------------------------------------------------


async def test_explaining_a_query_never_executes_it() -> None:
    catalog, adapter = clickhouse_catalog()

    await catalog.call("explain_plan", {"sql": "SELECT count() FROM hits", "namespace": "agentdb"})

    assert adapter.calls_named("execute") == []
    assert adapter.calls_named("explain") == [("SELECT count() FROM hits", ExplainMode.ESTIMATE)]


async def test_an_unreadable_plan_is_an_error_the_agent_can_route_around() -> None:
    catalog, adapter = clickhouse_catalog()
    adapter.plan = RawPlan(
        engine="clickhouse", mode=ExplainMode.ESTIMATE, sql="", payload="not a plan"
    )

    response = await catalog.call(
        "explain_plan", {"sql": "SELECT count() FROM hits", "namespace": "agentdb"}
    )

    assert response.is_error
    assert "run_query works without a plan review" in str(response.structured["suggestion"])


async def test_identical_drafts_get_no_recommendation_rather_than_an_arbitrary_one() -> None:
    catalog, _ = clickhouse_catalog()
    sql = "SELECT count() FROM hits WHERE UserID = 42"

    response = await catalog.call(
        "explain_diff", {"candidates": [sql, sql], "namespace": "agentdb"}
    )

    assert response.structured["recommended_index"] is None
    assert "indistinguishable" in str(response.structured["reason"])


async def test_the_draft_without_a_critical_finding_wins_over_an_equally_warned_one() -> None:
    """Both drafts prune identically here; only one skips the sort-key prefix."""
    catalog, adapter = clickhouse_catalog()
    adapter.plan = RawPlan(
        engine="clickhouse",
        mode=ExplainMode.ESTIMATE,
        sql="",
        payload=json.dumps([PRUNED_CLICKHOUSE_PLAN]),
    )

    response = await catalog.call(
        "explain_diff",
        {
            "candidates": [
                "SELECT count() FROM hits WHERE UserID = 42",
                "SELECT count() FROM hits WHERE CounterID = 62 GROUP BY UserID",
            ],
            "namespace": "agentdb",
        },
    )

    assert response.structured["recommended_index"] == 1
    assert "0 critical" in str(response.structured["reason"])


async def test_one_candidate_is_refused_with_a_pointer_to_the_right_tool() -> None:
    catalog, _ = clickhouse_catalog()

    response = await catalog.call(
        "explain_diff", {"candidates": ["SELECT 1"], "namespace": "agentdb"}
    )

    assert response.is_error
    assert response.structured["suggestion"] == "use explain_plan for a single query"


# --- execution -------------------------------------------------------------


async def test_a_caller_may_tighten_a_limit() -> None:
    catalog, adapter = clickhouse_catalog()

    await catalog.call("run_query", {"sql": "SELECT 1", "max_rows": 5, "timeout_s": 2})

    assert _limits(adapter) == Limits(
        timeout_s=2, max_result_rows=5, max_rows_to_read=Config().max_rows_to_read
    )


async def test_a_caller_may_not_raise_one_past_the_servers_own_ceiling() -> None:
    """A limit a caller can lift is not a limit."""
    config = Config(max_result_rows=100, query_timeout_s=5)
    catalog, adapter = clickhouse_catalog(config=config)

    await catalog.call("run_query", {"sql": "SELECT 1", "max_rows": 10**9, "timeout_s": 10**6})

    limits = _limits(adapter)
    assert (limits.max_result_rows, limits.timeout_s) == (100, 5)


# --- workload --------------------------------------------------------------


async def test_the_window_ends_now_and_is_as_long_as_asked_for() -> None:
    catalog, adapter = clickhouse_catalog()

    await catalog.call("mine_workload", {"hours": 6, "top_n": 3})

    window, top_n = cast(tuple[Any, int], adapter.calls_named("workload")[0])
    assert window.duration.total_seconds() == 6 * 3600
    assert top_n == 3


async def test_a_window_longer_than_the_log_keeps_is_refused() -> None:
    catalog, _ = clickhouse_catalog()

    response = await catalog.call("mine_workload", {"hours": 24 * 31})

    assert response.is_error
    assert "rotated" in str(response.structured["suggestion"])


async def test_an_engine_whose_log_is_unreadable_says_so_rather_than_returning_nothing() -> None:
    """An empty list would read as "this engine is idle", which is a different claim."""
    adapter = clickhouse_hits_fixture()
    adapter.capabilities = CLICKHOUSE_CAPABILITIES - {Capability.WORKLOAD_LOG}
    catalog = build_catalog(adapter)

    response = await catalog.call("mine_workload", {})

    assert response.is_error
    assert "cannot read the workload log" in str(response.structured["error"])


def _limits(adapter: Any) -> Limits:
    _, limits = cast(tuple[str, Limits], adapter.calls_named("execute")[0])
    return limits
