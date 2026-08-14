"""The plan-introspection call an agent makes before committing to a query.

The service is the whole of arm A3, so what matters is that it asks the engine
for a plan without executing anything, gathers only the facts the rules can use,
and profiles only what could change an answer.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, cast

import pytest

from agentdb.adapters import Capability, ExplainMode, RawPlan, RelationRef, SamplePolicy
from agentdb.config import Config
from agentdb.core import PlanExplainer, WarningCode
from tests.fakes import FakeAdapter, clickhouse_hits_fixture

ProfileCall = tuple[RelationRef, tuple[str, ...], SamplePolicy]

PLAN_WITH_NO_PRUNING: dict[str, Any] = {
    "Plan": {
        "Node Type": "ReadFromMergeTree",
        "Description": "agentdb.hits",
        "Indexes": [
            {"Type": "PrimaryKey", "Initial Granules": 1_000, "Selected Granules": 1_000},
        ],
    }
}


def _explainer(
    document: object = PLAN_WITH_NO_PRUNING, config: Config | None = None
) -> tuple[PlanExplainer, FakeAdapter]:
    """An explainer over the ClickBench-shaped fake, with a scripted plan."""
    adapter = clickhouse_hits_fixture()
    adapter.plan = RawPlan(
        engine="clickhouse",
        mode=ExplainMode.ESTIMATE,
        sql="",
        payload=json.dumps([document]),
    )
    return PlanExplainer(adapter=adapter, config=config or Config()), adapter


async def test_explaining_a_query_plans_it_without_executing_anything() -> None:
    explainer, adapter = _explainer()

    summary = await explainer.explain(
        "SELECT count() FROM hits WHERE SearchEngineID = 2", "agentdb"
    )

    assert adapter.calls_named("execute") == []
    (sql, mode) = cast(tuple[str, ExplainMode], adapter.calls_named("explain")[0])
    assert mode is ExplainMode.ESTIMATE
    assert sql.startswith("SELECT count()")
    assert summary.pruning_ratio == 1.0


async def test_the_warnings_combine_the_plan_the_layout_and_the_statistics() -> None:
    explainer, _ = _explainer()

    summary = await explainer.explain(
        "SELECT UserID, count() FROM hits WHERE SearchEngineID = 2 GROUP BY UserID", "agentdb"
    )
    codes = {warning.code for warning in summary.warnings}

    assert WarningCode.FULL_SCAN in codes  # from the plan
    assert WarningCode.SORT_KEY_UNUSED in codes  # from the layout
    assert WarningCode.HIGH_CARD_GROUP_BY in codes  # from the profile


async def test_only_the_grouping_columns_are_profiled() -> None:
    explainer, adapter = _explainer()

    await explainer.explain(
        "SELECT UserID, count() FROM hits WHERE SearchEngineID = 2 GROUP BY UserID", "agentdb"
    )

    calls = [cast(ProfileCall, call) for call in adapter.calls_named("column_profile")]
    assert [columns for (_, columns, _) in calls] == [("UserID",)]


async def test_a_query_that_groups_by_nothing_costs_no_profiling_probes() -> None:
    explainer, adapter = _explainer()

    await explainer.explain("SELECT count() FROM hits", "agentdb")

    assert adapter.calls_named("column_profile") == []


async def test_an_engine_without_column_statistics_still_gets_the_other_rules() -> None:
    explainer, adapter = _explainer()
    adapter.capabilities = frozenset()

    summary = await explainer.explain(
        "SELECT UserID, count() FROM hits WHERE SearchEngineID = 2 GROUP BY UserID", "agentdb"
    )
    codes = {warning.code for warning in summary.warnings}

    assert adapter.calls_named("column_profile") == []
    assert WarningCode.SORT_KEY_UNUSED in codes
    assert WarningCode.HIGH_CARD_GROUP_BY not in codes


async def test_a_grouping_column_the_relation_does_not_have_is_not_probed() -> None:
    explainer, adapter = _explainer()

    await explainer.explain("SELECT nope, count() FROM hits GROUP BY nope", "agentdb")

    assert adapter.calls_named("column_profile") == []


async def test_the_profiling_probe_runs_under_the_configured_bounds() -> None:
    explainer, adapter = _explainer(config=Config(profile_max_rows=1_234))

    await explainer.explain("SELECT UserID, count() FROM hits GROUP BY UserID", "agentdb")

    (_, _, policy) = cast(ProfileCall, adapter.calls_named("column_profile")[0])
    assert policy.max_rows == 1_234


@pytest.mark.parametrize("capability", [Capability.COLUMN_STATS])
def test_the_fixture_declares_the_capability_the_profiling_path_depends_on(
    capability: Capability,
) -> None:
    assert clickhouse_hits_fixture().supports(capability)


async def test_the_summary_keeps_the_query_it_explained() -> None:
    explainer, _ = _explainer()
    sql = "SELECT count() FROM hits WHERE CounterID = 42"

    summary = await explainer.explain(sql, "agentdb")

    assert summary.sql == sql
    assert summary.engine == "clickhouse"
    assert replace(summary, warnings=()).warnings == ()
