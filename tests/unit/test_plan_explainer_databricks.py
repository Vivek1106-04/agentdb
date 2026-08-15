"""The plan-introspection call, against Databricks (SPEC §7, §8.2).

Same contract as the ClickHouse arm — plan the query, gather only the facts the
rules can use, execute nothing — with one Databricks-specific obligation: the
plan reports how many files were *read*, and the denominator lives in
``DESCRIBE DETAIL``. The explainer joins the two, or reports no ratio at all.
"""

from __future__ import annotations

from typing import cast

from agentdb.adapters import ExplainMode, RawPlan, RelationRef
from agentdb.core import PlanExplainer, WarningCode
from tests.fakes import FakeAdapter, databricks_tpch_fixture

PLAN = """== Physical Plan ==
PhotonGroupingAgg (2)
+- PhotonScan parquet samples.tpch.lineitem (1)

(1) PhotonScan parquet samples.tpch.lineitem
PushedFilters: [IsNotNull(l_audit_note)]
number of files read: 1000
"""


def _explainer(payload: str = PLAN) -> tuple[PlanExplainer, FakeAdapter]:
    adapter = databricks_tpch_fixture()
    adapter.plan = RawPlan(engine="databricks", mode=ExplainMode.ESTIMATE, sql="", payload=payload)
    return PlanExplainer(adapter=adapter), adapter


async def test_the_databricks_plan_is_parsed_by_the_databricks_parser() -> None:
    explainer, adapter = _explainer()

    summary = await explainer.explain(
        "SELECT count(*) FROM samples.tpch.lineitem WHERE l_audit_note IS NOT NULL", "tpch"
    )

    assert adapter.calls_named("execute") == []
    assert summary.pruning_unit == "file"
    assert summary.photon_coverage == 1.0


async def test_the_file_total_comes_from_the_layout_not_from_the_plan() -> None:
    explainer, _ = _explainer()

    summary = await explainer.explain("SELECT count(*) FROM samples.tpch.lineitem", "tpch")

    # 1000 files read of the 1000 DESCRIBE DETAIL reported: nothing was skipped
    assert summary.pruning_ratio == 1.0
    assert summary.full_scan_relations == ("samples.tpch.lineitem",)


async def test_a_filter_past_the_statistics_limit_earns_the_warning_that_explains_it() -> None:
    explainer, _ = _explainer()

    summary = await explainer.explain(
        "SELECT count(*) FROM samples.tpch.lineitem WHERE l_audit_note = 'x'", "tpch"
    )

    codes = {warning.code for warning in summary.warnings}
    assert WarningCode.STATS_NOT_COLLECTED in codes
    assert WarningCode.CLUSTERING_KEY_UNUSED in codes


async def test_a_catalog_named_in_the_query_is_kept_rather_than_replaced() -> None:
    explainer, adapter = _explainer()

    await explainer.explain("SELECT count(*) FROM samples.tpch.lineitem", "tpch")

    ref = cast(RelationRef, adapter.calls_named("describe_relation")[0])
    assert ref == RelationRef(catalog="samples", namespace="tpch", name="lineitem")


async def test_an_under_qualified_query_still_resolves_against_the_given_namespace() -> None:
    explainer, adapter = _explainer()
    adapter.details["samples.tpch.lineitem"] = adapter.details["samples.tpch.lineitem"]
    adapter.details["tpch.lineitem"] = adapter.details["samples.tpch.lineitem"]
    adapter.layouts["tpch.lineitem"] = adapter.layouts["samples.tpch.lineitem"]

    summary = await explainer.explain("SELECT count(*) FROM lineitem", "tpch")

    ref = cast(RelationRef, adapter.calls_named("describe_relation")[0])
    assert ref == RelationRef(catalog=None, namespace="tpch", name="lineitem")
    assert WarningCode.UNQUALIFIED_RELATION in {warning.code for warning in summary.warnings}
