"""Reading ``EXPLAIN FORMATTED`` into the plan IR (SPEC §7, §8.2).

The fixture below is the shape Databricks returns: a numbered physical-plan tree
followed by one detail block per node. Getting this wrong is not a crash — it is
a quietly wrong pruning number — so the tests assert on the evidence itself:
which predicates could reach per-file statistics, how many files were read,
which nodes ran on Photon, and which side of the join was built.

``VERIFY:`` this fixture against a live warehouse before the first Databricks
run. The key names are documented, not yet observed here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentdb.adapters import RawPlan
from agentdb.adapters.models import ExplainMode
from agentdb.core.plan_analyzer_databricks import (
    PlanParseError,
    classify,
    parse_plan,
    summarize,
)
from agentdb.core.plan_ir import PlanOp

PHOTON_PLAN = """== Physical Plan ==
PhotonResultStage (7)
+- PhotonProject (6)
   +- PhotonBroadcastHashJoin Inner BuildRight (5)
      :- PhotonScan parquet samples.tpch.lineitem (2)
      +- PhotonScan parquet samples.tpch.orders (4)

(2) PhotonScan parquet samples.tpch.lineitem
Output [3]: [l_orderkey#1, l_shipdate#2, l_extendedprice#3]
Batched: true
Location: PreparedDeltaFileIndex [s3://bucket/lineitem]
PartitionFilters: [isnotnull(l_shipdate#2), (l_shipdate#2 >= 1995-01-01)]
PushedFilters: [IsNotNull(l_orderkey), GreaterThan(l_orderkey,100)]
DataFilters: [upper(l_comment#9) LIKE %RUSH%]
number of files read: 40
size of files read: 1.2 GiB
Statistics: 1.2 GiB, 6001215 rows

(4) PhotonScan parquet samples.tpch.orders
Output [2]: [o_orderkey#7, o_custkey#8]
Location: PreparedDeltaFileIndex [s3://bucket/orders]
PushedFilters: []
number of files read: 8
Statistics: 128.0 MiB, 1500000 rows

(5) PhotonBroadcastHashJoin Inner BuildRight
Left keys [1]: [l_orderkey#1]
Right keys [1]: [o_orderkey#7]
Join condition: None

(6) PhotonProject
Output [2]: [l_orderkey#1, o_custkey#8]

(7) PhotonResultStage
Arguments: 1
"""

MIXED_PLAN = """== Physical Plan ==
Sort (4)
+- HashAggregate (3)
   +- PhotonScan parquet samples.tpch.lineitem (1)

(1) PhotonScan parquet samples.tpch.lineitem
PushedFilters: [IsNotNull(l_shipdate)]
number of files read: 100

(3) HashAggregate
Input [1]: [l_shipdate#2]

(4) Sort
Arguments: [l_shipdate#2 ASC NULLS FIRST]
"""


def _raw(payload: str, sql: str = "SELECT 1") -> RawPlan:
    return RawPlan(engine="databricks", mode=ExplainMode.ESTIMATE, sql=sql, payload=payload)


def test_the_tree_is_rebuilt_from_the_plans_own_indentation() -> None:
    root = parse_plan(_raw(PHOTON_PLAN))

    assert root.node_type == "PhotonResultStage"
    join = root.children[0].children[0]
    assert join.op is PlanOp.JOIN
    assert [child.relation for child in join.children] == [
        "samples.tpch.lineitem",
        "samples.tpch.orders",
    ]


def test_a_scan_separates_the_filters_that_can_skip_files_from_those_that_cannot() -> None:
    scans = [node for node in parse_plan(_raw(PHOTON_PLAN)).walk() if node.op is PlanOp.SCAN]
    lineitem = scans[0]

    # partition filters prune before the scan; pushed filters answer from per-file
    # statistics; a data filter reaches neither and is evaluated row by row
    assert lineitem.partition_filters == ("isnotnull(l_shipdate)", "(l_shipdate >= 1995-01-01)")
    assert lineitem.pushed_filters == ("IsNotNull(l_orderkey)", "GreaterThan(l_orderkey,100)")
    assert lineitem.data_filters == ("upper(l_comment) LIKE %RUSH%",)


def test_a_scan_reports_the_files_it_read_and_the_statistics_it_planned_on() -> None:
    lineitem = next(n for n in parse_plan(_raw(PHOTON_PLAN)).walk() if n.op is PlanOp.SCAN)

    assert lineitem.files_selected == 40
    assert lineitem.estimated_rows == 6_001_215
    assert lineitem.estimated_cost == int(1.2 * 1024**3)


def test_the_join_strategy_comes_from_the_node_class_because_nothing_else_states_it() -> None:
    join = next(n for n in parse_plan(_raw(PHOTON_PLAN)).walk() if n.op is PlanOp.JOIN)

    assert join.join_strategy == "broadcast_hash"


def test_photon_is_inferred_from_node_names_and_never_claimed_as_reported() -> None:
    summary = summarize(_raw(PHOTON_PLAN))

    assert summary.photon_coverage == 1.0
    assert all(node.photon for node in summary.root.walk())


def test_a_partial_photon_plan_reports_the_fraction_that_fell_back() -> None:
    summary = summarize(_raw(MIXED_PLAN))

    assert summary.photon_coverage == pytest.approx(1 / 3)


def test_a_pruning_ratio_needs_a_denominator_the_plan_does_not_carry() -> None:
    summary = summarize(_raw(PHOTON_PLAN))

    # the plan says 40 files were read; how many exist is a property of the table
    assert summary.pruning_ratio is None
    assert summary.pruning_unit is None


def test_file_totals_from_describe_detail_complete_the_ratio() -> None:
    summary = summarize(
        _raw(PHOTON_PLAN),
        files_total={"samples.tpch.lineitem": 400, "samples.tpch.orders": 16},
    )

    assert summary.pruning_ratio == (40 + 8) / (400 + 16)
    assert summary.pruning_unit == "file"
    assert summary.estimated_bytes_read == int(1.2 * 1024**3) + 128 * 1024**2


def test_a_totals_map_keyed_by_bare_table_name_still_matches() -> None:
    summary = summarize(_raw(MIXED_PLAN), files_total={"lineitem": 100})

    assert summary.pruning_ratio == 1.0
    assert summary.full_scan_relations == ("samples.tpch.lineitem",)


def test_an_unknown_relation_keeps_its_scan_out_of_the_ratio() -> None:
    summary = summarize(_raw(MIXED_PLAN), files_total={"orders": 10})

    assert summary.pruning_ratio is None


def test_output_without_a_physical_plan_is_refused_rather_than_half_read() -> None:
    with pytest.raises(PlanParseError, match="no physical plan tree"):
        parse_plan(_raw("== Parsed Logical Plan ==\nnothing useful here"))


@pytest.mark.parametrize(
    ("label", "op"),
    [
        ("PhotonScan parquet samples.tpch.lineitem", PlanOp.SCAN),
        ("Scan parquet spark_catalog.default.t", PlanOp.SCAN),
        ("ColumnarToRow", PlanOp.OTHER),
        ("SortMergeJoin Inner", PlanOp.JOIN),
        ("HashAggregate", PlanOp.AGGREGATE),
        ("PhotonGroupingAgg", PlanOp.AGGREGATE),
        ("Sort", PlanOp.SORT),
        ("Exchange hashpartitioning", PlanOp.EXCHANGE),
        ("AQEShuffleRead", PlanOp.EXCHANGE),
        ("TakeOrderedAndProject", PlanOp.LIMIT),
        ("CollectLimit", PlanOp.LIMIT),
        ("Filter", PlanOp.FILTER),
        ("PhotonProject", PlanOp.OTHER),
        ("", PlanOp.OTHER),
    ],
)
def test_node_classes_map_onto_the_shared_vocabulary(label: str, op: PlanOp) -> None:
    assert classify(label) is op


@pytest.mark.parametrize(
    ("label", "strategy"),
    [
        ("BroadcastHashJoin Inner BuildRight", "broadcast_hash"),
        ("SortMergeJoin Inner", "sort_merge"),
        ("ShuffledHashJoin LeftOuter", "shuffle_hash"),
        ("BroadcastNestedLoopJoin", "broadcast_nested_loop"),
    ],
)
def test_every_documented_join_class_maps_to_a_strategy(label: str, strategy: str) -> None:
    plan = f"== Physical Plan ==\n{label} (1)\n\n(1) {label}\nJoin condition: None\n"

    assert parse_plan(_raw(plan)).join_strategy == strategy


def test_a_scan_with_no_metrics_reports_unknown_rather_than_zero() -> None:
    plan = (
        "== Physical Plan ==\n"
        "Scan parquet samples.tpch.region (1)\n\n"
        "(1) Scan parquet samples.tpch.region\n"
        "PushedFilters: []\n"
    )

    scan = parse_plan(_raw(plan))

    assert scan.files_selected is None
    assert scan.estimated_rows is None
    assert scan.pushed_filters == ()
    assert scan.photon is False


def test_a_scan_whose_label_names_no_relation_says_so() -> None:
    plan = "== Physical Plan ==\nScan (1)\n\n(1) Scan\nPushedFilters: []\n"

    assert parse_plan(_raw(plan)).relation is None


def test_a_plan_without_the_header_is_still_read_from_its_first_line() -> None:
    plan = "Scan parquet samples.tpch.region (1)\n\n(1) Scan parquet samples.tpch.region\n"

    assert parse_plan(_raw(plan)).relation == "samples.tpch.region"


def test_a_filter_condition_becomes_the_nodes_filters() -> None:
    plan = (
        "== Physical Plan ==\n"
        "Filter (1)\n\n"
        "(1) Filter\n"
        "Condition : (isnotnull(l_orderkey#1) AND (l_orderkey#1 > 100))\n"
    )

    assert parse_plan(_raw(plan)).filters == ("(isnotnull(l_orderkey) AND (l_orderkey > 100))",)


def test_a_statistics_line_without_a_row_count_still_yields_bytes() -> None:
    plan = (
        "== Physical Plan ==\n"
        "Scan parquet samples.tpch.region (1)\n\n"
        "(1) Scan parquet samples.tpch.region\n"
        "Statistics: 4.0 MiB\n"
    )

    scan = parse_plan(_raw(plan))

    assert scan.estimated_cost == 4 * 1024**2
    assert scan.estimated_rows is None


def test_an_unreadable_metric_leaves_the_field_unknown() -> None:
    plan = (
        "== Physical Plan ==\n"
        "Scan parquet samples.tpch.region (1)\n\n"
        "(1) Scan parquet samples.tpch.region\n"
        "number of files read: unknown\n"
        "Statistics: unavailable\n"
    )

    scan = parse_plan(_raw(plan))

    assert scan.files_selected is None
    assert scan.estimated_cost is None


def test_two_top_level_nodes_do_not_swallow_each_other() -> None:
    # AQE output can list sibling stages at the same depth; the second is not a
    # child of the first, and treating it as one would misreport the shape
    plan = (
        "== Physical Plan ==\n"
        "PhotonScan parquet samples.tpch.lineitem (1)\n"
        "PhotonScan parquet samples.tpch.orders (2)\n\n"
        "(1) PhotonScan parquet samples.tpch.lineitem\n"
        "number of files read: 4\n\n"
        "(2) PhotonScan parquet samples.tpch.orders\n"
        "number of files read: 2\n"
    )

    root = parse_plan(_raw(plan))

    assert root.relation == "samples.tpch.lineitem"
    assert root.children == ()


def test_a_scan_the_plan_did_not_name_is_left_out_of_the_file_totals() -> None:
    plan = "== Physical Plan ==\nScan (1)\n\n(1) Scan\nnumber of files read: 4\n"

    summary = summarize(_raw(plan), files_total={"lineitem": 100})

    assert summary.pruning_ratio is None


# --------------------------------------------------------------------------
# pinned to observed output
#
# The fixture below was captured from a live Free Edition warehouse by
# scripts/verify_databricks.py (DBSQL 2026.20). Everything above this line was
# written from documentation; everything below is what the engine actually
# printed, and the two disagreed in ways that mattered.
# --------------------------------------------------------------------------

LIVE_PLAN = (
    Path(__file__).resolve().parents[1] / "fixtures" / "databricks" / "explain_formatted.txt"
)


@pytest.fixture(scope="module")
def live() -> RawPlan:
    if not LIVE_PLAN.is_file():
        pytest.skip(f"{LIVE_PLAN.name} not captured; run scripts/verify_databricks.py")
    return _raw(LIVE_PLAN.read_text(encoding="utf-8"))


def test_a_real_photon_plan_parses_into_a_tree(live: RawPlan) -> None:
    root = parse_plan(live)

    # every Databricks plan is wrapped in AdaptiveSparkPlan
    assert root.node_type == "AdaptiveSparkPlan"
    scans = [node for node in root.walk() if node.op is PlanOp.SCAN]
    assert [scan.relation for scan in scans] == ["samples.tpch.lineitem"]


def test_a_photon_scan_states_its_filters_under_databricks_own_spelling(
    live: RawPlan,
) -> None:
    scan = next(node for node in parse_plan(live).walk() if node.op is PlanOp.SCAN)

    # PhotonScan prints RequiredDataFilters and DictionaryFilters; it prints no
    # PushedFilters line at all, which the documented parser read as "pushed
    # nothing"
    assert scan.pushed_filters == (
        "isnotnull(l_shipdate)",
        "(l_shipdate >= 1995-01-01)",
    )
    assert scan.data_filters == ("(l_shipdate >= 1995-01-01)",)


def test_a_photon_plan_carries_no_file_counts_so_no_ratio_is_claimed(
    live: RawPlan,
) -> None:
    # the plan says nothing about files; a denominator alone must not become 0%
    summary = summarize(live, files_total={"samples.tpch.lineitem": 10})

    assert summary.pruning_ratio is None
    assert summary.pruning_unit is None
    assert summary.full_scan_relations == ()


def test_the_wrapper_node_does_not_count_as_a_photon_fallback(live: RawPlan) -> None:
    summary = summarize(live)

    # AdaptiveSparkPlan never carries the prefix; every operator below it does
    assert summary.photon_coverage == 1.0
