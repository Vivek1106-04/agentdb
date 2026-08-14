"""Parsing an engine's plan into the IR (SPEC §7, §8.1).

The pruning numbers are the point. A parser that silently produced zero granules
where the engine reported none would turn "no evidence" into "perfect pruning",
so the tests here are as much about what the parser refuses as what it reads.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from agentdb.adapters import ExplainMode, RawPlan
from agentdb.core.plan_analyzer import PlanParseError, classify, parse_plan, summarize
from agentdb.core.plan_ir import PlanOp, PlanSummary, PlanWarning, Severity, WarningCode

SCAN_WITH_INDEXES: dict[str, Any] = {
    "Node Type": "ReadFromMergeTree",
    "Description": "agentdb.hits",
    "Indexes": [
        {
            "Type": "MinMax",
            "Keys": ["EventDate"],
            "Condition": "(EventDate in [15887, +inf))",
            "Initial Parts": 10,
            "Selected Parts": 8,
            "Initial Granules": 1000,
            "Selected Granules": 800,
        },
        {
            "Type": "PrimaryKey",
            "Keys": ["CounterID"],
            "Condition": "(CounterID in [42, 42])",
            "Initial Parts": 8,
            "Selected Parts": 3,
            "Initial Granules": 800,
            "Selected Granules": 40,
        },
        {
            "Type": "Skip",
            "Name": "idx_url",
            "Description": "bloom_filter GRANULARITY 4",
            "Initial Parts": 3,
            "Selected Parts": 3,
            "Initial Granules": 40,
            "Selected Granules": 40,
        },
    ],
}

PLAN = {
    "Plan": {
        "Node Type": "Expression",
        "Plans": [
            {
                "Node Type": "Aggregating",
                "Plans": [{"Node Type": "Filter", "Plans": [SCAN_WITH_INDEXES]}],
            }
        ],
    }
}


def raw(document: object) -> RawPlan:
    return RawPlan(
        engine="clickhouse",
        mode=ExplainMode.ESTIMATE,
        sql="SELECT count() FROM hits WHERE CounterID = 42",
        payload=json.dumps([document]),
    )


# --------------------------------------------------------------------------
# node classification
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("node_type", "op"),
    [
        ("ReadFromMergeTree", PlanOp.SCAN),
        ("ReadFromStorage", PlanOp.SCAN),
        ("ReadFromRemote", PlanOp.EXCHANGE),
        ("ReadFromPreparedSource", PlanOp.OTHER),
        ("Union", PlanOp.EXCHANGE),
        ("FilledJoin", PlanOp.JOIN),
        ("Aggregating", PlanOp.AGGREGATE),
        ("MergingAggregated", PlanOp.AGGREGATE),
        ("Sorting", PlanOp.SORT),
        ("Limit", PlanOp.LIMIT),
        ("Offset", PlanOp.LIMIT),
        ("Filter", PlanOp.FILTER),
        ("Projection", PlanOp.PROJECTION_READ),
        ("Expression", PlanOp.OTHER),
    ],
)
def test_every_node_type_lands_somewhere_in_the_shared_vocabulary(
    node_type: str, op: PlanOp
) -> None:
    assert classify(node_type) is op


# --------------------------------------------------------------------------
# pruning evidence
# --------------------------------------------------------------------------


def test_the_scan_carries_what_the_indexes_started_with_and_what_survived() -> None:
    root = parse_plan(raw(PLAN))
    scan = next(node for node in root.walk() if node.op is PlanOp.SCAN)

    assert scan.relation == "agentdb.hits"
    assert scan.granules_total == 1000
    assert scan.granules_selected == 40
    assert scan.parts_total == 10
    assert scan.parts_selected == 3
    assert scan.pruning_ratio == 0.04


def test_only_the_indexes_that_removed_data_are_reported_as_having_fired() -> None:
    scan = next(node for node in parse_plan(raw(PLAN)).walk() if node.op is PlanOp.SCAN)

    assert scan.index_used == ("MinMax", "PrimaryKey")
    assert "idx_url" not in scan.index_used


def test_clickhouse_plans_never_claim_measured_rows() -> None:
    assert all(node.actual_rows is None for node in parse_plan(raw(PLAN)).walk())


def test_the_summary_aggregates_pruning_across_every_scan() -> None:
    second_scan = {
        "Node Type": "ReadFromMergeTree",
        "Description": "agentdb.visits",
        "Indexes": [
            {
                "Type": "PrimaryKey",
                "Initial Granules": 1000,
                "Selected Granules": 960,
                "Initial Parts": 4,
                "Selected Parts": 4,
            }
        ],
    }
    document = {"Plan": {"Node Type": "Join", "Plans": [SCAN_WITH_INDEXES, second_scan]}}

    summary = summarize(raw(document))

    assert summary.pruning_ratio == pytest.approx((40 + 960) / (1000 + 1000))
    assert summary.full_scan_relations == ()
    assert len(summary.scans) == 2


def test_a_scan_that_pruned_nothing_is_named_as_a_full_scan() -> None:
    unpruned = {
        "Node Type": "ReadFromMergeTree",
        "Description": "agentdb.hits",
        "Indexes": [
            {"Type": "PrimaryKey", "Initial Granules": 500, "Selected Granules": 500},
        ],
    }

    summary = summarize(raw({"Plan": unpruned}))

    assert summary.full_scan_relations == ("agentdb.hits",)
    assert summary.pruning_ratio == 1.0


def test_a_plan_with_no_index_evidence_reports_no_pruning_rather_than_zero() -> None:
    summary = summarize(raw({"Plan": {"Node Type": "ReadFromStorage", "Description": "hits"}}))

    assert summary.pruning_ratio is None
    assert summary.scans[0].granules_total is None
    assert summary.scans[0].pruning_ratio is None
    assert "no pruning evidence" in summary.render()


def test_index_entries_that_report_nothing_are_skipped_not_counted_as_zero() -> None:
    document = {
        "Plan": {
            "Node Type": "ReadFromMergeTree",
            "Description": "hits",
            "Indexes": [{"Type": "PrimaryKey", "Condition": "(CounterID in [1, 1])"}, "garbage"],
        }
    }

    scan = summarize(raw(document)).scans[0]

    assert scan.granules_total is None
    assert scan.index_used == ()
    assert scan.filters == ("(CounterID in [1, 1])",)


def test_an_indexes_field_that_is_not_a_list_is_ignored() -> None:
    document = {"Plan": {"Node Type": "ReadFromMergeTree", "Description": "hits", "Indexes": "no"}}

    scan = summarize(raw(document)).scans[0]

    assert scan.granules_total is None
    assert scan.index_used == ()


def test_a_projection_that_served_the_query_is_recorded() -> None:
    document = {
        "Plan": {"Node Type": "ReadFromMergeTree", "Description": "hits", "Projection": "by_user"}
    }

    scan = summarize(raw(document)).scans[0]

    assert scan.projection_used == "by_user"


def test_a_filter_node_keeps_the_condition_the_engine_stated() -> None:
    document = {"Plan": {"Node Type": "Filter", "Filter Column": "greater(UserID, 100)"}}

    root = parse_plan(raw(document))

    assert root.filters == ("greater(UserID, 100)",)
    assert root.relation is None


def test_row_and_cost_estimates_are_read_when_an_engine_supplies_them() -> None:
    document = {
        "Plan": {
            "Node Type": "ReadFromStorage",
            "Description": "hits",
            "Estimated Rows": 1000,
            "Estimated Cost": 12.5,
        }
    }

    scan = summarize(raw(document)).scans[0]

    assert scan.estimated_rows == 1000
    assert scan.estimated_cost == 12.5


# --------------------------------------------------------------------------
# refusing what cannot be read
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    ["", "Expression (Projection)\n  ReadFromMergeTree", "[]", '{"NotAPlan": 1}', '[{"Plan": 3}]'],
)
def test_plan_output_that_cannot_be_read_is_refused_rather_than_guessed(payload: str) -> None:
    plan = RawPlan(engine="clickhouse", mode=ExplainMode.ESTIMATE, sql="SELECT 1", payload=payload)

    with pytest.raises(PlanParseError):
        parse_plan(plan)


def test_a_bare_plan_object_is_accepted_as_well_as_a_one_element_list() -> None:
    plan = RawPlan(
        engine="clickhouse",
        mode=ExplainMode.ESTIMATE,
        sql="SELECT 1",
        payload=json.dumps(PLAN),
    )

    assert parse_plan(plan).node_type == "Expression"


def test_a_node_without_a_type_is_kept_rather_than_dropped() -> None:
    plan = RawPlan(
        engine="clickhouse",
        mode=ExplainMode.ESTIMATE,
        sql="SELECT 1",
        payload=json.dumps({"Plan": {"Plans": [{}]}}),
    )

    root = parse_plan(plan)

    assert root.node_type == "Unknown"
    assert root.children[0].op is PlanOp.OTHER


def test_a_scan_named_by_a_table_key_rather_than_a_description_is_still_named() -> None:
    document = {"Plan": {"Node Type": "ReadFromMergeTree", "Table": "agentdb.hits"}}

    assert summarize(raw(document)).scans[0].relation == "agentdb.hits"


def test_a_scan_the_engine_did_not_name_stays_unnamed() -> None:
    document = {"Plan": {"Node Type": "ReadFromMergeTree"}}

    summary = summarize(raw(document))

    assert summary.scans[0].relation is None
    assert summary.full_scan_relations == ()


def test_a_scan_reporting_a_start_but_no_survivors_reports_no_ratio() -> None:
    document = {
        "Plan": {
            "Node Type": "ReadFromMergeTree",
            "Description": "hits",
            "Indexes": [{"Type": "PrimaryKey", "Initial Granules": 100}],
        }
    }

    scan = summarize(raw(document)).scans[0]

    assert scan.granules_total == 100
    assert scan.granules_selected is None
    assert scan.pruning_ratio is None


def test_the_rendered_summary_states_every_fact_it_holds() -> None:
    summary = summarize(raw(PLAN))
    rendered = PlanSummary(
        root=summary.root,
        engine=summary.engine,
        sql=summary.sql,
        pruning_ratio=1.0,
        full_scan_relations=("agentdb.hits",),
        estimated_bytes_read=14_779_976_446,
        warnings=(
            PlanWarning(
                code=WarningCode.FULL_SCAN,
                severity=Severity.WARNING,
                human_message="nothing was pruned",
            ),
        ),
    ).render()

    assert "- full scan: agentdb.hits" in rendered
    assert "- estimated bytes read: 14,779,976,446" in rendered
    assert "- [warning] FULL_SCAN: nothing was pruned" in rendered
    assert "try:" not in rendered
