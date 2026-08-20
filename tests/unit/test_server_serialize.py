"""Turning value objects into the JSON the schemas promise.

One rule carries most of these tests: ``None`` survives. An engine that could
not report a figure must not come out the other side reporting zero, because an
agent cannot tell those apart and will state the zero as a fact.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from agentdb.adapters import (
    ColumnProfile,
    PhysicalLayout,
    Projection,
    RelationRef,
    ResultSet,
    SkipIndex,
)
from agentdb.core.plan_ir import PlanNode, PlanOp
from agentdb.server import serialize

REF = RelationRef(namespace="agentdb", name="hits")


def test_a_two_part_reference_reports_a_null_catalog_and_a_usable_name() -> None:
    assert serialize.relation_ref(REF) == {
        "catalog": None,
        "namespace": "agentdb",
        "name": "hits",
        "fqn": "agentdb.hits",
    }


def test_a_three_part_reference_keeps_its_catalog_in_the_name_to_write() -> None:
    ref = RelationRef(catalog="samples", namespace="tpch", name="lineitem")

    assert serialize.relation_ref(ref)["fqn"] == "samples.tpch.lineitem"


def test_an_unknown_key_stays_null_and_an_empty_one_stays_empty() -> None:
    """ "We did not look" and "there is nothing there" are different facts."""
    layout = PhysicalLayout(
        engine="clickhouse",
        ref=REF,
        create_statement="CREATE TABLE agentdb.hits (…)",
        order_by=(),
    )

    rendered = serialize.physical_layout(layout)

    assert rendered["order_by"] == []
    assert rendered["partition_by"] is None


def test_skip_indexes_and_projections_are_rendered_in_full() -> None:
    layout = PhysicalLayout(
        engine="clickhouse",
        ref=REF,
        create_statement="…",
        skip_indexes=(
            SkipIndex(
                name="idx_url",
                index_type="bloom_filter",
                expression="URL",
                granularity=4,
                compressed_bytes=1_024,
            ),
        ),
        projections=(Projection(name="p_by_date", query="SELECT * ORDER BY EventDate"),),
    )

    rendered = serialize.physical_layout(layout)

    assert rendered["skip_indexes"] == [
        {
            "name": "idx_url",
            "index_type": "bloom_filter",
            "expression": "URL",
            "granularity": 4,
            "compressed_bytes": 1_024,
        }
    ]
    assert rendered["projections"] == [
        {"name": "p_by_date", "query": "SELECT * ORDER BY EventDate"}
    ]


def test_top_values_become_named_pairs_rather_than_positional_ones() -> None:
    profile = ColumnProfile(
        name="SearchEngineID",
        data_type="UInt16",
        sample_method="sample",
        sampled_rows=1_000,
        top_values=(("2", 500),),
    )

    assert serialize.column_profile(profile)["top_values"] == [{"value": "2", "count": 500}]


def test_a_value_json_has_no_room_for_becomes_its_string_form() -> None:
    """Coercing a Decimal to a float would corrupt exactly the aggregates we grade."""
    result = ResultSet(
        columns=("day", "revenue", "ok", "nothing"),
        rows=((date(1998, 1, 1), Decimal("1234.56"), True, None),),
        row_count=1,
        truncated=False,
    )

    assert serialize.result_set(result)["rows"] == [["1998-01-01", "1234.56", True, None]]


def test_a_plan_node_carries_its_subtree_and_its_own_pruning_ratio() -> None:
    node = PlanNode(
        op=PlanOp.AGGREGATE,
        node_type="Aggregating",
        children=(
            PlanNode(
                op=PlanOp.SCAN,
                node_type="ReadFromMergeTree",
                relation="agentdb.hits",
                granules_total=1_000,
                granules_selected=10,
            ),
        ),
    )

    rendered = serialize.plan_node(node)

    assert rendered["pruning_ratio"] is None
    children = rendered["children"]
    assert isinstance(children, list)
    child = children[0]
    assert isinstance(child, dict)
    assert child["pruning_ratio"] == 0.01
