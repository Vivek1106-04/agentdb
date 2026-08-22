"""The schema fingerprint and what it does and does not notice (SPEC §10.3).

Two failures are being tested for, and they pull in opposite directions. A
fingerprint that misses a physical-design change serves an agent a query whose
pruning silently evaporated; one that notices row counts invalidates the entire
store every time somebody inserts a row. The cross-engine symmetry matters as
much: a Databricks clustering-key change has to invalidate exactly as a
ClickHouse sort-key change does, or the memory arm means something different on
each engine and the cross-engine table stops being comparable.
"""

from __future__ import annotations

import pytest

from agentdb.adapters import (
    ColumnDef,
    PhysicalLayout,
    Projection,
    RelationDetail,
    RelationRef,
    SkipIndex,
)
from agentdb.core.memory import (
    NamespaceSnapshot,
    RelationSnapshot,
    fingerprint,
    invalidation_reason,
    snapshot,
    snapshot_from_json,
    snapshot_to_json,
)

HITS = RelationRef(namespace="agentdb", name="hits")
LINEITEM = RelationRef(namespace="tpch", name="lineitem")


def hits_detail(*, event_date_type: str = "Date") -> RelationDetail:
    return RelationDetail(
        ref=HITS,
        columns=(
            ColumnDef(name="CounterID", data_type="UInt32", is_nullable=False),
            ColumnDef(name="EventDate", data_type=event_date_type, is_nullable=False),
            ColumnDef(name="URL", data_type="String", is_nullable=False),
        ),
        create_statement="CREATE TABLE agentdb.hits (...)",
    )


def hits_layout(
    *, order_by: tuple[str, ...] = ("CounterID", "EventDate"), **overrides: object
) -> PhysicalLayout:
    fields: dict[str, object] = {
        "engine": "clickhouse",
        "ref": HITS,
        "create_statement": "CREATE TABLE agentdb.hits (...)",
        "table_engine": "MergeTree",
        "order_by": order_by,
        "partition_by": ("toYYYYMM(EventDate)",),
        "approx_rows": 99_997_497,
        "on_disk_bytes": 14_779_976_446,
    }
    fields.update(overrides)
    return PhysicalLayout(**fields)  # type: ignore[arg-type]  # test factory, keyed by field name


def lineitem_layout(*, clustering: tuple[str, ...] = ("l_shipdate",)) -> PhysicalLayout:
    return PhysicalLayout(
        engine="databricks",
        ref=LINEITEM,
        create_statement="CREATE TABLE tpch.lineitem (...)",
        table_format="delta",
        clustering_columns=clustering,
        stats_columns=("l_shipdate", "l_orderkey"),
        deletion_vectors_enabled=True,
        num_files=140,
    )


def lineitem_detail() -> RelationDetail:
    return RelationDetail(
        ref=LINEITEM,
        columns=(
            ColumnDef(name="l_orderkey", data_type="bigint", is_nullable=False),
            ColumnDef(name="l_shipdate", data_type="date", is_nullable=False),
        ),
        create_statement="CREATE TABLE tpch.lineitem (...)",
    )


# --------------------------------------------------------------------------
# what the digest covers
# --------------------------------------------------------------------------


def test_the_digest_is_stable_against_the_order_an_adapter_returned_things_in() -> None:
    forward = snapshot("clickhouse", "agentdb", [hits_detail(), lineitem_detail()])
    backward = snapshot("clickhouse", "agentdb", [lineitem_detail(), hits_detail()])

    assert fingerprint(forward) == fingerprint(backward)


def test_a_clickhouse_sort_key_change_moves_the_digest() -> None:
    before = snapshot("clickhouse", "agentdb", [hits_detail()], [hits_layout()])
    after = snapshot(
        "clickhouse", "agentdb", [hits_detail()], [hits_layout(order_by=("EventDate", "CounterID"))]
    )

    assert fingerprint(before) != fingerprint(after)


def test_a_databricks_clustering_key_change_moves_the_digest_exactly_as_a_sort_key_does() -> None:
    before = snapshot("databricks", "tpch", [lineitem_detail()], [lineitem_layout()])
    after = snapshot(
        "databricks",
        "tpch",
        [lineitem_detail()],
        [lineitem_layout(clustering=("l_orderkey",))],
    )

    assert fingerprint(before) != fingerprint(after)


def test_a_column_retype_moves_the_digest() -> None:
    before = snapshot("clickhouse", "agentdb", [hits_detail()])
    after = snapshot("clickhouse", "agentdb", [hits_detail(event_date_type="DateTime")])

    assert fingerprint(before) != fingerprint(after)


def test_inserting_rows_does_not_move_the_digest() -> None:
    """Design, not size — otherwise every insert invalidates the whole store."""
    before = snapshot("clickhouse", "agentdb", [hits_detail()], [hits_layout()])
    after = snapshot(
        "clickhouse",
        "agentdb",
        [hits_detail()],
        [hits_layout(approx_rows=120_000_000, on_disk_bytes=20_000_000_000, num_files=900)],
    )

    assert fingerprint(before) == fingerprint(after)


def test_skip_indexes_and_projections_are_part_of_the_design() -> None:
    plain = snapshot("clickhouse", "agentdb", [hits_detail()], [hits_layout()])
    indexed = snapshot(
        "clickhouse",
        "agentdb",
        [hits_detail()],
        [
            hits_layout(
                skip_indexes=(
                    SkipIndex(
                        name="url_bloom",
                        index_type="bloom_filter",
                        expression="URL",
                        granularity=4,
                    ),
                ),
                projections=(Projection(name="by_url", query="SELECT URL ORDER BY URL"),),
            )
        ],
    )

    assert fingerprint(plain) != fingerprint(indexed)


def test_whitespace_inside_a_type_is_not_a_schema_change() -> None:
    spaced = snapshot(
        "databricks",
        "tpch",
        [
            RelationDetail(
                ref=LINEITEM,
                columns=(ColumnDef(name="l_tax", data_type="decimal(10, 2)", is_nullable=False),),
                create_statement="...",
            )
        ],
    )
    tight = snapshot(
        "databricks",
        "tpch",
        [
            RelationDetail(
                ref=LINEITEM,
                columns=(ColumnDef(name="l_tax", data_type="decimal(10,2)", is_nullable=False),),
                create_statement="...",
            )
        ],
    )

    assert fingerprint(spaced) == fingerprint(tight)


def test_layout_absent_is_not_the_same_state_as_layout_present() -> None:
    without = snapshot("clickhouse", "agentdb", [hits_detail()])
    with_layout = snapshot("clickhouse", "agentdb", [hits_detail()], [hits_layout()])

    assert fingerprint(without) != fingerprint(with_layout)


# --------------------------------------------------------------------------
# the layout_json round trip
# --------------------------------------------------------------------------


def test_the_stored_payload_decodes_back_to_the_same_snapshot() -> None:
    original = snapshot("clickhouse", "agentdb", [hits_detail()], [hits_layout()])

    restored = snapshot_from_json(snapshot_to_json(original))

    assert restored == original
    assert fingerprint(restored) == fingerprint(original)


def test_a_payload_from_another_snapshot_version_fails_loudly() -> None:
    payload = dict(snapshot_to_json(snapshot("clickhouse", "agentdb", [hits_detail()])))
    payload["version"] = 99

    with pytest.raises(ValueError, match="snapshot version 99"):
        snapshot_from_json(payload)


# --------------------------------------------------------------------------
# re-validation: why an exemplar died
# --------------------------------------------------------------------------


def current() -> NamespaceSnapshot:
    return snapshot("clickhouse", "agentdb", [hits_detail()], [hits_layout()])


def test_a_still_true_exemplar_has_no_reason() -> None:
    assert invalidation_reason(("hits",), ("CounterID", "hits.EventDate"), current()) is None


def test_a_dropped_relation_is_named() -> None:
    reason = invalidation_reason(("visits",), ("CounterID",), current())

    assert reason == "relation 'visits' no longer exists"


def test_a_qualified_column_on_a_dropped_relation_names_the_relation() -> None:
    reason = invalidation_reason((), ("visits.UserID",), current())

    assert reason == "relation 'visits' no longer exists"


def test_a_renamed_qualified_column_is_named() -> None:
    reason = invalidation_reason(("hits",), ("hits.UserID",), current())

    assert reason == "column 'hits.UserID' no longer exists"


def test_a_bare_column_holds_while_any_named_relation_offers_it() -> None:
    joined = snapshot("clickhouse", "agentdb", [hits_detail(), lineitem_detail()])

    assert invalidation_reason(("lineitem", "hits"), ("URL",), joined) is None


def test_a_bare_column_no_relation_offers_is_named() -> None:
    reason = invalidation_reason(("hits",), ("UserID",), current())

    assert reason == "column 'UserID' no longer exists on any relation this exemplar names"


def test_without_the_previous_snapshot_a_retype_cannot_be_reported() -> None:
    retyped = snapshot("clickhouse", "agentdb", [hits_detail(event_date_type="DateTime")])

    assert invalidation_reason(("hits",), ("EventDate",), retyped) is None


def test_a_retype_is_reported_against_the_version_the_exemplar_was_written_under() -> None:
    retyped = snapshot("clickhouse", "agentdb", [hits_detail(event_date_type="DateTime")])

    reason = invalidation_reason(("hits",), ("EventDate",), retyped, current())

    assert reason == "column hits.EventDate changed type from Date to DateTime"


def test_a_qualified_retype_is_reported_the_same_way() -> None:
    retyped = snapshot("clickhouse", "agentdb", [hits_detail(event_date_type="DateTime")])

    reason = invalidation_reason(("hits",), ("hits.EventDate",), retyped, current())

    assert reason == "column hits.EventDate changed type from Date to DateTime"


def test_an_unchanged_type_reports_nothing() -> None:
    assert invalidation_reason(("hits",), ("EventDate",), current(), current()) is None


def test_a_relation_absent_from_one_side_is_skipped_rather_than_guessed_at() -> None:
    """A relation the previous snapshot never held cannot have changed type."""
    previous = NamespaceSnapshot(engine="clickhouse", namespace="agentdb", relations=())

    assert invalidation_reason(("hits",), ("EventDate",), current(), previous) is None


def test_a_column_added_since_the_exemplar_was_written_is_not_a_retype() -> None:
    previous = NamespaceSnapshot(
        engine="clickhouse",
        namespace="agentdb",
        relations=(RelationSnapshot(name="hits", columns=(), layout=()),),
    )

    assert invalidation_reason(("hits",), ("EventDate",), current(), previous) is None
