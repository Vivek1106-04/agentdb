"""The Databricks statements are the spec (SPEC §8.2), so assert on their text.

A mock cannot catch a statement that reads the wrong table property. These tests
read the strings the adapter will send, because a wrong property name here does
not crash — it produces a plausible layout with the pruning fact missing.
"""

from __future__ import annotations

import pytest

from agentdb.adapters import databricks_sql as dbx
from agentdb.adapters.models import ExplainMode, RelationRef

LINEITEM = RelationRef(catalog="samples", namespace="tpch", name="lineitem")


def test_a_reference_is_quoted_at_all_three_levels() -> None:
    assert dbx.qualified(LINEITEM) == "`samples`.`tpch`.`lineitem`"


def test_a_reference_without_a_catalog_is_refused_rather_than_defaulted() -> None:
    # A two-part name resolves against session USE state a stateless server lacks
    with pytest.raises(dbx.IdentifierError, match="all three name parts"):
        dbx.qualified(RelationRef(namespace="tpch", name="lineitem"))


@pytest.mark.parametrize("name", ["", "2fast", "with space", "drop`table", "l-shipdate"])
def test_a_name_that_is_not_an_identifier_never_reaches_a_statement(name: str) -> None:
    with pytest.raises(dbx.IdentifierError):
        dbx.quote_identifier(name)


def test_statements_carry_the_attribution_comment() -> None:
    tagged = dbx.tag("SELECT 1", context_id="bench", turn_id="abc123")

    assert tagged.startswith("/* agentdb:bench:abc123 */")
    assert tagged.endswith("SELECT 1")


def test_listings_read_unity_catalogs_information_schema_with_parameters() -> None:
    assert "system.information_schema.tables" in dbx.LIST_RELATIONS
    assert ":catalog" in dbx.LIST_RELATIONS
    assert ":schema" in dbx.LIST_RELATIONS
    assert "information_schema" in dbx.LIST_RELATIONS_ALL


def test_columns_are_read_in_ordinal_order_because_delta_statistics_stop_at_an_ordinal() -> None:
    assert "ordinal_position" in dbx.DESCRIBE_COLUMNS
    assert dbx.DESCRIBE_COLUMNS.rstrip().endswith("ORDER BY ordinal_position")


def test_workload_reads_the_query_history_system_table() -> None:
    assert "system.query.history" in dbx.WORKLOAD
    assert "statement_id" in dbx.WORKLOAD
    # failed statements are workload too: they are the shapes an agent got wrong
    assert "execution_status" in dbx.WORKLOAD
    assert "WHERE execution_status" not in dbx.WORKLOAD


def test_layout_statements_name_the_relation_fully() -> None:
    assert dbx.describe_detail(LINEITEM) == "DESCRIBE DETAIL `samples`.`tpch`.`lineitem`"
    assert dbx.show_tblproperties(LINEITEM) == "SHOW TBLPROPERTIES `samples`.`tpch`.`lineitem`"
    assert dbx.show_create_table(LINEITEM) == "SHOW CREATE TABLE `samples`.`tpch`.`lineitem`"
    assert dbx.describe_history(LINEITEM, limit=50).endswith("`lineitem` LIMIT 50")


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (ExplainMode.ESTIMATE, "EXPLAIN FORMATTED SELECT 1"),
        (ExplainMode.COST, "EXPLAIN COST SELECT 1"),
        (ExplainMode.PIPELINE, "EXPLAIN FORMATTED SELECT 1"),
        (ExplainMode.SYNTAX, "EXPLAIN EXTENDED SELECT 1"),
    ],
)
def test_each_explain_mode_maps_to_the_documented_databricks_form(
    mode: ExplainMode, expected: str
) -> None:
    assert dbx.explain_statement("SELECT 1", mode) == expected


def test_the_profile_probe_samples_rather_than_scanning() -> None:
    statement = dbx.profile_statement(
        source="`samples`.`tpch`.`lineitem`", column="l_shipdate", sample_percent=1.0
    )

    assert "TABLESAMPLE (1.0 PERCENT)" in statement
    assert "approx_count_distinct(`l_shipdate`)" in statement
    assert "count_if(`l_shipdate` IS NULL)" in statement


def test_top_values_are_a_second_query_because_databricks_has_no_topk() -> None:
    statement = dbx.top_values_statement(
        source="`samples`.`tpch`.`lineitem`", column="l_returnflag", top_k=10, sample_percent=1.0
    )

    assert statement.startswith("SELECT cast(`l_returnflag` AS STRING)")
    assert statement.endswith("GROUP BY 1 ORDER BY 2 DESC LIMIT 10")


def test_results_are_read_by_column_name_not_by_position() -> None:
    mapping = dbx.row_mapping(("format", "numFiles"), ("delta", 12))

    assert mapping == {"format": "delta", "numFiles": 12}


@pytest.mark.parametrize(
    ("table_type", "kind"),
    [
        ("MANAGED", "table"),
        ("EXTERNAL", "table"),
        ("VIEW", "view"),
        ("MATERIALIZED_VIEW", "materialized_view"),
        ("STREAMING_TABLE", "materialized_view"),
        ("FOREIGN", "foreign_table"),
    ],
)
def test_unity_catalog_table_types_map_onto_relation_kinds(table_type: str, kind: str) -> None:
    assert dbx.relation_kind(table_type) == kind


@pytest.mark.parametrize(
    ("value", "expected"),
    [("YES", True), ("yes", True), ("NO", False), (None, False)],
)
def test_nullability_is_the_information_schema_string(value: object, expected: bool) -> None:
    assert dbx.is_nullable(value) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (["a", "b"], ("a", "b")),
        ('["a","b"]', ("a", "b")),
        # an empty list is a measurement — "this table has no clustering key" —
        # and stays distinct from the unreadable cases below, which are unknown
        ("[]", ()),
        ([], ()),
        ("", None),
        (None, None),
        ("not json", None),
        (7, None),
    ],
)
def test_list_shaped_values_survive_both_transports(
    value: object, expected: tuple[str, ...] | None
) -> None:
    # An unparsed clustering key must read as unknown, never as "no clustering key"
    assert dbx.string_tuple(value) == expected


def test_table_properties_become_a_mapping_and_short_rows_are_ignored() -> None:
    rows = [("delta.dataSkippingNumIndexedCols", "8"), ("broken",)]

    assert dbx.properties(rows) == {"delta.dataSkippingNumIndexedCols": "8"}


@pytest.mark.parametrize(
    ("value", "expected"),
    [(12, 12), (12.9, 12), ("12", 12), (None, None), (True, None), ("many", None)],
)
def test_counts_are_read_without_turning_unknown_into_zero(
    value: object, expected: int | None
) -> None:
    assert dbx.optional_int(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [("true", True), ("TRUE", True), ("false", False), ("0", False), (True, True), (None, None)],
)
def test_delta_boolean_properties_are_strings(value: object, expected: bool | None) -> None:
    assert dbx.optional_bool(value) is expected


def test_an_unreadable_boolean_property_is_unknown() -> None:
    assert dbx.optional_bool("maybe") is None


def test_zorder_columns_come_from_the_latest_optimize_in_the_history() -> None:
    history = [
        {"operation": "WRITE", "operationParameters": {}},
        {"operation": "OPTIMIZE", "operationParameters": {"zOrderBy": '["l_partkey"]'}},
        {"operation": "OPTIMIZE", "operationParameters": {"zOrderBy": '["l_orderkey"]'}},
    ]

    # the newer entry wins: an older OPTIMIZE describes a layout that no longer exists
    assert dbx.zorder_columns(history) == ("l_partkey",)


@pytest.mark.parametrize(
    "history",
    [
        [],
        [{"operation": "WRITE", "operationParameters": {"zOrderBy": '["x"]'}}],
        [{"operation": "OPTIMIZE", "operationParameters": {}}],
        [{"operation": "OPTIMIZE", "operationParameters": "not a mapping"}],
        [{"operation": "OPTIMIZE", "operationParameters": {"zOrderBy": "[]"}}],
    ],
)
def test_a_table_that_was_never_zordered_reports_no_zorder_columns(
    history: list[dict[str, object]],
) -> None:
    assert dbx.zorder_columns(history) is None
