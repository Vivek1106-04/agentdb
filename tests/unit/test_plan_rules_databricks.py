"""The Databricks rules fire on evidence and stay silent without it (SPEC §7).

Each test states the fact the rule is derived from. The one worth reading twice
is ``STATS_NOT_COLLECTED``: it is the warning with no ClickHouse counterpart, it
costs a hundredfold on a wide table, and neither the query nor the schema shows
the problem.
"""

from __future__ import annotations

import pytest

from agentdb.adapters import PhysicalLayout, RelationRef
from agentdb.config import Config
from agentdb.core.plan_ir import PlanNode, PlanOp, PlanSummary, WarningCode
from agentdb.core.plan_rules import RelationFacts, evaluate
from agentdb.core.query_shape import analyze

CONFIG = Config()
LINEITEM = RelationRef(catalog="samples", namespace="tpch", name="lineitem")
RELATION = "samples.tpch.lineitem"

ORDINALS = {
    "l_orderkey": 1,
    "l_shipdate": 11,
    "l_comment": 16,
    "l_audit_note": 40,
}


def _layout(**overrides: object) -> PhysicalLayout:
    base: dict[str, object] = {
        "engine": "databricks",
        "ref": LINEITEM,
        "create_statement": "CREATE TABLE samples.tpch.lineitem (...) USING delta",
        "table_format": "delta",
        "clustering_columns": ("l_shipdate",),
        "approx_rows": 6_001_215,
        "stats_indexed_columns": 32,
    }
    return PhysicalLayout(**{**base, **overrides})  # type: ignore[arg-type]


def _summary(
    *,
    files_total: int | None = 1_000,
    files_selected: int | None = 40,
    partition_filters: tuple[str, ...] = (),
    photon: bool = True,
    sql: str = "SELECT 1",
    relation: str | None = RELATION,
) -> PlanSummary:
    scan = PlanNode(
        op=PlanOp.SCAN,
        node_type=("PhotonScan parquet " if photon else "Scan parquet ") + (relation or ""),
        relation=relation,
        files_total=files_total,
        files_selected=files_selected,
        partition_filters=partition_filters,
        photon=photon,
    )
    ratio = None if not files_total or files_selected is None else files_selected / files_total
    return PlanSummary(
        root=scan,
        engine="databricks",
        sql=sql,
        pruning_ratio=ratio,
        pruning_unit="file" if ratio is not None else None,
        photon_coverage=1.0 if photon else 0.0,
    )


def _facts(layout: PhysicalLayout | None = None, **kwargs: object) -> dict[str, RelationFacts]:
    return {
        RELATION: RelationFacts(
            layout=layout or _layout(),
            column_ordinals=ORDINALS,
            **kwargs,  # type: ignore[arg-type]
        )
    }


def _codes(sql: str, *, summary: PlanSummary | None = None, **facts: object) -> set[WarningCode]:
    resolved = summary or _summary(sql=sql)
    evaluated = evaluate(
        resolved,
        analyze(sql, "databricks"),
        _facts(**facts),  # type: ignore[arg-type]
        CONFIG,
    )
    return {warning.code for warning in evaluated.warnings}


# -- clustering -------------------------------------------------------------


def test_a_filter_that_misses_the_clustering_key_is_flagged() -> None:
    sql = "SELECT count(*) FROM samples.tpch.lineitem WHERE l_orderkey = 42"

    assert WarningCode.CLUSTERING_KEY_UNUSED in _codes(sql)


def test_a_filter_on_the_clustering_key_is_not_flagged() -> None:
    sql = "SELECT count(*) FROM samples.tpch.lineitem WHERE l_shipdate > '1995-01-01'"

    assert WarningCode.CLUSTERING_KEY_UNUSED not in _codes(sql)


def test_a_zordered_table_is_judged_on_its_zorder_columns() -> None:
    sql = "SELECT count(*) FROM samples.tpch.lineitem WHERE l_comment IS NOT NULL"
    layout = _layout(clustering_columns=None, zorder_columns=("l_shipdate",))

    assert WarningCode.CLUSTERING_KEY_UNUSED in _codes(sql, layout=layout)


def test_a_table_with_no_clustering_at_all_earns_no_clustering_warning() -> None:
    sql = "SELECT count(*) FROM samples.tpch.lineitem WHERE l_orderkey = 42"
    layout = _layout(clustering_columns=None)

    assert WarningCode.CLUSTERING_KEY_UNUSED not in _codes(sql, layout=layout)


def test_a_query_with_no_filter_earns_no_clustering_warning() -> None:
    sql = "SELECT count(*) FROM samples.tpch.lineitem"

    assert WarningCode.CLUSTERING_KEY_UNUSED not in _codes(sql)


# -- statistics -------------------------------------------------------------


def test_a_filter_on_a_column_past_the_indexed_limit_cannot_skip_files() -> None:
    sql = "SELECT count(*) FROM samples.tpch.lineitem WHERE l_audit_note = 'x'"
    layout = _layout(stats_indexed_columns=32)

    codes = _codes(sql, layout=layout)

    assert WarningCode.STATS_NOT_COLLECTED in codes


def test_the_statistics_warning_names_the_limit_it_fired_on() -> None:
    sql = "SELECT count(*) FROM samples.tpch.lineitem WHERE l_audit_note = 'x'"
    evaluated = evaluate(_summary(sql=sql), analyze(sql, "databricks"), _facts(), CONFIG)

    warning = next(w for w in evaluated.warnings if w.code is WarningCode.STATS_NOT_COLLECTED)
    assert "first 32 columns" in warning.human_message
    assert warning.columns == ("l_audit_note",)


def test_an_explicit_statistics_column_list_is_quoted_in_the_warning() -> None:
    sql = "SELECT count(*) FROM samples.tpch.lineitem WHERE l_shipdate > '1995-01-01'"
    layout = _layout(stats_columns=("l_orderkey",))
    evaluated = evaluate(
        _summary(sql=sql), analyze(sql, "databricks"), _facts(layout=layout), CONFIG
    )

    warning = next(w for w in evaluated.warnings if w.code is WarningCode.STATS_NOT_COLLECTED)
    assert "collects statistics only for l_orderkey" in warning.human_message


def test_an_indexed_column_earns_no_statistics_warning() -> None:
    sql = "SELECT count(*) FROM samples.tpch.lineitem WHERE l_shipdate > '1995-01-01'"

    assert WarningCode.STATS_NOT_COLLECTED not in _codes(sql)


def test_without_ordinals_the_statistics_rule_stays_silent_rather_than_guessing() -> None:
    sql = "SELECT count(*) FROM samples.tpch.lineitem WHERE l_audit_note = 'x'"
    facts = {RELATION: RelationFacts(layout=_layout())}

    evaluated = evaluate(_summary(sql=sql), analyze(sql, "databricks"), facts, CONFIG)

    assert WarningCode.STATS_NOT_COLLECTED not in {w.code for w in evaluated.warnings}


def test_a_filter_on_a_column_the_catalogue_does_not_know_is_not_flagged() -> None:
    sql = "SELECT count(*) FROM samples.tpch.lineitem WHERE mystery = 1"

    assert WarningCode.STATS_NOT_COLLECTED not in _codes(sql)


# -- partitions -------------------------------------------------------------


def test_a_partitioned_table_with_no_pushed_partition_filter_is_flagged() -> None:
    # the query does constrain the column — wrapped in a function, so nothing pushed
    sql = "SELECT count(*) FROM samples.tpch.lineitem WHERE year(l_shipdate) = 1995"
    layout = _layout(partition_by=("l_shipdate",))

    codes = _codes(sql, layout=layout)

    assert WarningCode.MISSING_PARTITION_PREDICATE in codes


def test_a_pushed_partition_filter_clears_the_warning() -> None:
    sql = "SELECT count(*) FROM samples.tpch.lineitem WHERE l_shipdate > '1995-01-01'"
    summary = _summary(sql=sql, partition_filters=("(l_shipdate > 1995-01-01)",))

    codes = _codes(sql, summary=summary, layout=_layout(partition_by=("l_shipdate",)))

    assert WarningCode.MISSING_PARTITION_PREDICATE not in codes


def test_an_unpartitioned_table_earns_no_partition_warning() -> None:
    sql = "SELECT count(*) FROM samples.tpch.lineitem WHERE l_orderkey = 1"

    assert WarningCode.MISSING_PARTITION_PREDICATE not in _codes(sql)


def test_a_plan_with_no_scan_for_this_relation_leaves_the_partition_rule_silent() -> None:
    # no scan node means no pushdown evidence, and a warning without evidence is a guess
    sql = "SELECT count(*) FROM lineitem WHERE l_orderkey = 1"
    summary = PlanSummary(
        root=PlanNode(op=PlanOp.AGGREGATE, node_type="PhotonGroupingAgg", photon=True),
        engine="databricks",
        sql=sql,
        photon_coverage=1.0,
    )
    facts = {
        "lineitem": RelationFacts(
            layout=_layout(partition_by=("l_shipdate",)), column_ordinals=ORDINALS
        )
    }

    evaluated = evaluate(summary, analyze(sql, "databricks"), facts, CONFIG)

    assert WarningCode.MISSING_PARTITION_PREDICATE not in {w.code for w in evaluated.warnings}


# -- small files ------------------------------------------------------------


def test_a_fragmented_table_is_flagged_as_a_maintenance_problem() -> None:
    sql = "SELECT count(*) FROM samples.tpch.lineitem"
    layout = _layout(num_files=5_000, avg_file_bytes=4 * 1024 * 1024)

    codes = _codes(sql, layout=layout)

    assert WarningCode.SMALL_FILES in codes


@pytest.mark.parametrize(
    ("num_files", "avg_bytes"),
    [
        (10, 4 * 1024 * 1024),  # few files, however small
        (5_000, 256 * 1024 * 1024),  # many files, but each one large
        (None, 4 * 1024 * 1024),  # the engine reported no count
        (5_000, None),  # the engine reported no size
    ],
)
def test_small_files_stays_silent_without_both_halves_of_the_evidence(
    num_files: int | None, avg_bytes: float | None
) -> None:
    sql = "SELECT count(*) FROM samples.tpch.lineitem"
    layout = _layout(num_files=num_files, avg_file_bytes=avg_bytes)

    assert WarningCode.SMALL_FILES not in _codes(sql, layout=layout)


# -- Photon -----------------------------------------------------------------


def test_a_plan_that_fell_off_photon_is_flagged_with_the_nodes_that_did() -> None:
    sql = "SELECT count(*) FROM samples.tpch.lineitem"
    summary = _summary(sql=sql, photon=False)

    evaluated = evaluate(summary, analyze(sql, "databricks"), _facts(), CONFIG)

    warning = next(w for w in evaluated.warnings if w.code is WarningCode.PHOTON_FALLBACK)
    assert "inferred from the node names" in warning.human_message
    assert warning.columns == ("Scan parquet samples.tpch.lineitem",)


def test_a_fully_vectorized_plan_earns_no_photon_warning() -> None:
    sql = "SELECT count(*) FROM samples.tpch.lineitem"

    assert WarningCode.PHOTON_FALLBACK not in _codes(sql)


def test_a_plan_with_no_photon_evidence_stays_silent() -> None:
    sql = "SELECT count(*) FROM samples.tpch.lineitem"
    summary = PlanSummary(
        root=PlanNode(op=PlanOp.SCAN, node_type="Scan", relation=RELATION),
        engine="databricks",
        sql=sql,
    )

    assert WarningCode.PHOTON_FALLBACK not in _codes(sql, summary=summary)


# -- qualification ----------------------------------------------------------


def test_an_under_qualified_table_is_a_correctness_warning() -> None:
    sql = "SELECT count(*) FROM lineitem WHERE l_shipdate > '1995-01-01'"

    evaluated = evaluate(_summary(sql=sql), analyze(sql, "databricks"), _facts(), CONFIG)

    warning = next(w for w in evaluated.warnings if w.code is WarningCode.UNQUALIFIED_RELATION)
    assert warning.suggested_rewrite == "qualify every table as catalog.schema.table"


def test_a_fully_qualified_query_earns_no_qualification_warning() -> None:
    sql = "SELECT count(*) FROM samples.tpch.lineitem WHERE l_shipdate > '1995-01-01'"

    assert WarningCode.UNQUALIFIED_RELATION not in _codes(sql)


def test_an_unparseable_query_produces_no_text_derived_warnings() -> None:
    sql = "SELECT count(*) FROM WHERE"

    assert WarningCode.UNQUALIFIED_RELATION not in _codes(sql)


# -- engine separation ------------------------------------------------------


def test_clickhouse_rules_do_not_fire_on_a_databricks_plan() -> None:
    sql = "SELECT count(*) FROM samples.tpch.lineitem WHERE l_orderkey = 42"

    codes = _codes(sql)

    # sort keys, projections and ClickHouse's build-side join heuristic are not
    # Delta mechanisms, and asserting them here would be advice about the wrong engine
    assert WarningCode.SORT_KEY_UNUSED not in codes
    assert WarningCode.PROJECTION_AVAILABLE_UNUSED not in codes
    assert WarningCode.JOIN_ORDER_SUSPECT not in codes


def test_the_full_scan_warning_counts_files_and_points_at_the_clustering_key() -> None:
    sql = "SELECT count(*) FROM samples.tpch.lineitem WHERE l_orderkey = 42"
    summary = _summary(sql=sql, files_total=1_000, files_selected=1_000)

    evaluated = evaluate(summary, analyze(sql, "databricks"), _facts(), CONFIG)

    warning = next(w for w in evaluated.warnings if w.code is WarningCode.FULL_SCAN)
    assert "100% of files were read" in warning.human_message
    assert warning.suggested_rewrite is not None
    assert "clustering key column, to skip files" in warning.suggested_rewrite


def test_a_scan_with_no_relation_name_does_not_answer_for_a_partitioned_table() -> None:
    # a plan that named no relation cannot testify about this table's pushdown
    sql = "SELECT count(*) FROM lineitem WHERE l_orderkey = 1"
    summary = PlanSummary(
        root=PlanNode(op=PlanOp.SCAN, node_type="Scan", relation=None, photon=True),
        engine="databricks",
        sql=sql,
        photon_coverage=1.0,
    )
    facts = {
        "lineitem": RelationFacts(
            layout=_layout(partition_by=("l_shipdate",)), column_ordinals=ORDINALS
        )
    }

    evaluated = evaluate(summary, analyze(sql, "databricks"), facts, CONFIG)

    assert WarningCode.MISSING_PARTITION_PREDICATE not in {w.code for w in evaluated.warnings}
