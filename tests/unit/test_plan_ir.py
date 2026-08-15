"""The plan IR carries pruning evidence from two engines without conflating them.

One number, two mechanisms: ClickHouse prunes granules and Databricks prunes
files. The unit travels with the ratio everywhere, because a reader who compares
a granule ratio to a file ratio has been misled by the report, not by the engine.
"""

from __future__ import annotations

from agentdb.adapters.models import QueryMetrics
from agentdb.core.plan_ir import PlanNode, PlanOp, PlanSummary, PlanWarning, Severity, WarningCode


def _delta_scan(**overrides: object) -> PlanNode:
    base: dict[str, object] = {
        "op": PlanOp.SCAN,
        "node_type": "PhotonScan parquet samples.tpch.lineitem",
        "relation": "samples.tpch.lineitem",
        "files_total": 1_000,
        "files_selected": 40,
        "photon": True,
    }
    return PlanNode(**{**base, **overrides})  # type: ignore[arg-type]


def test_a_delta_scan_reports_its_pruning_in_files() -> None:
    scan = _delta_scan()

    assert scan.pruning_unit == "file"
    assert scan.pruning_ratio == 0.04


def test_a_mergetree_scan_reports_its_pruning_in_granules() -> None:
    scan = PlanNode(
        op=PlanOp.SCAN,
        node_type="ReadFromMergeTree",
        relation="agentdb.hits",
        granules_total=1_000,
        granules_selected=10,
    )

    assert scan.pruning_unit == "granule"
    assert scan.pruning_ratio == 0.01


def test_a_scan_that_reported_no_pruning_evidence_says_so_rather_than_guessing() -> None:
    scan = PlanNode(op=PlanOp.SCAN, node_type="ReadFromRemote")

    assert scan.pruning_unit is None
    assert scan.pruning_ratio is None


def test_a_partial_file_count_is_not_turned_into_a_ratio() -> None:
    # files_total known, files_selected absent: the honest answer is "unknown"
    scan = _delta_scan(files_selected=None)

    assert scan.pruning_unit == "file"
    assert scan.pruning_ratio is None


def test_a_delta_scan_separates_filters_that_skip_files_from_filters_that_cannot() -> None:
    scan = _delta_scan(
        partition_filters=("l_shipdate >= '1995-01-01'",),
        pushed_filters=("l_orderkey > 100",),
        data_filters=("upper(l_comment) LIKE '%RUSH%'",),
        join_strategy="broadcast_hash",
    )

    assert scan.partition_filters == ("l_shipdate >= '1995-01-01'",)
    assert scan.pushed_filters == ("l_orderkey > 100",)
    assert scan.data_filters == ("upper(l_comment) LIKE '%RUSH%'",)
    assert scan.join_strategy == "broadcast_hash"


def _databricks_summary(**overrides: object) -> PlanSummary:
    base: dict[str, object] = {
        "root": _delta_scan(),
        "engine": "databricks",
        "sql": "SELECT count(*) FROM samples.tpch.lineitem",
        "pruning_ratio": 0.04,
        "pruning_unit": "file",
        "photon_coverage": 0.5,
    }
    return PlanSummary(**{**base, **overrides})  # type: ignore[arg-type]


def test_the_rendered_summary_names_the_unit_it_pruned_in() -> None:
    rendered = _databricks_summary().render()

    assert "- files read after pruning: 4.0% of those considered" in rendered
    assert "- plan nodes on Photon: 50%" in rendered


def test_a_summary_with_a_ratio_but_no_unit_does_not_invent_one() -> None:
    rendered = _databricks_summary(pruning_unit=None, photon_coverage=None).render()

    assert "- units read after pruning: 4.0%" in rendered
    assert "Photon" not in rendered


def test_warnings_are_attached_without_dropping_the_databricks_evidence() -> None:
    warning = PlanWarning(
        code=WarningCode.CLUSTERING_KEY_UNUSED,
        severity=Severity.WARNING,
        relation="samples.tpch.lineitem",
        human_message="no filtered column is in the clustering key",
    )

    carried = _databricks_summary().with_warnings((warning,))

    assert carried.pruning_unit == "file"
    assert carried.photon_coverage == 0.5
    assert carried.warnings == (warning,)
    assert "CLUSTERING_KEY_UNUSED" in carried.render()


# -- measured evidence -------------------------------------------------------
#
# On Databricks the estimate plan carries no file counts at all, so a summary
# only ever learns what was pruned after the query has run. Keeping the two
# apart is the point: one is a prediction, the other is a result.


def _summary(**overrides: object) -> PlanSummary:
    root = PlanNode(op=PlanOp.SCAN, node_type="PhotonScan", relation="lineitem")
    return PlanSummary(root=root, engine="databricks", sql="SELECT 1", **overrides)  # type: ignore[arg-type]


def _metrics(**overrides: object) -> QueryMetrics:
    values: dict[str, object] = {
        "statement_id": "01f1-abc",
        "engine": "databricks",
        "source": "query_history_api",
        "from_result_cache": False,
        "files_read": 3,
        "files_pruned": 37,
        "bytes_read": 1_000,
        "bytes_in_files_read": 10_000,
    }
    values.update(overrides)
    return QueryMetrics(**values)  # type: ignore[arg-type]


def test_a_measurement_replaces_an_estimate_and_says_that_it_did() -> None:
    summary = _summary(pruning_ratio=0.8, pruning_unit="file", pruning_source="estimated")

    updated = summary.with_measured(_metrics())

    assert updated.pruning_ratio == 0.075
    assert updated.pruning_source == "measured"
    assert updated.bytes_ratio == 0.1
    assert updated.measured_bytes_read == 1_000
    # the original is untouched: summaries are built, never mutated
    assert summary.pruning_ratio == 0.8


def test_a_plan_that_estimated_nothing_still_gets_the_measured_ratio() -> None:
    # every Photon plan observed live: no file counts in EXPLAIN whatsoever
    summary = _summary()

    updated = summary.with_measured(_metrics())

    assert updated.pruning_ratio == 0.075
    assert updated.pruning_unit == "file"


def test_a_cache_hit_leaves_the_estimate_alone_rather_than_zeroing_it() -> None:
    summary = _summary(pruning_ratio=0.8, pruning_unit="file", pruning_source="estimated")

    updated = summary.with_measured(_metrics(from_result_cache=True, files_read=0, files_pruned=0))

    assert updated.pruning_ratio == 0.8
    assert updated.pruning_source == "estimated"
    assert updated.measured_bytes_read is None


def test_a_clickhouse_measurement_keeps_the_granule_unit() -> None:
    summary = PlanSummary(
        root=PlanNode(op=PlanOp.SCAN, node_type="ReadFromMergeTree"),
        engine="clickhouse",
        sql="SELECT 1",
        pruning_unit="granule",
    )

    updated = summary.with_measured(_metrics(engine="clickhouse"))

    assert updated.pruning_unit == "granule"


def test_the_rendered_summary_labels_a_measured_ratio_as_measured() -> None:
    rendered = _summary().with_measured(_metrics()).render()

    assert "files read after pruning: 7.5% of those considered (measured)" in rendered
    assert "bytes fetched from the files read: 10.0%" in rendered
    assert "measured bytes read: 1,000" in rendered


def test_a_measured_byte_count_is_reported_instead_of_the_estimated_one() -> None:
    summary = _summary(estimated_bytes_read=999_999)

    assert "estimated bytes read: 999,999" in summary.render()
    assert "estimated bytes read" not in summary.with_measured(_metrics()).render()


def test_warnings_survive_a_measurement() -> None:
    warning = PlanWarning(
        code=WarningCode.FULL_SCAN, severity=Severity.WARNING, human_message="reads everything"
    )
    summary = _summary().with_warnings((warning,))

    assert summary.with_measured(_metrics()).warnings == (warning,)
