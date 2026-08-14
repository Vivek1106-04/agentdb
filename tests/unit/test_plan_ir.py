"""The plan IR carries pruning evidence from two engines without conflating them.

One number, two mechanisms: ClickHouse prunes granules and Databricks prunes
files. The unit travels with the ratio everywhere, because a reader who compares
a granule ratio to a file ratio has been misled by the report, not by the engine.
"""

from __future__ import annotations

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
