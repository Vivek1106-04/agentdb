"""The Databricks advisor's rules, and the ones it refuses to port (SPEC §9.2).

The asymmetry with the ClickHouse advisor is the thing under test as much as the
advice is. A reader who finds cardinality ordering here, or a confident file-count
estimate where Delta reports none, has found a rule copied rather than reasoned
about — and the cross-engine comparison stops meaning anything at that point.
"""

from __future__ import annotations

from agentdb.adapters import ColumnDef, ColumnProfile, PhysicalLayout, RelationDetail, RelationRef
from agentdb.config import Config
from agentdb.core.advisor import Confidence, DatabricksAdvisor, Kind, demand_from_queries
from agentdb.core.advisor.base import Demand, Recommendation
from agentdb.core.plan_ir import (
    PlanNode,
    PlanOp,
    PlanSummary,
    PlanWarning,
    Severity,
    WarningCode,
)
from agentdb.core.query_shape import analyze

LINEITEM = RelationRef(catalog="samples", namespace="tpch", name="lineitem")

COLUMNS = (
    "l_orderkey",
    "l_partkey",
    "l_suppkey",
    "l_linenumber",
    "l_quantity",
    "l_extendedprice",
    "l_discount",
    "l_tax",
    "l_returnflag",
    "l_shipdate",
    "l_comment",
)


def detail() -> RelationDetail:
    return RelationDetail(
        ref=LINEITEM,
        columns=tuple(
            ColumnDef(name=name, data_type="bigint", is_nullable=False) for name in COLUMNS
        ),
        create_statement="CREATE TABLE samples.tpch.lineitem (...)",
    )


def profile(name: str, distinct: int) -> ColumnProfile:
    return ColumnProfile(
        name=name,
        data_type="date",
        sample_method="sample",
        sampled_rows=100_000,
        approx_distinct=distinct,
    )


def layout(**overrides: object) -> PhysicalLayout:
    fields: dict[str, object] = {
        "engine": "databricks",
        "ref": LINEITEM,
        "create_statement": "CREATE TABLE samples.tpch.lineitem (...)",
        "table_format": "delta",
        "approx_rows": 30_000_000,
        "num_files": 40,
        "avg_file_bytes": 60 * 1024 * 1024,
        "stats_indexed_columns": 4,
    }
    fields.update(overrides)
    return PhysicalLayout(**fields)  # type: ignore[arg-type]  # test factory, keyed by field name


def demand_for(*queries: str) -> Demand:
    return demand_from_queries("lineitem", [analyze(sql, "databricks") for sql in queries])


def plan_with(*codes: WarningCode) -> PlanSummary:
    return PlanSummary(
        root=PlanNode(op=PlanOp.SCAN, node_type="PhotonScan", relation="lineitem"),
        engine="databricks",
        sql="SELECT 1",
        warnings=tuple(
            PlanWarning(code=code, severity=Severity.WARNING, human_message="observed in the plan")
            for code in codes
        ),
    )


def advise(
    *queries: str,
    profiles: list[ColumnProfile] | None = None,
    physical: PhysicalLayout | None = None,
    plan: PlanSummary | None = None,
    config: Config | None = None,
) -> tuple[Recommendation, ...]:
    return DatabricksAdvisor(config=config or Config()).advise(
        ref=LINEITEM,
        layout=physical or layout(),
        detail=detail(),
        profiles=profiles or [profile("l_shipdate", 2_500)],
        demand=demand_for(*queries),
        plan=plan,
    )


def of_kind(found: tuple[Recommendation, ...], kind: Kind) -> list[Recommendation]:
    return [item for item in found if item.kind is kind]


# --------------------------------------------------------------------------
# A. data-skipping statistics — the headline
# --------------------------------------------------------------------------


def test_a_filter_past_the_indexed_column_limit_is_reported_as_unskippable() -> None:
    """l_shipdate is the tenth column; statistics stop at the fourth."""
    found = advise("SELECT count(*) FROM lineitem WHERE l_shipdate > '1995-01-01'")

    stats = of_kind(found, Kind.STATS_COLUMNS)[0]
    assert "l_shipdate" in stats.rationale
    assert "skips no files at all" in stats.rationale
    assert "delta.dataSkippingStatsColumns" in (stats.ddl or "")
    assert "OPTIMIZE" in (stats.ddl or "")


def test_the_non_retroactivity_of_the_property_is_stated_not_implied() -> None:
    stats = of_kind(
        advise("SELECT count(*) FROM lineitem WHERE l_shipdate > '1995-01-01'"),
        Kind.STATS_COLUMNS,
    )[0]

    assert any("not retroactive" in note for note in stats.risk_notes)
    assert any("write cost" in note for note in stats.risk_notes)


def test_a_filter_inside_the_indexed_prefix_needs_no_advice() -> None:
    found = advise("SELECT count(*) FROM lineitem WHERE l_orderkey = 42")

    assert of_kind(found, Kind.STATS_COLUMNS) == []


def test_an_explicit_statistics_list_is_honoured_over_the_default_prefix() -> None:
    explicit = layout(stats_columns=("l_shipdate", "l_orderkey"))

    found = advise(
        "SELECT count(*) FROM lineitem WHERE l_shipdate > '1995-01-01'", physical=explicit
    )

    assert of_kind(found, Kind.STATS_COLUMNS) == []


def test_an_explicit_list_that_omits_a_filtered_column_names_the_list_it_read() -> None:
    """The rationale quotes the property, so a reader can check it against the table."""
    explicit = layout(stats_columns=("l_orderkey", "l_partkey"))

    stats = of_kind(
        advise("SELECT count(*) FROM lineitem WHERE l_shipdate > '1995-01-01'", physical=explicit),
        Kind.STATS_COLUMNS,
    )[0]

    assert "delta.dataSkippingStatsColumns = l_orderkey, l_partkey" in stats.rationale
    assert "l_shipdate" in (stats.ddl or "")


def test_a_table_that_reports_no_coverage_at_all_earns_no_guess() -> None:
    unknown = layout(stats_indexed_columns=None, stats_columns=None)

    found = advise(
        "SELECT count(*) FROM lineitem WHERE l_shipdate > '1995-01-01'", physical=unknown
    )

    assert of_kind(found, Kind.STATS_COLUMNS) == []


def test_the_statistics_recommendation_does_not_claim_a_number_it_cannot_derive() -> None:
    stats = of_kind(
        advise("SELECT count(*) FROM lineitem WHERE l_shipdate > '1995-01-01'"),
        Kind.STATS_COLUMNS,
    )[0]

    assert stats.confidence is Confidence.HEURISTIC
    assert stats.expected_effect.after is None
    assert "not estimated" in stats.expected_effect.method


# --------------------------------------------------------------------------
# B. liquid clustering — where the ClickHouse rule is deliberately not ported
# --------------------------------------------------------------------------


def test_clustering_ranks_by_filter_frequency_not_by_lowest_cardinality() -> None:
    """The §9.1.A rule exists for a sparse primary index and does not transfer."""
    found = advise(
        "SELECT count(*) FROM lineitem WHERE l_shipdate > '1995-01-01'",
        "SELECT count(*) FROM lineitem WHERE l_shipdate > '1996-01-01'",
        "SELECT count(*) FROM lineitem WHERE l_returnflag = 'R'",
        profiles=[profile("l_shipdate", 2_500), profile("l_returnflag", 3)],
    )

    cluster = of_kind(found, Kind.CLUSTER_BY)[0]
    assert (cluster.ddl or "").startswith(
        "ALTER TABLE samples.tpch.lineitem CLUSTER BY (l_shipdate"
    )
    assert "deliberately not the cardinality-ordering rule" in cluster.rationale


def test_the_clustering_key_is_capped() -> None:
    columns = ("l_orderkey", "l_partkey", "l_suppkey", "l_quantity", "l_discount")
    query = "SELECT count(*) FROM lineitem WHERE " + " AND ".join(f"{name} = 1" for name in columns)

    found = advise(
        query,
        profiles=[profile(name, 1_000) for name in columns],
        config=Config(clustering_key_max_columns=2),
    )

    cluster = of_kind(found, Kind.CLUSTER_BY)[0]
    assert (cluster.ddl or "").count(",") == 1, "two columns, one separator"


def test_a_managed_table_says_the_engine_may_already_be_tuning_it() -> None:
    """Advising against a tuner that is already working is the §3.1 failure in miniature."""
    found = advise(
        "SELECT count(*) FROM lineitem WHERE l_shipdate > '1995-01-01'",
        physical=layout(is_managed=True),
    )

    cluster = of_kind(found, Kind.CLUSTER_BY)[0]
    assert any("predictive optimization" in note for note in cluster.risk_notes)
    assert "managed" in cluster.rationale


def test_a_zordered_table_is_told_the_two_are_mutually_exclusive() -> None:
    found = advise(
        "SELECT count(*) FROM lineitem WHERE l_shipdate > '1995-01-01'",
        physical=layout(zorder_columns=("l_orderkey",)),
    )

    cluster = of_kind(found, Kind.CLUSTER_BY)[0]
    assert "mutually exclusive" in cluster.rationale
    assert any("one-way" in note for note in cluster.risk_notes)


def test_a_clustering_key_that_already_matches_earns_nothing() -> None:
    found = advise(
        "SELECT count(*) FROM lineitem WHERE l_shipdate > '1995-01-01'",
        physical=layout(clustering_columns=("l_shipdate",)),
    )

    assert of_kind(found, Kind.CLUSTER_BY) == []


def test_a_column_with_no_profile_is_not_proposed_as_a_clustering_key() -> None:
    found = advise("SELECT count(*) FROM lineitem WHERE l_comment = 'x'", profiles=[])

    assert of_kind(found, Kind.CLUSTER_BY) == []


# --------------------------------------------------------------------------
# C. compaction
# --------------------------------------------------------------------------


def test_many_small_files_earn_a_compaction_recommendation() -> None:
    fragmented = layout(num_files=5_000, avg_file_bytes=2 * 1024 * 1024)

    found = advise("SELECT count(*) FROM lineitem WHERE l_orderkey = 1", physical=fragmented)

    compaction = of_kind(found, Kind.COMPACTION)[0]
    assert "5,000 files" in compaction.rationale
    assert (compaction.ddl or "").startswith("OPTIMIZE samples.tpch.lineitem;")
    assert "delta.targetFileSize" in (compaction.ddl or "")


def test_compaction_projects_a_file_count_and_never_a_latency() -> None:
    fragmented = layout(num_files=5_000, avg_file_bytes=2 * 1024 * 1024)

    compaction = of_kind(
        advise("SELECT count(*) FROM lineitem WHERE l_orderkey = 1", physical=fragmented),
        Kind.COMPACTION,
    )[0]

    assert compaction.expected_effect.metric == "files_read"
    assert compaction.expected_effect.after is not None
    assert compaction.expected_effect.after < compaction.expected_effect.before  # type: ignore[operator]  # both set
    assert "no claim about latency" in compaction.expected_effect.method


def test_healthy_file_sizes_earn_no_compaction_advice() -> None:
    found = advise("SELECT count(*) FROM lineitem WHERE l_orderkey = 1")

    assert of_kind(found, Kind.COMPACTION) == []


def test_a_table_that_reports_no_file_counts_is_left_alone() -> None:
    silent = layout(num_files=None, avg_file_bytes=None)

    found = advise("SELECT count(*) FROM lineitem WHERE l_orderkey = 1", physical=silent)

    assert of_kind(found, Kind.COMPACTION) == []


# --------------------------------------------------------------------------
# D. join strategy
# --------------------------------------------------------------------------


def test_a_suspect_join_is_told_to_collect_statistics_before_pinning_a_hint() -> None:
    found = advise(
        "SELECT count(*) FROM lineitem WHERE l_orderkey = 1",
        plan=plan_with(WarningCode.JOIN_ORDER_SUSPECT),
    )

    join = of_kind(found, Kind.BROADCAST_HINT)[0]
    assert (join.ddl or "").startswith("ANALYZE TABLE")
    assert "BROADCAST" in (join.rewritten_sql or "")
    assert any("outlives the conditions" in note for note in join.risk_notes)


def test_no_join_warning_means_no_join_advice() -> None:
    found = advise("SELECT count(*) FROM lineitem WHERE l_orderkey = 1", plan=plan_with())

    assert of_kind(found, Kind.BROADCAST_HINT) == []


def test_without_a_plan_there_is_no_join_evidence_to_reason_from() -> None:
    found = advise("SELECT count(*) FROM lineitem WHERE l_orderkey = 1")

    assert of_kind(found, Kind.BROADCAST_HINT) == []
