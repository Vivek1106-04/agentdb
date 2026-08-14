"""Value objects are the boundary between engines and core: validate them hard."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

from agentdb.adapters import (
    ColumnDef,
    ColumnProfile,
    DialectRules,
    ExplainMode,
    Limits,
    PhysicalLayout,
    Projection,
    RawPlan,
    Relation,
    RelationDetail,
    RelationRef,
    ResultSet,
    SamplePolicy,
    SkipIndex,
    TimeWindow,
    WorkloadEntry,
)
from agentdb.adapters.models import MAX_TOP_VALUES

HITS = RelationRef(namespace="agentdb", name="hits")


def test_relation_ref_renders_qualified() -> None:
    assert str(HITS) == "agentdb.hits"


def test_value_objects_are_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        HITS.name = "other"  # type: ignore[misc]


def test_column_def_reports_compression_ratio() -> None:
    # Arrange
    column = ColumnDef(
        name="URL",
        data_type="String",
        is_nullable=False,
        compressed_bytes=1_000,
        uncompressed_bytes=8_000,
    )

    # Act / Assert
    assert column.compression_ratio == 8.0


@pytest.mark.parametrize(
    ("compressed", "uncompressed"),
    [(None, 8_000), (1_000, None), (0, 8_000)],
)
def test_compression_ratio_is_none_when_unknowable(
    compressed: int | None, uncompressed: int | None
) -> None:
    column = ColumnDef(
        name="URL",
        data_type="String",
        is_nullable=False,
        compressed_bytes=compressed,
        uncompressed_bytes=uncompressed,
    )
    assert column.compression_ratio is None


def test_relation_detail_lists_column_names() -> None:
    detail = RelationDetail(
        ref=HITS,
        columns=(
            ColumnDef(name="EventDate", data_type="Date", is_nullable=False),
            ColumnDef(name="UserID", data_type="UInt64", is_nullable=False),
        ),
        create_statement="CREATE TABLE hits (...) ENGINE = MergeTree",
    )
    assert detail.column_names == ("EventDate", "UserID")


def test_relation_carries_only_cheap_facts() -> None:
    relation = Relation(
        ref=HITS,
        kind="table",
        engine_type="MergeTree",
        approx_rows=99_997_497,
        on_disk_bytes=14_000_000_000,
    )
    assert relation.engine_type == "MergeTree"
    assert relation.comment is None


def _layout(**overrides: object) -> PhysicalLayout:
    base: dict[str, object] = {
        "engine": "clickhouse",
        "ref": HITS,
        "create_statement": (
            "CREATE TABLE hits (...) ENGINE = MergeTree ORDER BY (CounterID, EventDate)"
        ),
        "table_engine": "MergeTree",
        "order_by": ("CounterID", "EventDate"),
    }
    return PhysicalLayout(**{**base, **overrides})  # type: ignore[arg-type]


def test_layout_exposes_the_column_that_decides_pruning() -> None:
    assert _layout().leading_sort_column == "CounterID"


def test_layout_without_a_sort_key_has_no_leading_column() -> None:
    assert _layout(order_by=None).leading_sort_column is None
    assert _layout(order_by=()).leading_sort_column is None


def test_layout_reports_whether_sample_is_available() -> None:
    assert _layout().is_sampleable is False
    assert _layout(sampling_key="cityHash64(UserID)").is_sampleable is True


def test_layout_holds_clickhouse_physical_design() -> None:
    layout = _layout(
        skip_indexes=(
            SkipIndex(name="idx_url", index_type="bloom_filter", expression="URL", granularity=4),
        ),
        projections=(Projection(name="proj_by_date", query="SELECT EventDate, count()"),),
    )
    assert layout.skip_indexes[0].index_type == "bloom_filter"
    assert layout.projections[0].name == "proj_by_date"


def test_layout_holds_delta_physical_design() -> None:
    layout = PhysicalLayout(
        engine="databricks",
        ref=RelationRef(catalog="samples", namespace="tpch", name="lineitem"),
        create_statement="CREATE TABLE samples.tpch.lineitem (...) USING delta",
        table_format="delta",
        clustering_columns=("l_shipdate",),
        zorder_columns=("l_partkey",),
        is_managed=True,
        deletion_vectors_enabled=True,
        num_files=4_096,
        avg_file_bytes=8_388_608.0,
    )
    assert layout.clustering_columns == ("l_shipdate",)
    assert layout.leading_sort_column is None
    assert layout.is_sampleable is False


def test_delta_statistics_stop_at_the_indexed_column_count() -> None:
    # Arrange — the table says nothing, so the Delta default applies
    layout = PhysicalLayout(
        engine="databricks",
        ref=RelationRef(catalog="main", namespace="tpch", name="wide"),
        create_statement="CREATE TABLE main.tpch.wide (...) USING delta",
    )

    # Act / Assert — column 40 of a wide table cannot skip a single file
    assert layout.has_file_statistics("c1", 1, default_indexed=32) is True
    assert layout.has_file_statistics("c40", 40, default_indexed=32) is False


def test_delta_statistics_honour_the_table_property_over_the_default() -> None:
    layout = PhysicalLayout(
        engine="databricks",
        ref=RelationRef(catalog="main", namespace="tpch", name="wide"),
        create_statement="CREATE TABLE main.tpch.wide (...) USING delta",
        stats_indexed_columns=64,
    )
    assert layout.has_file_statistics("c40", 40, default_indexed=32) is True


def test_an_explicit_statistics_column_list_overrides_the_ordinal_rule() -> None:
    layout = PhysicalLayout(
        engine="databricks",
        ref=RelationRef(catalog="main", namespace="tpch", name="wide"),
        create_statement="CREATE TABLE main.tpch.wide (...) USING delta",
        stats_indexed_columns=32,
        stats_columns=("c40",),
    )
    assert layout.has_file_statistics("c40", 40, default_indexed=32) is True
    assert layout.has_file_statistics("c1", 1, default_indexed=32) is False


def test_a_relation_ref_carries_the_three_level_databricks_namespace() -> None:
    two_part = RelationRef(namespace="agentdb", name="hits")
    three_part = RelationRef(catalog="samples", namespace="tpch", name="lineitem")

    assert str(two_part) == "agentdb.hits"
    assert two_part.parts == ("agentdb", "hits")
    assert str(three_part) == "samples.tpch.lineitem"
    assert three_part.parts == ("samples", "tpch", "lineitem")


def test_profile_flags_low_cardinality_against_a_configured_threshold() -> None:
    # Arrange
    profile = ColumnProfile(
        name="SearchEngineID",
        data_type="UInt16",
        sample_method="sample",
        sampled_rows=1_000_000,
        approx_distinct=42,
    )

    # Act / Assert — the threshold is config, so it is an argument, not a stored flag
    assert profile.is_low_cardinality(threshold=10_000) is True
    assert profile.is_low_cardinality(threshold=10) is False


def test_profile_without_a_distinct_estimate_is_never_low_cardinality() -> None:
    profile = ColumnProfile(
        name="URL", data_type="String", sample_method="unavailable", sampled_rows=0
    )
    assert profile.is_low_cardinality(threshold=10_000) is False


@pytest.mark.parametrize(
    ("method", "is_estimate"),
    [("full", False), ("sample", True), ("system_table", True), ("unavailable", True)],
)
def test_profile_says_plainly_whether_it_is_an_estimate(method: str, is_estimate: bool) -> None:
    profile = ColumnProfile(
        name="UserID",
        data_type="UInt64",
        sample_method=method,  # type: ignore[arg-type]
        sampled_rows=10,
    )
    assert profile.is_estimate is is_estimate


def test_profile_rejects_negative_sample_size() -> None:
    with pytest.raises(ValueError, match="sampled_rows must be >= 0"):
        ColumnProfile(name="c", data_type="UInt8", sample_method="sample", sampled_rows=-1)


@pytest.mark.parametrize("ratio", [-0.01, 1.01])
def test_profile_rejects_impossible_null_ratio(ratio: float) -> None:
    with pytest.raises(ValueError, match=r"null_ratio must be in \[0, 1\]"):
        ColumnProfile(
            name="c",
            data_type="UInt8",
            sample_method="sample",
            sampled_rows=10,
            null_ratio=ratio,
        )


def test_profile_caps_top_values_at_the_probe_width() -> None:
    too_many = tuple((str(i), i) for i in range(MAX_TOP_VALUES + 1))
    with pytest.raises(ValueError, match="top_values holds at most 10"):
        ColumnProfile(
            name="c",
            data_type="UInt8",
            sample_method="sample",
            sampled_rows=10,
            top_values=too_many,
        )


def test_raw_plan_keeps_the_statements_that_produced_it() -> None:
    plan = RawPlan(
        engine="clickhouse",
        mode=ExplainMode.ESTIMATE,
        sql="SELECT 1",
        payload="Expression",
        statements=("EXPLAIN indexes = 1 SETTINGS use_query_condition_cache = 0 SELECT 1",),
    )
    assert "use_query_condition_cache = 0" in plan.statements[0]


def test_result_set_rejects_rows_that_do_not_match_its_columns() -> None:
    with pytest.raises(ValueError, match="row has 1 values but result declares 2 columns"):
        ResultSet(columns=("a", "b"), rows=((1,),), row_count=1, truncated=False)


def test_result_set_carries_engine_intrinsic_efficiency_measures() -> None:
    result = ResultSet(
        columns=("n",),
        rows=((1,),),
        row_count=1,
        truncated=False,
        rows_read=99_997_497,
        bytes_read=14_000_000_000,
    )
    assert result.bytes_read == 14_000_000_000
    assert result.truncated is False


def test_workload_entry_holds_a_concrete_instance_for_the_advisor() -> None:
    entry = WorkloadEntry(
        normalized_sql="SELECT count() FROM hits WHERE CounterID = ?",
        calls=500,
        sample_sql="SELECT count() FROM hits WHERE CounterID = 62",
    )
    assert entry.sample_sql is not None
    assert entry.calls == 500


def test_dialect_quotes_identifiers_per_engine() -> None:
    clickhouse = DialectRules(engine="clickhouse", version="25.9", identifier_quote="`")
    databricks = DialectRules(engine="databricks", version="16.4", identifier_quote='"')

    assert clickhouse.quote_identifier("Order") == "`Order`"
    assert databricks.quote_identifier("Order") == '"Order"'


def test_dialect_escapes_embedded_quotes() -> None:
    rules = DialectRules(engine="clickhouse", version="25.9", identifier_quote="`")
    assert rules.quote_identifier("we`ird") == "`we``ird`"


@pytest.mark.parametrize(
    ("identifier", "needs"),
    [
        ("user_id", False),
        ("_leading", False),
        ("UserID2", False),
        ("", True),
        ("2fast", True),
        ("with space", True),
        ("select", True),
    ],
)
def test_dialect_knows_which_identifiers_must_be_quoted(identifier: str, needs: bool) -> None:
    rules = DialectRules(
        engine="databricks",
        version="16.4",
        identifier_quote='"',
        reserved_words=frozenset({"SELECT"}),
    )
    assert rules.needs_quoting(identifier) is needs


def test_limits_reject_unbounded_configurations() -> None:
    with pytest.raises(ValueError, match="timeout_s must be > 0"):
        Limits(timeout_s=0, max_result_rows=10)
    with pytest.raises(ValueError, match="max_result_rows must be > 0"):
        Limits(timeout_s=1, max_result_rows=0)


def test_limits_accept_a_bounded_configuration() -> None:
    limits = Limits(timeout_s=30, max_result_rows=10_000, max_bytes_to_read=1 << 30)
    assert limits.max_rows_to_read is None
    assert limits.max_bytes_to_read == 1 << 30


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"fraction": 0.0}, r"fraction must be in \(0, 1\]"),
        ({"fraction": 1.5}, r"fraction must be in \(0, 1\]"),
        ({"max_rows": 0}, "max_rows must be > 0"),
        ({"timeout_s": 0}, "timeout_s must be > 0"),
    ],
)
def test_sample_policy_never_permits_an_accidental_full_scan(
    kwargs: dict[str, float], message: str
) -> None:
    base: dict[str, float] = {"fraction": 0.01, "max_rows": 1_000_000, "timeout_s": 30}
    with pytest.raises(ValueError, match=message):
        SamplePolicy(**{**base, **kwargs})  # type: ignore[arg-type]


def test_time_window_reports_its_length_in_minutes() -> None:
    window = TimeWindow(
        start=datetime(2026, 8, 6, 10, 0, tzinfo=UTC),
        end=datetime(2026, 8, 6, 11, 30, tzinfo=UTC),
    )
    assert window.minutes == 90.0
    assert window.duration.total_seconds() == 5400.0


def test_time_window_must_move_forwards() -> None:
    moment = datetime(2026, 8, 6, 10, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="must be after start"):
        TimeWindow(start=moment, end=moment)
