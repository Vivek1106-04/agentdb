"""The grounded context: what each ablation level contains, and how it reads.

Every assertion here is really an assertion about the benchmark. A level that
leaked a fact from the level above it, or a payload whose rendering drifted
between two runs of one arm, would make the published deltas meaningless.
"""

from __future__ import annotations

from typing import cast

import pytest

from agentdb.adapters import (
    Capability,
    ColumnDef,
    ColumnProfile,
    PhysicalLayout,
    Projection,
    RelationDetail,
    RelationRef,
    SkipIndex,
    UnsupportedCapabilityError,
)
from agentdb.adapters.models import SamplePolicy
from agentdb.config import Config
from agentdb.core import ContextBuilder, GroundedContext, GroundingLevel, RelationContext
from tests.fakes import FakeAdapter, clickhouse_hits_fixture

REF = RelationRef(namespace="agentdb", name="hits")

ProfileCall = tuple[RelationRef, tuple[str, ...], SamplePolicy]


def _profile_call(adapter: FakeAdapter) -> ProfileCall:
    """The arguments of the first ``column_profile`` call the builder made."""
    return cast(ProfileCall, adapter.calls_named("column_profile")[0])


# --------------------------------------------------------------------------
# the ladder
# --------------------------------------------------------------------------


async def test_schema_level_carries_the_ddl_and_nothing_else() -> None:
    adapter = clickhouse_hits_fixture()

    context = await ContextBuilder(adapter=adapter).build("agentdb", GroundingLevel.SCHEMA)
    rendered = context.render()

    assert "CREATE TABLE agentdb.hits" in rendered
    assert "sort key" not in rendered
    assert "Column profiles" not in rendered
    assert adapter.calls_named("column_profile") == []
    assert adapter.calls_named("physical_layout") == []


async def test_stats_level_adds_profiles_but_still_hides_the_physical_layout() -> None:
    adapter = clickhouse_hits_fixture()

    context = await ContextBuilder(adapter=adapter).build("agentdb", GroundingLevel.STATS)
    rendered = context.render()

    assert "Column profiles for agentdb.hits" in rendered
    assert "~42 distinct" in rendered
    assert "sort key" not in rendered
    assert adapter.calls_named("physical_layout") == []


async def test_layout_level_adds_the_facts_a_ddl_dump_cannot_convey() -> None:
    adapter = clickhouse_hits_fixture()

    context = await ContextBuilder(adapter=adapter).build("agentdb", GroundingLevel.LAYOUT)
    rendered = context.render()

    assert "sort key (ORDER BY): CounterID, EventDate, UserID" in rendered
    assert "prune granules" in rendered
    assert "partition key: toYYYYMM(EventDate)" in rendered
    assert "approx rows: 99,997,497" in rendered
    assert "Column profiles for agentdb.hits" in rendered


def test_the_ladder_is_cumulative() -> None:
    assert GroundingLevel.SCHEMA.includes_stats is False
    assert GroundingLevel.SCHEMA.includes_layout is False
    assert GroundingLevel.STATS.includes_stats is True
    assert GroundingLevel.STATS.includes_layout is False
    assert GroundingLevel.LAYOUT.includes_stats is True
    assert GroundingLevel.LAYOUT.includes_layout is True


# --------------------------------------------------------------------------
# what the builder asks the engine
# --------------------------------------------------------------------------


async def test_profiling_always_goes_through_a_bounded_sample_policy() -> None:
    adapter = clickhouse_hits_fixture()
    config = Config(default_sample_fraction=0.05, profile_max_rows=250_000, query_timeout_s=7)

    await ContextBuilder(adapter=adapter, config=config).build("agentdb", GroundingLevel.STATS)

    (_, _, policy) = _profile_call(adapter)
    assert policy.fraction == 0.05
    assert policy.max_rows == 250_000
    assert policy.timeout_s == 7


async def test_a_wide_table_is_profiled_up_to_the_budget_and_says_how_far_it_got() -> None:
    adapter = clickhouse_hits_fixture()
    config = Config(max_profiled_columns=2)

    context = await ContextBuilder(adapter=adapter, config=config).build(
        "agentdb", GroundingLevel.STATS
    )

    (_, columns, _) = _profile_call(adapter)
    assert columns == ("CounterID", "EventDate")
    assert context.relations[0].profiled_columns_available == 5


async def test_naming_relations_narrows_the_payload_without_listing_the_namespace() -> None:
    adapter = clickhouse_hits_fixture()

    context = await ContextBuilder(adapter=adapter).build(
        "agentdb", GroundingLevel.SCHEMA, relations=["hits"]
    )

    assert adapter.calls_named("list_relations") == []
    assert context.relations[0].ref == REF


async def test_a_relation_with_no_columns_is_not_profiled_at_all() -> None:
    empty = RelationDetail(ref=REF, columns=(), create_statement="CREATE TABLE agentdb.hits ()")
    adapter = FakeAdapter(details={str(REF): empty})

    context = await ContextBuilder(adapter=adapter).build(
        "agentdb", GroundingLevel.STATS, relations=["hits"]
    )

    assert adapter.calls_named("column_profile") == []
    assert context.relations[0].profiles == ()


async def test_a_level_the_engine_cannot_serve_is_refused_not_quietly_downgraded() -> None:
    adapter = FakeAdapter(capabilities=frozenset())

    with pytest.raises(UnsupportedCapabilityError) as caught:
        await ContextBuilder(adapter=adapter).build("agentdb", GroundingLevel.STATS)

    assert caught.value.capability is Capability.COLUMN_STATS


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


async def test_the_same_facts_render_byte_identically_every_time() -> None:
    first = await ContextBuilder(adapter=clickhouse_hits_fixture()).build(
        "agentdb", GroundingLevel.LAYOUT
    )
    second = await ContextBuilder(adapter=clickhouse_hits_fixture()).build(
        "agentdb", GroundingLevel.LAYOUT
    )

    assert first.render() == second.render()
    assert first.size_bytes == len(first.render().encode("utf-8"))


def test_a_partial_profile_says_how_many_columns_it_covered() -> None:
    detail = RelationDetail(
        ref=REF,
        columns=tuple(
            ColumnDef(name=f"c{i}", data_type="UInt8", is_nullable=False) for i in range(105)
        ),
        create_statement="CREATE TABLE agentdb.hits (...)",
    )
    context = GroundedContext(
        engine="clickhouse",
        namespace="agentdb",
        level=GroundingLevel.STATS,
        relations=(
            RelationContext(
                detail=detail,
                profiles=(
                    ColumnProfile(
                        name="c0", data_type="UInt8", sample_method="sample", sampled_rows=1_000
                    ),
                ),
                profiled_columns_available=105,
            ),
        ),
    )

    assert "(1 of 105 columns)" in context.render()
    assert "- c0: UInt8; [sample, 1,000 rows]" in context.render()


def test_a_profile_renders_every_fact_the_probe_returned() -> None:
    profile = ColumnProfile(
        name="SearchEngineID",
        data_type="UInt16",
        sample_method="sample",
        sampled_rows=999_974,
        approx_distinct=42,
        null_ratio=0.25,
        min_value="0",
        max_value="120",
        top_values=(("2", 500_000),),
    )
    context = GroundedContext(
        engine="clickhouse",
        namespace="agentdb",
        level=GroundingLevel.STATS,
        relations=(
            RelationContext(
                detail=RelationDetail(ref=REF, columns=(), create_statement="CREATE TABLE x"),
                profiles=(profile,),
                profiled_columns_available=1,
            ),
        ),
    )

    line = context.render().splitlines()[-1]
    assert line == (
        "- SearchEngineID: UInt16; ~42 distinct; 25% null; range 0..120; "
        "top: 2 (500,000); [sample, 999,974 rows]"
    )


def test_layout_rendering_omits_what_the_engine_did_not_report() -> None:
    bare = PhysicalLayout(engine="clickhouse", ref=REF, create_statement="CREATE TABLE x")
    context = GroundedContext(
        engine="clickhouse",
        namespace="agentdb",
        level=GroundingLevel.LAYOUT,
        relations=(
            RelationContext(
                detail=RelationDetail(ref=REF, columns=(), create_statement="CREATE TABLE x"),
                layout=bare,
            ),
        ),
    )

    rendered = context.render()
    assert rendered.endswith("Physical layout of agentdb.hits:")
    assert "sort key" not in rendered
    assert "none" not in rendered


def test_layout_rendering_covers_every_kind_of_physical_object() -> None:
    layout = PhysicalLayout(
        engine="clickhouse",
        ref=REF,
        create_statement="CREATE TABLE x",
        table_engine="MergeTree",
        order_by=("CounterID",),
        partition_by=("toYYYYMM(EventDate)",),
        primary_key=("CounterID",),
        sampling_key="intHash32(UserID)",
        skip_indexes=(
            SkipIndex(name="idx_url", index_type="bloom_filter", expression="URL", granularity=4),
        ),
        projections=(Projection(name="by_user", query="SELECT UserID, count()"),),
        approx_rows=1_000,
        compression_ratio=4.2,
    )
    context = GroundedContext(
        engine="clickhouse",
        namespace="agentdb",
        level=GroundingLevel.LAYOUT,
        relations=(
            RelationContext(
                detail=RelationDetail(ref=REF, columns=(), create_statement="CREATE TABLE x"),
                layout=layout,
            ),
        ),
    )

    rendered = context.render()
    assert "- engine: MergeTree" in rendered
    assert "- sampling key: intHash32(UserID)" in rendered
    assert "- skip index idx_url: bloom_filter on URL (granularity 4)" in rendered
    assert "- projection by_user: SELECT UserID, count()" in rendered
    assert "- compression ratio: 4.2x" in rendered


def test_layout_renders_the_delta_facts_a_schema_dump_hides() -> None:
    # Arrange — a Databricks table whose statistics stop before the filtered column
    ref = RelationRef(catalog="samples", namespace="tpch", name="lineitem")
    layout = PhysicalLayout(
        engine="databricks",
        ref=ref,
        create_statement="CREATE TABLE samples.tpch.lineitem (...) USING delta",
        table_format="delta",
        clustering_columns=("l_shipdate", "l_orderkey"),
        zorder_columns=("l_partkey",),
        stats_indexed_columns=32,
        num_files=4_096,
        avg_file_bytes=8 * 1024 * 1024,
        deletion_vectors_enabled=True,
    )
    context = GroundedContext(
        engine="databricks",
        namespace="tpch",
        level=GroundingLevel.LAYOUT,
        relations=(
            RelationContext(
                detail=RelationDetail(ref=ref, columns=(), create_statement="CREATE TABLE x"),
                layout=layout,
            ),
        ),
    )

    # Act
    rendered = context.render()

    # Assert
    assert "- table format: delta" in rendered
    assert "- clustering key (CLUSTER BY): l_shipdate, l_orderkey" in rendered
    assert "- legacy Z-ORDER columns: l_partkey" in rendered
    assert "first 32 columns in schema order" in rendered
    assert "- files: 4,096, averaging 8.0 MiB" in rendered
    assert "deletion vectors are enabled" in rendered


def test_file_count_renders_without_an_average_size_when_the_engine_reported_none() -> None:
    layout = PhysicalLayout(
        engine="databricks",
        ref=RelationRef(catalog="main", namespace="tpch", name="orders"),
        create_statement="CREATE TABLE main.tpch.orders (...) USING delta",
        num_files=12,
    )
    context = GroundedContext(
        engine="databricks",
        namespace="tpch",
        level=GroundingLevel.LAYOUT,
        relations=(
            RelationContext(
                detail=RelationDetail(
                    ref=layout.ref, columns=(), create_statement="CREATE TABLE x"
                ),
                layout=layout,
            ),
        ),
    )

    rendered = context.render()
    assert "- files: 12" in rendered
    assert "averaging" not in rendered


def test_layout_prefers_an_explicit_statistics_column_list_over_the_ordinal_rule() -> None:
    layout = PhysicalLayout(
        engine="databricks",
        ref=RelationRef(catalog="main", namespace="tpch", name="orders"),
        create_statement="CREATE TABLE main.tpch.orders (...) USING delta",
        stats_indexed_columns=32,
        stats_columns=("o_orderdate", "o_custkey"),
    )
    context = GroundedContext(
        engine="databricks",
        namespace="tpch",
        level=GroundingLevel.LAYOUT,
        relations=(
            RelationContext(
                detail=RelationDetail(
                    ref=layout.ref, columns=(), create_statement="CREATE TABLE x"
                ),
                layout=layout,
            ),
        ),
    )

    rendered = context.render()
    assert "collected for: o_orderdate, o_custkey" in rendered
    assert "first 32 columns" not in rendered
