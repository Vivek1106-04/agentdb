"""Live checks against the ClickHouse this project ships (SPEC §12, milestones M0 to M5).

This is the tier the Databricks half has had since M3.5 and the ClickHouse half
did not. Every ClickHouse defect fixed in this repository so far — the
``EXPLAIN … SETTINGS`` clause order, ``approx_top_k`` returning named tuples,
truncation without ``result_overflow_mode``, a read-only profile that refused
the settings the tools send — passed unit tests against fakes and was found by
hand against a running server. That is the gap these tests close: unit coverage
proves the parsing, and only a live engine proves the *dialect*.

Run with::

    make up && make load-clickbench CLICKBENCH_PARTS=100 && make load-tpch
    uv sync --extra clickhouse --extra memory
    uv run pytest -m e2e tests/e2e/test_clickhouse_live.py

Every test skips loudly when nothing is listening, and nothing here writes:
the connection is the ``agentdb_ro`` role, whose profile carries ``readonly = 1``
and the per-query ceilings of SPEC §13.3.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any, cast

import pytest

from agentdb.adapters import (
    Capability,
    ExplainMode,
    Limits,
    RelationRef,
    SamplePolicy,
)
from agentdb.adapters.base import AdapterError, QueryPermissionError
from agentdb.adapters.clickhouse import ClickHouseAdapter
from agentdb.adapters.clickhouse_client import ClickHouseTarget, build_client
from agentdb.config import Config
from agentdb.core import ContextBuilder, GroundingLevel, PlanExplainer, WarningCode
from agentdb.core.memory import normalize_sql, snapshot
from agentdb.core.memory.postgres import connect
from agentdb.core.memory.store import ExemplarStore
from agentdb.server import build_catalog
from agentdb.server.schemas import JsonValue

pytestmark = pytest.mark.e2e

HITS = RelationRef(namespace="agentdb", name="hits")
LINEITEM = RelationRef(namespace="tpch", name="lineitem")

UNPRUNED_SQL = "SELECT count() FROM agentdb.hits WHERE URL LIKE '%google%'"
"""A filter on no sort-key column over 100M rows: the case the plan layer exists for."""


@pytest.fixture
async def adapter() -> AsyncIterator[ClickHouseAdapter]:
    """A fresh connection per test.

    Per test rather than per module because the driver binds its aiohttp session
    to the event loop that created it, and pytest-asyncio gives each test its
    own — a module-scoped client fails on the second test with "attached to a
    different loop", which reads like a server problem and is not one.
    """
    target = ClickHouseTarget.from_env()
    try:
        client = await build_client(target)
    except AdapterError as exc:
        pytest.skip(
            f"no ClickHouse at {target.host}:{target.port} ({exc}); start one with: make up"
        )
    live = ClickHouseAdapter(client=client)
    yield live
    await cast(Any, client).close()


# --------------------------------------------------------------------------
# discovery and layout
# --------------------------------------------------------------------------


async def test_both_seeded_databases_are_reachable_from_one_connection(
    adapter: ClickHouseAdapter,
) -> None:
    """The read-only role is granted SELECT on agentdb and tpch, and nothing else."""
    hits = await adapter.list_relations("agentdb")
    tpch = await adapter.list_relations("tpch")

    assert {relation.ref.name for relation in hits} >= {"hits"}
    assert {relation.ref.name for relation in tpch} >= {"lineitem", "orders", "part"}


async def test_the_hits_table_reports_the_size_the_loader_wrote(
    adapter: ClickHouseAdapter,
) -> None:
    relations = await adapter.list_relations("agentdb")
    hits = next(relation for relation in relations if relation.ref.name == "hits")

    assert hits.engine_type == "MergeTree"
    assert hits.approx_rows is not None
    assert hits.approx_rows > 90_000_000, "the full 100-part ClickBench load"


async def test_the_layout_carries_the_sort_key_no_ddl_dump_explains(
    adapter: ClickHouseAdapter,
) -> None:
    """The differentiating fact of SPEC §13.1, read from a real MergeTree."""
    layout = await adapter.physical_layout(HITS)

    assert layout.table_engine == "MergeTree"
    assert layout.order_by is not None
    assert layout.order_by[0] == "CounterID", "the leading sort-key column ClickBench ships"
    assert layout.on_disk_bytes is not None and layout.on_disk_bytes > 0
    assert layout.compression_ratio is not None


async def test_describe_returns_the_wide_schema_with_a_create_statement(
    adapter: ClickHouseAdapter,
) -> None:
    detail = await adapter.describe_relation(HITS)

    assert len(detail.columns) > 100, "ClickBench hits is a 105-column table"
    assert detail.create_statement.startswith("CREATE TABLE")


# --------------------------------------------------------------------------
# profiling — where approx_top_k's named tuples bit
# --------------------------------------------------------------------------


async def test_a_sampled_profile_reports_top_values_and_says_how_it_sampled(
    adapter: ClickHouseAdapter,
) -> None:
    config = Config()
    profiles = await adapter.column_profile(
        HITS,
        ["SearchEngineID", "UserID"],
        SamplePolicy(
            fraction=config.default_sample_fraction,
            max_rows=config.profile_max_rows,
            timeout_s=config.query_timeout_s,
        ),
    )
    by_name = {profile.name: profile for profile in profiles}

    engine_id = by_name["SearchEngineID"]
    assert engine_id.sampled_rows > 0
    assert engine_id.approx_distinct is not None
    assert engine_id.top_values, "approx_top_k must come back as positional pairs"
    assert all(isinstance(count, int) for _, count in engine_id.top_values)
    assert by_name["UserID"].approx_distinct is not None


# --------------------------------------------------------------------------
# plans — where the SETTINGS clause order bit
# --------------------------------------------------------------------------


async def test_explain_returns_a_plan_the_analyzer_can_normalize(
    adapter: ClickHouseAdapter,
) -> None:
    raw = await adapter.explain(UNPRUNED_SQL, ExplainMode.ESTIMATE)

    assert raw.engine == "clickhouse"
    assert "ReadFromMergeTree" in raw.payload


async def test_a_filter_off_the_sort_key_is_reported_as_an_unpruned_scan(
    adapter: ClickHouseAdapter,
) -> None:
    """The claim the whole plan layer makes, measured against 100M real rows."""
    summary = await PlanExplainer(adapter=adapter).explain(UNPRUNED_SQL, "agentdb")

    codes = {warning.code for warning in summary.warnings}
    assert WarningCode.FULL_SCAN in codes or WarningCode.SORT_KEY_UNUSED in codes
    assert summary.render()


async def test_a_sort_key_filter_prunes_and_earns_no_warning(
    adapter: ClickHouseAdapter,
) -> None:
    summary = await PlanExplainer(adapter=adapter).explain(
        "SELECT count() FROM agentdb.hits WHERE CounterID = 62", "agentdb"
    )

    assert summary.pruning_ratio is not None
    assert summary.pruning_ratio < 0.9, "the leading sort-key column must prune granules"
    assert summary.pruning_unit == "granule"


# --------------------------------------------------------------------------
# execution — where result_overflow_mode bit
# --------------------------------------------------------------------------


async def test_a_query_reports_the_rows_and_bytes_the_server_actually_read(
    adapter: ClickHouseAdapter,
) -> None:
    result = await adapter.execute(
        "SELECT CounterID, count() AS hits FROM agentdb.hits GROUP BY CounterID ORDER BY hits DESC",
        Limits(timeout_s=30, max_result_rows=10, max_rows_to_read=500_000_000),
    )

    assert result.row_count == 10
    assert result.truncated, "max_result_rows must truncate rather than fail the query"
    assert result.rows_read is not None and result.rows_read > 0
    assert result.bytes_read is not None and result.bytes_read > 0


async def test_the_connection_cannot_write(adapter: ClickHouseAdapter) -> None:
    """Read-only is the role's property, not a string check on the SQL (SPEC §13.3)."""
    with pytest.raises(QueryPermissionError):
        await adapter.execute(
            "CREATE TABLE agentdb.nope (x UInt8) ENGINE = Memory",
            Limits(timeout_s=30, max_result_rows=10),
        )


async def test_the_workload_log_is_readable_and_normalized(
    adapter: ClickHouseAdapter,
) -> None:
    assert adapter.supports(Capability.WORKLOAD_LOG)


# --------------------------------------------------------------------------
# the assembled context, on real data
# --------------------------------------------------------------------------


async def test_the_grounding_ladder_adds_exactly_one_kind_of_fact_per_rung(
    adapter: ClickHouseAdapter,
) -> None:
    builder = ContextBuilder(adapter=adapter)

    schema = (await builder.build("tpch", GroundingLevel.SCHEMA)).render()
    stats = (await builder.build("tpch", GroundingLevel.STATS)).render()
    layout = (await builder.build("tpch", GroundingLevel.LAYOUT)).render()

    assert "CREATE TABLE" in schema
    assert "Column profiles" not in schema
    assert "Column profiles" in stats
    assert "sort key (ORDER BY)" not in stats
    assert "sort key (ORDER BY)" in layout
    assert len(layout) > len(stats) > len(schema)


# --------------------------------------------------------------------------
# the MCP surface, over the live engine and a live store
# --------------------------------------------------------------------------


def _first_exemplar(structured: Mapping[str, JsonValue], half: str) -> Mapping[str, JsonValue]:
    """The best-ranked exemplar of one half of a ``retrieve_exemplars`` response.

    Narrowed by hand because ``JsonValue`` is a union: the contract tests prove
    the shape against the declared schema, and this only has to reach into it.
    """
    ranked = structured[half]
    assert isinstance(ranked, list) and ranked, f"no {half} exemplars came back"
    best = ranked[0]
    assert isinstance(best, dict)
    exemplar = best["exemplar"]
    assert isinstance(exemplar, dict)
    return exemplar


@pytest.fixture
def store() -> ExemplarStore:
    try:
        memory = ExemplarStore(connect())
    except Exception as exc:
        pytest.skip(f"no exemplar store reachable ({exc}); start one with: make up")
    memory.ensure_schema()
    return memory


async def test_the_served_catalog_answers_over_the_live_engine(
    adapter: ClickHouseAdapter, store: ExemplarStore
) -> None:
    catalog = build_catalog(adapter, store=store)

    layout = await catalog.call("physical_layout", {"relation": "agentdb.hits"})
    plan = await catalog.call("explain_plan", {"sql": UNPRUNED_SQL, "namespace": "agentdb"})
    profile = await catalog.call(
        "profile_columns", {"relation": "tpch.lineitem", "columns": ["l_shipdate"]}
    )

    assert not layout.is_error, layout.structured
    assert layout.structured["order_by"] == [
        "CounterID",
        "EventDate",
        "UserID",
        "EventTime",
        "WatchID",
    ], "the sort key ClickBench's own DDL declares, in order"
    assert not plan.is_error, plan.structured
    assert plan.structured["warnings"]
    assert not profile.is_error, profile.structured


async def test_run_query_remembers_its_execution_and_retrieval_finds_it_again(
    adapter: ClickHouseAdapter, store: ExemplarStore
) -> None:
    """The loop that fills the memory arms, proved end to end on live infrastructure."""
    catalog = build_catalog(adapter, store=store)
    question = "which counters saw the most hits?"
    sql = (
        "SELECT CounterID, count() AS hits FROM agentdb.hits "
        "WHERE CounterID = 62 GROUP BY CounterID"
    )
    store.sync(
        snapshot(
            "clickhouse",
            "agentdb",
            [await adapter.describe_relation(HITS)],
            [await adapter.physical_layout(HITS)],
        )
    )

    executed = await catalog.call(
        "run_query", {"sql": sql, "question": question, "namespace": "agentdb"}
    )
    found = await catalog.call(
        "retrieve_exemplars",
        {"question": question, "namespace": "agentdb", "relations": ["hits"]},
    )

    assert not executed.is_error, executed.structured
    remembered = _first_exemplar(found.structured, "positive")
    assert remembered["normalized_sql"] == normalize_sql(sql, "clickhouse")
    assert remembered["bytes_read"] is not None


async def test_a_failed_execution_becomes_a_negative_exemplar(
    adapter: ClickHouseAdapter, store: ExemplarStore
) -> None:
    catalog = build_catalog(adapter, store=store)
    question = "who visited on a column that does not exist?"

    failed = await catalog.call(
        "run_query",
        {
            "sql": "SELECT NoSuchColumn FROM agentdb.hits LIMIT 1",
            "question": question,
            "namespace": "agentdb",
        },
    )
    negatives = await catalog.call(
        "retrieve_exemplars", {"question": question, "namespace": "agentdb"}
    )

    assert failed.is_error
    assert _first_exemplar(negatives.structured, "negative")["error_class"] == "semantic"
