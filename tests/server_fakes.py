"""Fully scripted catalogs for the MCP server tests.

Built on :mod:`tests.fakes`, so the server is proved against the same adapter
protocol the rest of core is proved against and no engine has to be running.
The plan and result payloads here are the shapes the real parsers consume, not
convenient stand-ins: a contract test that validated a hand-made response would
prove the schema matched the test rather than the server.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from agentdb.adapters import (
    ColumnDef,
    ExplainMode,
    RawPlan,
    RelationDetail,
    RelationRef,
    ResultSet,
    WorkloadEntry,
)
from agentdb.config import Config
from agentdb.core.memory import snapshot, snapshot_to_json
from agentdb.core.memory.store import ExemplarStore
from agentdb.server import ToolCatalog, build_catalog
from tests.fakes import FakeAdapter, clickhouse_hits_fixture, databricks_tpch_fixture
from tests.memory_fakes import FakeConnection, exemplar_row, version_row

CLICKHOUSE_PLAN: dict[str, Any] = {
    "Plan": {
        "Node Type": "ReadFromMergeTree",
        "Description": "agentdb.hits",
        "Indexes": [
            {"Type": "PrimaryKey", "Initial Granules": 1_000, "Selected Granules": 1_000},
        ],
    }
}
"""A scan that pruned nothing — the case the whole plan layer exists to name."""

PRUNED_CLICKHOUSE_PLAN: dict[str, Any] = {
    "Plan": {
        "Node Type": "ReadFromMergeTree",
        "Description": "agentdb.hits",
        "Indexes": [
            {"Type": "PrimaryKey", "Initial Granules": 1_000, "Selected Granules": 10},
        ],
    }
}

DATABRICKS_PLAN = """== Physical Plan ==
PhotonResultStage (3)
+- PhotonScan parquet samples.tpch.lineitem (1)

(1) PhotonScan parquet samples.tpch.lineitem
Output [1]: [l_shipdate#2]
Location: PreparedDeltaFileIndex [s3://bucket/lineitem]
PushedFilters: [IsNotNull(l_shipdate), GreaterThan(l_shipdate,1998-01-01)]
number of files read: 40
size of files read: 1.2 GiB
Statistics: 1.2 GiB, 6001215 rows

(3) PhotonResultStage
Arguments: 1
"""
"""One Photon scan that pruned 40 of 1,000 files — the Delta half of the IR."""

RESULT = ResultSet(
    columns=("CounterID", "hits"),
    rows=((62, 1_000), (63, 900)),
    row_count=2,
    truncated=False,
    duration_ms=41,
    rows_read=1_000_000,
    bytes_read=8_000_000,
    query_id="agentdb:test:1",
)

WORKLOAD = (
    WorkloadEntry(
        normalized_sql="SELECT count() FROM hits WHERE CounterID = ?",
        calls=412,
        relations=("agentdb.hits",),
        total_duration_ms=91_000.0,
        mean_duration_ms=220.9,
        rows_read=99_000_000,
        bytes_read=1_400_000_000,
        sample_sql="SELECT count() FROM hits WHERE CounterID = 62",
    ),
)


MEMORY_NOW = datetime(2026, 8, 22, 12, 0, tzinfo=UTC)
MEMORY_EARLIER = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)


def memory_connection(engine: str = "clickhouse", namespace: str = "agentdb") -> FakeConnection:
    """The scripted Postgres behind :func:`memory_store`.

    Exposed separately from the store so a test can assert what was written —
    which relations and columns a recorded exemplar carries is the whole basis
    of re-validation, and it is derived rather than supplied.
    """
    state = snapshot(
        engine,  # type: ignore[arg-type]  # Literal, from the caller
        namespace,
        [
            RelationDetail(
                ref=RelationRef(namespace=namespace, name="hits"),
                columns=(ColumnDef(name="CounterID", data_type="UInt32", is_nullable=False),),
                create_statement=f"CREATE TABLE {namespace}.hits (...)",
            )
        ],
    )
    layout = json.dumps(snapshot_to_json(state))
    version = version_row(
        id=1, engine=engine, namespace=namespace, layout_json=layout, observed_at=MEMORY_EARLIER
    )
    remembered = exemplar_row(
        id=7,
        engine=engine,
        namespace=namespace,
        embedding="[]",
        bytes_read=4_096,
        valid_from=MEMORY_EARLIER,
        tx_from=MEMORY_EARLIER,
    )
    return FakeConnection(
        {
            "ORDER BY observed_at DESC LIMIT 1": [[version]],
            "ORDER BY observed_at, id": [[version]],
            "ORDER BY (relations &&": [[remembered]],
            "INSERT INTO agentdb_exemplar": [[remembered]],
            "ORDER BY tx_from, id": [[remembered]],
        }
    )


def memory_store(connection: FakeConnection | None = None) -> ExemplarStore:
    """An exemplar store whose Postgres is scripted rather than running.

    The memory tools are contract-tested like every other tool, so the catalog
    the tests build has to have a store behind it. What it does not have to have
    is a database: the store speaks a structural connection protocol precisely so
    this is possible (SPEC §10.2).
    """
    return ExemplarStore(connection or memory_connection(), clock=lambda: MEMORY_NOW)


def scripted_clickhouse_adapter(plan: dict[str, Any] | None = None) -> FakeAdapter:
    """The ClickBench-shaped fake with plan, result and workload scripted.

    Exposed so a test can build its own catalog around it — one with a shadow
    runner, say — without reaching for the private scripting helper.
    """
    adapter = clickhouse_hits_fixture()
    _script(adapter, plan or CLICKHOUSE_PLAN)
    return adapter


def clickhouse_catalog(
    plan: dict[str, Any] | None = None, config: Config | None = None
) -> tuple[ToolCatalog, FakeAdapter]:
    """A catalog over the ClickBench-shaped fake, with everything scripted."""
    adapter = scripted_clickhouse_adapter(plan)
    return build_catalog(adapter, config, store=memory_store()), adapter


def databricks_catalog() -> tuple[ToolCatalog, FakeAdapter]:
    """A catalog over the TPC-H-shaped Databricks fake."""
    adapter = databricks_tpch_fixture()
    adapter.result = RESULT
    adapter.workload_entries = WORKLOAD
    adapter.plan = RawPlan(
        engine="databricks",
        mode=ExplainMode.ESTIMATE,
        sql="",
        payload=DATABRICKS_PLAN,
    )
    store = memory_store(memory_connection("databricks", "samples.tpch"))
    return build_catalog(adapter, store=store), adapter


def _script(adapter: FakeAdapter, plan: dict[str, Any]) -> None:
    adapter.plan = RawPlan(
        engine="clickhouse",
        mode=ExplainMode.ESTIMATE,
        sql="",
        payload=json.dumps([plan]),
    )
    adapter.result = RESULT
    adapter.workload_entries = WORKLOAD
