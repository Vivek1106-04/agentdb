"""Fully scripted catalogs for the MCP server tests.

Built on :mod:`tests.fakes`, so the server is proved against the same adapter
protocol the rest of core is proved against and no engine has to be running.
The plan and result payloads here are the shapes the real parsers consume, not
convenient stand-ins: a contract test that validated a hand-made response would
prove the schema matched the test rather than the server.
"""

from __future__ import annotations

import json
from typing import Any

from agentdb.adapters import ExplainMode, RawPlan, ResultSet, WorkloadEntry
from agentdb.config import Config
from agentdb.server import ToolCatalog, build_catalog
from tests.fakes import FakeAdapter, clickhouse_hits_fixture, databricks_tpch_fixture

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


def clickhouse_catalog(
    plan: dict[str, Any] | None = None, config: Config | None = None
) -> tuple[ToolCatalog, FakeAdapter]:
    """A catalog over the ClickBench-shaped fake, with everything scripted."""
    adapter = clickhouse_hits_fixture()
    _script(adapter, plan or CLICKHOUSE_PLAN)
    return build_catalog(adapter, config), adapter


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
    return build_catalog(adapter), adapter


def _script(adapter: FakeAdapter, plan: dict[str, Any]) -> None:
    adapter.plan = RawPlan(
        engine="clickhouse",
        mode=ExplainMode.ESTIMATE,
        sql="",
        payload=json.dumps([plan]),
    )
    adapter.result = RESULT
    adapter.workload_entries = WORKLOAD
