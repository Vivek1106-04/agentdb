"""Check the Databricks adapter against a real warehouse, and freeze what it saw.

Every Databricks fact in this repository was written from documentation:
``DESCRIBE DETAIL`` column names, the ``EXPLAIN FORMATTED`` layout,
``system.query.history`` columns, the shape of ``current_version()``. SPEC §8.2
marks each of them ``VERIFY:`` for a reason — Unity Catalog system tables gain
and rename columns between releases, and a spec that hardcodes a stale name
produces a confusing first bug rather than a loud one.

This script is that verification. It runs the adapter's own code paths against a
workspace, prints what came back, and writes the raw responses into
``tests/fixtures/databricks/`` so the unit suite can be re-pinned to observed
output instead of documented output.

Usage::

    uv run --extra databricks python scripts/verify_databricks.py

Credentials come from the environment or from a ``.env`` file at the repository
root, which is gitignored. Either prefix works::

    AGENTDB_DBX_HOST=https://dbc-....cloud.databricks.com
    AGENTDB_DBX_WAREHOUSE_ID=...
    AGENTDB_DBX_TOKEN=dapi...

Free Edition is enough: ``samples.tpch`` ships pre-loaded, so nothing needs
uploading and no cluster needs configuring. Nothing here writes: every statement
is a read, and ``ANALYZE`` is never issued.
"""

from __future__ import annotations

import asyncio
import json
import os
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from agentdb.adapters.databricks import DatabricksAdapter, DatabricksClient
from agentdb.adapters.databricks_client import DatabricksTarget, build_client
from agentdb.adapters.models import ExplainMode, RelationRef, SamplePolicy
from agentdb.config import Config
from agentdb.core.explain import PlanExplainer

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "databricks"
ENV_FILE = ROOT / ".env"

CATALOG = os.environ.get("AGENTDB_DBX_CATALOG", "samples")
SCHEMA = os.environ.get("AGENTDB_DBX_SCHEMA", "tpch")
TABLE = os.environ.get("AGENTDB_DBX_TABLE", "lineitem")

PROBE_SQL = f"""
SELECT l_returnflag, count(*) AS orders, sum(l_extendedprice) AS revenue
FROM {CATALOG}.{SCHEMA}.{TABLE}
WHERE l_shipdate >= DATE '1995-01-01'
GROUP BY l_returnflag
ORDER BY revenue DESC
""".strip()
"""One query that exercises everything the plan layer reads: a filter that can
push down, a grouping aggregate, and a scan whose file counts are worth seeing."""


@dataclass
class Outcome:
    """What one check produced, so a failure does not stop the rest."""

    name: str
    ok: bool
    detail: str


def load_env(path: Path = ENV_FILE) -> None:
    """Read ``KEY=value`` lines into the environment, without overriding it.

    A tiny parser rather than a dependency: this script must run before anyone
    has decided whether the project takes a dotenv library.
    """
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def mirror_prefixes() -> None:
    """Accept the harness's ``AGENTEVAL_DBX_*`` names for the server's variables.

    One workspace, two consumers; asking an operator to write the same token
    twice is how one of the two ends up stale.
    """
    for name in (
        "HOST",
        "WAREHOUSE_ID",
        "TOKEN",
        "CLIENT_ID",
        "CLIENT_SECRET",
        "CATALOG",
        "SCHEMA",
    ):
        harness = os.environ.get(f"AGENTEVAL_DBX_{name}")
        if harness and not os.environ.get(f"AGENTDB_DBX_{name}"):
            os.environ[f"AGENTDB_DBX_{name}"] = harness


def write_fixture(name: str, content: str) -> Path:
    """Freeze one raw response so a unit test can be pinned to observed output."""
    FIXTURES.mkdir(parents=True, exist_ok=True)
    path = FIXTURES / name
    path.write_text(content, encoding="utf-8")
    return path


def render_rows(columns: Sequence[str], rows: Sequence[Sequence[Any]], limit: int = 5) -> str:
    lines = [f"columns: {list(columns)}"]
    for row in rows[:limit]:
        lines.append(f"  {list(row)}")
    if len(rows) > limit:
        lines.append(f"  ... {len(rows) - limit} more row(s)")
    return "\n".join(lines)


async def raw(
    client: DatabricksClient, sql: str, parameters: Mapping[str, Any] | None = None
) -> tuple[tuple[str, ...], tuple[tuple[Any, ...], ...]]:
    result = await client.statement(sql, parameters=parameters or {})
    return tuple(result.columns), tuple(tuple(row) for row in result.rows)


async def check_version(client: DatabricksClient) -> Outcome:
    """``current_version()`` — the field name is documented, not observed."""
    try:
        columns, rows = await raw(client, "SELECT current_version() AS raw_version")
    except Exception as exc:
        return Outcome("current_version", False, f"{type(exc).__name__}: {exc}")
    write_fixture(
        "current_version.json", json.dumps({"columns": columns, "rows": rows}, default=str)
    )
    return Outcome("current_version", True, render_rows(columns, rows))


async def check_describe_detail(client: DatabricksClient) -> Outcome:
    """The column names this adapter reads by name: format, clusteringColumns, numFiles…"""
    sql = f"DESCRIBE DETAIL `{CATALOG}`.`{SCHEMA}`.`{TABLE}`"
    try:
        columns, rows = await raw(client, sql)
    except Exception as exc:
        return Outcome("DESCRIBE DETAIL", False, f"{type(exc).__name__}: {exc}")

    write_fixture(
        "describe_detail.json", json.dumps({"columns": columns, "rows": rows}, default=str)
    )
    expected = {"format", "partitionColumns", "clusteringColumns", "numFiles", "sizeInBytes"}
    missing = sorted(expected - set(columns))
    detail = render_rows(columns, rows, limit=1)
    if missing:
        detail += f"\n  MISSING the adapter reads: {missing}"
    return Outcome("DESCRIBE DETAIL", not missing, detail)


async def check_tblproperties(client: DatabricksClient) -> Outcome:
    sql = f"SHOW TBLPROPERTIES `{CATALOG}`.`{SCHEMA}`.`{TABLE}`"
    try:
        columns, rows = await raw(client, sql)
    except Exception as exc:
        return Outcome("SHOW TBLPROPERTIES", False, f"{type(exc).__name__}: {exc}")
    write_fixture("tblproperties.json", json.dumps({"columns": columns, "rows": rows}, default=str))
    return Outcome("SHOW TBLPROPERTIES", True, render_rows(columns, rows, limit=20))


async def check_history(client: DatabricksClient) -> Outcome:
    """Z-ORDER is mined from here; on a read-only sample catalog this may be denied."""
    sql = f"DESCRIBE HISTORY `{CATALOG}`.`{SCHEMA}`.`{TABLE}` LIMIT 3"
    try:
        columns, rows = await raw(client, sql)
    except Exception as exc:
        return Outcome("DESCRIBE HISTORY", False, f"{type(exc).__name__}: {exc}")
    write_fixture(
        "describe_history.json", json.dumps({"columns": columns, "rows": rows}, default=str)
    )
    return Outcome("DESCRIBE HISTORY", True, render_rows(columns, rows, limit=3))


async def check_information_schema(client: DatabricksClient) -> Outcome:
    """Ordinals come from here, and ``STATS_NOT_COLLECTED`` is uncomputable without them."""
    sql = (
        "SELECT column_name, ordinal_position, full_data_type, is_nullable, comment\n"
        "FROM system.information_schema.columns\n"
        "WHERE table_catalog = :catalog AND table_schema = :schema AND table_name = :table\n"
        "ORDER BY ordinal_position"
    )
    try:
        columns, rows = await raw(
            client, sql, {"catalog": CATALOG, "schema": SCHEMA, "table": TABLE}
        )
    except Exception as exc:
        return Outcome("information_schema.columns", False, f"{type(exc).__name__}: {exc}")
    write_fixture(
        "information_schema_columns.json",
        json.dumps({"columns": columns, "rows": rows}, default=str),
    )
    return Outcome("information_schema.columns", bool(rows), render_rows(columns, rows, limit=6))


async def check_query_history(client: DatabricksClient) -> Outcome:
    """``system.query.history`` needs a grant; a denial here is a finding, not a crash."""
    sql = "SELECT * FROM system.query.history LIMIT 1"
    try:
        columns, _ = await raw(client, sql)
    except Exception as exc:
        return Outcome(
            "system.query.history",
            False,
            f"{type(exc).__name__}: {exc}\n  (workload mining needs SELECT on system.query)",
        )
    write_fixture("query_history_columns.json", json.dumps({"columns": columns}, default=str))
    expected = {"statement_id", "statement_text", "execution_status", "total_duration_ms"}
    missing = sorted(expected - set(columns))
    detail = f"columns: {list(columns)}"
    if missing:
        detail += f"\n  MISSING the adapter reads: {missing}"
    return Outcome("system.query.history", not missing, detail)


async def check_history_lag(client: DatabricksClient) -> Outcome:
    """How far ``system.query.history`` trails the warehouse clock.

    Measured, not assumed, because the answer decides an architectural choice. If
    the system table were near-real-time it would be the obvious place to read
    measured pruning from — it is SQL, it needs no second API, and SPEC §8.2
    names it. It ran **1,514 to 23,290 seconds** behind across two measurements,
    so the adapter reads the Query History API instead, and this check exists to
    notice if that ever stops being true.
    """
    sql = (
        "SELECT timestampdiff(SECOND, max(start_time), current_timestamp()) AS lag_seconds, "
        "count(*) AS rows_in_history FROM system.query.history"
    )
    try:
        columns, rows = await raw(client, sql)
    except Exception as exc:
        return Outcome("system.query.history lag", False, f"{type(exc).__name__}: {exc}")
    if not rows:
        return Outcome("system.query.history lag", False, "history is empty")
    entry = dict(zip(columns, rows[0], strict=False))
    lag = int(entry.get("lag_seconds") or 0)
    write_fixture("query_history_lag.json", json.dumps(entry, default=str))
    return Outcome(
        "system.query.history lag",
        True,
        f"{lag}s behind the warehouse clock ({entry.get('rows_in_history')} rows)\n"
        + (
            "  usable for inline attribution"
            if lag < 60
            else "  too far behind for inline attribution; the Query History API is used instead"
        ),
    )


async def check_measured_metrics(adapter: DatabricksAdapter, client: DatabricksClient) -> Outcome:
    """The only source of file-pruning evidence Databricks has.

    Runs a real aggregate, then asks the warehouse what it measured.

    The nonce goes into a **predicate**, not a comment. A comment does not defeat
    the Databricks result cache — this check first carried its nonce as a trailing
    ``-- {hex}`` and the second run came back ``result_from_cache: true`` with
    every counter at zero, which is exactly the shape of a measurement that means
    nothing. Comments are evidently normalized out of the cache key; a predicate
    that changes the statement's meaning is not.
    """
    nonce = int(uuid4().hex[:4], 16) % 97
    sql = PROBE_SQL.replace(
        "WHERE l_shipdate >= DATE '1995-01-01'",
        f"WHERE l_shipdate >= DATE '1995-01-01' AND l_orderkey >= {nonce}",
    )
    try:
        result = await client.statement(sql, parameters={})
        measured = await adapter.query_metrics(result.statement_id or "")
    except Exception as exc:
        return Outcome("adapter.query_metrics", False, _failure(exc))

    if measured is None:
        return Outcome(
            "adapter.query_metrics", False, "the query history reported nothing for the statement"
        )
    write_fixture(
        "query_metrics.json",
        json.dumps(
            {
                "statement_id": measured.statement_id,
                "files_read": measured.files_read,
                "files_pruned": measured.files_pruned,
                "rows_read": measured.rows_read,
                "bytes_read": measured.bytes_read,
                "bytes_in_files_read": measured.bytes_in_files_read,
                "from_result_cache": measured.from_result_cache,
                "photon_time_ms": measured.photon_time_ms,
            },
            default=str,
            indent=1,
        ),
    )
    return Outcome("adapter.query_metrics", measured.measured, measured.render())


async def check_explain(client: DatabricksClient) -> Outcome:
    """The plan text the parser is built against. This is the highest-risk fixture."""
    try:
        _, rows = await raw(client, f"EXPLAIN FORMATTED {PROBE_SQL}")
    except Exception as exc:
        return Outcome("EXPLAIN FORMATTED", False, f"{type(exc).__name__}: {exc}")
    payload = "\n".join(str(row[0]) for row in rows)
    path = write_fixture("explain_formatted.txt", payload)
    return Outcome(
        "EXPLAIN FORMATTED",
        "Physical Plan" in payload,
        f"{len(payload.splitlines())} lines -> {path.relative_to(ROOT)}\n"
        + "\n".join(f"  {line}" for line in payload.splitlines()[:20]),
    )


async def check_adapter(adapter: DatabricksAdapter) -> list[Outcome]:
    """The adapter's own methods, which is what the benchmark actually calls."""
    ref = RelationRef(catalog=CATALOG, namespace=SCHEMA, name=TABLE)
    outcomes: list[Outcome] = []

    try:
        relations = await adapter.list_relations(SCHEMA)
        outcomes.append(
            Outcome(
                "adapter.list_relations",
                bool(relations),
                ", ".join(str(relation.ref) for relation in relations[:8]),
            )
        )
    except Exception as exc:
        outcomes.append(Outcome("adapter.list_relations", False, _failure(exc)))

    try:
        detail = await adapter.describe_relation(ref)
        outcomes.append(
            Outcome(
                "adapter.describe_relation",
                bool(detail.columns),
                f"{len(detail.columns)} columns; first: {detail.column_names[:4]}",
            )
        )
    except Exception as exc:
        outcomes.append(Outcome("adapter.describe_relation", False, _failure(exc)))

    try:
        layout = await adapter.physical_layout(ref)
        outcomes.append(
            Outcome(
                "adapter.physical_layout",
                layout.table_format is not None,
                "\n".join(
                    f"  {key}: {value}"
                    for key, value in (
                        ("format", layout.table_format),
                        ("clustering", layout.clustering_columns),
                        ("zorder", layout.zorder_columns),
                        ("partition", layout.partition_by),
                        ("num_files", layout.num_files),
                        ("avg_file_bytes", layout.avg_file_bytes),
                        ("stats_indexed_columns", layout.stats_indexed_columns),
                        ("stats_columns", layout.stats_columns),
                        ("approx_rows", layout.approx_rows),
                        ("deletion_vectors", layout.deletion_vectors_enabled),
                    )
                ),
            )
        )
    except Exception as exc:
        outcomes.append(Outcome("adapter.physical_layout", False, _failure(exc)))

    try:
        rules = await adapter.dialect_rules()
        outcomes.append(Outcome("adapter.dialect_rules", True, f"version: {rules.version}"))
    except Exception as exc:
        outcomes.append(Outcome("adapter.dialect_rules", False, _failure(exc)))

    try:
        profiles = await adapter.column_profile(
            ref, ["l_returnflag"], SamplePolicy(fraction=0.01, max_rows=1_000_000, timeout_s=30)
        )
        profile = profiles[0]
        outcomes.append(
            Outcome(
                "adapter.column_profile",
                profile.sampled_rows > 0,
                f"distinct={profile.approx_distinct} nulls={profile.null_ratio} "
                f"rows={profile.sampled_rows} top={profile.top_values[:3]}",
            )
        )
    except Exception as exc:
        outcomes.append(Outcome("adapter.column_profile", False, _failure(exc)))

    try:
        plan = await adapter.explain(PROBE_SQL, ExplainMode.ESTIMATE)
        outcomes.append(
            Outcome("adapter.explain", bool(plan.payload), f"{len(plan.payload)} chars of plan")
        )
    except Exception as exc:
        outcomes.append(Outcome("adapter.explain", False, _failure(exc)))

    return outcomes


async def check_explainer(adapter: DatabricksAdapter) -> Outcome:
    """The whole A3 path: plan, normalize, evaluate rules, render for an agent."""
    try:
        summary = await PlanExplainer(adapter=adapter, config=Config()).explain(PROBE_SQL, SCHEMA)
    except Exception as exc:
        return Outcome("PlanExplainer.explain", False, _failure(exc))

    write_fixture("plan_summary.txt", summary.render())
    return Outcome(
        "PlanExplainer.explain",
        True,
        f"pruning={summary.pruning_ratio} unit={summary.pruning_unit} "
        f"photon={summary.photon_coverage}\n"
        + "\n".join(f"  {warning.code.value}" for warning in summary.warnings),
    )


def _failure(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}\n" + "".join(
        f"  {line}" for line in traceback.format_exc(limit=2).splitlines(keepends=True)[-3:]
    )


def report(outcomes: Sequence[Outcome]) -> int:
    print("\n" + "=" * 72)
    print("Databricks verification")
    print("=" * 72)
    for outcome in outcomes:
        mark = "PASS" if outcome.ok else "FAIL"
        print(f"\n[{mark}] {outcome.name}")
        for line in outcome.detail.splitlines():
            print(f"    {line}")
    failed = [outcome.name for outcome in outcomes if not outcome.ok]
    print("\n" + "-" * 72)
    if failed:
        print(f"{len(failed)} check(s) failed: {', '.join(failed)}")
        print("Fixtures for the checks that did pass are in tests/fixtures/databricks/.")
        return 1
    print(f"all {len(outcomes)} checks passed; fixtures written to tests/fixtures/databricks/")
    return 0


async def main() -> int:
    load_env()
    mirror_prefixes()
    target = DatabricksTarget.from_env()
    print(f"workspace: {target.host}")
    print(f"warehouse: {target.warehouse_id}")
    print(f"table:     {CATALOG}.{SCHEMA}.{TABLE}")

    client = await build_client(target)
    adapter = DatabricksAdapter(client=client, catalog=CATALOG, context_id="verify")

    outcomes = [
        await check_version(client),
        await check_describe_detail(client),
        await check_tblproperties(client),
        await check_history(client),
        await check_information_schema(client),
        await check_query_history(client),
        await check_history_lag(client),
        await check_explain(client),
        *await check_adapter(adapter),
        await check_explainer(adapter),
        await check_measured_metrics(adapter, client),
    ]
    return report(outcomes)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
