"""Shadow validation against a real ClickHouse, including the kill (SPEC §9.1.B).

Two things are proved here that no fake can prove. The first is that the DDL is
right: a sampled copy of a real table, carrying a real candidate index, whose
plan ClickHouse will actually produce. The second is the one the spec singles
out — a process killed mid-validation leaves an orphan, and the reaper finds it
on next start.

The kill is real. A child process creates the shadow table and then sends itself
SIGKILL before its ``finally`` can run, which is exactly what a container
eviction does and exactly what a ``try/finally`` cannot survive.

This tier needs a *writable* connection, which the rest of the suite deliberately
does not have: the read-only role is the boundary that makes the server safe to
hand an agent. It writes only into ``tpch``, only tables carrying the
``__agentdb_shadow`` marker, and drops every one of them.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

from agentdb.adapters import ExplainMode, RelationRef
from agentdb.adapters.base import AdapterError
from agentdb.adapters.clickhouse import ClickHouseAdapter
from agentdb.adapters.clickhouse_client import ClickHouseTarget, build_client
from agentdb.adapters.clickhouse_shadow import ClickHouseShadowRunner
from agentdb.config import Config
from agentdb.core.advisor import MARKER, ShadowValidator, reap_orphans

pytestmark = pytest.mark.e2e

ORDERS = RelationRef(namespace="tpch", name="orders")
"""7.5M rows, of which the shadow copies one percent.

Not ``nation``: at 25 rows ClickHouse answers a count from metadata — the
"optimized trivial count" — and a plan with no scan in it has no pruning to
measure. A validation probe has to be a query that actually reads."""

WRITABLE = ClickHouseTarget(
    host=os.environ.get("AGENTDB_CLICKHOUSE_HOST", "localhost"),
    port=int(os.environ.get("AGENTDB_CLICKHOUSE_PORT", "58123")),
    username="agentdb",
    password="agentdb",
    database="tpch",
)
"""The writable principal, named explicitly. Never the role the tools use."""

ALLOW = Config(allow_shadow=True)

CHILD = """
import asyncio, os, signal
from agentdb.adapters.clickhouse_client import ClickHouseTarget, build_client
from agentdb.adapters.clickhouse_shadow import ClickHouseShadowRunner

async def main():
    client = await build_client(
        ClickHouseTarget(host="{host}", port={port}, username="agentdb",
                         password="agentdb", database="tpch")
    )
    runner = ClickHouseShadowRunner(client=client)
    await runner.run(
        "CREATE TABLE tpch.orders{marker}_{token} ENGINE = MergeTree ORDER BY (o_orderkey) "
        "AS SELECT * FROM tpch.orders LIMIT 1000"
    )
    # No cleanup: this is the eviction a finally block cannot survive.
    os.kill(os.getpid(), signal.SIGKILL)

asyncio.run(main())
"""


@pytest.fixture
async def runner() -> AsyncIterator[ClickHouseShadowRunner]:
    try:
        client = await build_client(WRITABLE)
    except AdapterError as exc:
        pytest.skip(f"no writable ClickHouse ({exc}); start one with: make up")
    live = ClickHouseShadowRunner(client=cast(Any, client))
    try:
        yield live
    finally:
        await reap_orphans(live, ["tpch"])
        await cast(Any, client).close()


async def test_a_candidate_index_is_measured_on_a_sampled_copy(
    runner: ClickHouseShadowRunner,
) -> None:
    """The whole mechanism, on real data: build, arm, plan, drop."""
    read_only = await build_client(ClickHouseTarget.from_env())
    try:
        layout = await ClickHouseAdapter(client=read_only).physical_layout(ORDERS)
    finally:
        await cast(Any, read_only).close()

    validator = ShadowValidator(runner=runner, config=ALLOW)
    measurement = await validator.measure(
        ref=ORDERS,
        layout=layout,
        probe_sql=(
            "SELECT count() FROM tpch.orders WHERE o_orderdate >= '1995-01-01' "
            "AND o_orderdate < '1995-02-01'"
        ),
        baseline=1.0,
        order_by=("o_orderdate", "o_orderkey"),
    )

    assert measurement.after is not None
    assert 0.0 <= measurement.after <= 1.0
    assert measurement.unit == "granule"
    assert "shadow table holding" in measurement.method
    assert MARKER not in await _tables(runner), "the shadow was dropped"


async def test_a_failed_probe_still_leaves_nothing_behind(
    runner: ClickHouseShadowRunner,
) -> None:
    """The candidate was rejected; the sampled copy still goes."""
    read_only = await build_client(ClickHouseTarget.from_env())
    try:
        layout = await ClickHouseAdapter(client=read_only).physical_layout(ORDERS)
    finally:
        await cast(Any, read_only).close()

    validator = ShadowValidator(runner=runner, config=ALLOW)

    with pytest.raises(AdapterError):
        await validator.measure(
            ref=ORDERS,
            layout=layout,
            probe_sql="SELECT no_such_column FROM tpch.orders WHERE o_custkey = 1",
            baseline=None,
        )

    assert MARKER not in await _tables(runner)


async def test_a_process_killed_mid_validation_is_cleaned_up_on_next_start(
    runner: ClickHouseShadowRunner,
) -> None:
    """The chaos test SPEC §9.1.B asks for, with a real SIGKILL.

    A ``finally`` cannot run when the process is killed, which is why the shadow
    name carries a marker: the next start can recognise what it left behind and
    drop it. Without the marker an evicted container would leak a copy of the
    table it was validating, forever, silently.
    """
    token = "chaos001"
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            CHILD.format(host=WRITABLE.host, port=WRITABLE.port, marker=MARKER, token=token),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert child.returncode == -signal.SIGKILL, child.stderr
    orphan = f"orders{MARKER}_{token}"
    assert orphan in await runner.list_tables("tpch"), "the kill really did leak a table"

    dropped = await reap_orphans(runner, ["tpch"])

    assert f"tpch.{orphan}" in dropped
    assert orphan not in await runner.list_tables("tpch")


async def test_the_reaper_leaves_the_tables_it_did_not_create_alone(
    runner: ClickHouseShadowRunner,
) -> None:
    """A reaper that took an unmarked table would be a data-loss bug, not a tidy-up."""
    before = set(await runner.list_tables("tpch"))

    await reap_orphans(runner, ["tpch"])

    assert set(await runner.list_tables("tpch")) == before


async def test_the_write_channel_reads_a_plan_the_analyzer_understands(
    runner: ClickHouseShadowRunner,
) -> None:
    plan = await runner.explain(
        "SELECT count() FROM tpch.orders WHERE o_custkey = 1", ExplainMode.ESTIMATE
    )

    assert plan.engine == "clickhouse"
    assert "ReadFromMergeTree" in plan.payload


async def _tables(runner: ClickHouseShadowRunner) -> str:
    return " ".join(await runner.list_tables("tpch"))
