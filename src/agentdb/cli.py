"""``agentdb`` on the command line: connect to one engine and serve it over stdio.

This module is where the two halves meet — the engine-neutral tool catalog and a
concrete adapter — so it is the one place allowed to name an engine. Everything
below it reasons through the adapter protocol, which is what keeps a third
engine an afternoon's work.

Connection details come from the environment (``AGENTDB_CLICKHOUSE_*``,
``AGENTDB_DBX_*``), never from arguments: a warehouse token on a command line
ends up in shell history and in every process listing on the box.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence

from agentdb.adapters.base import Adapter
from agentdb.adapters.clickhouse import ClickHouseAdapter
from agentdb.adapters.clickhouse_client import ClickHouseTarget, build_client
from agentdb.adapters.databricks import DatabricksAdapter
from agentdb.adapters.databricks_client import DatabricksTarget
from agentdb.adapters.databricks_client import build_client as build_databricks_client
from agentdb.config import Config
from agentdb.core.memory.postgres import connect
from agentdb.core.memory.store import ExemplarStore
from agentdb.server import build_catalog
from agentdb.server.transports import serve_stdio

ENGINES = ("clickhouse", "databricks")


def parser() -> argparse.ArgumentParser:
    """The argument parser, exposed so ``--help`` is testable without a process."""
    parsed = argparse.ArgumentParser(
        prog="agentdb",
        description="Serve agentdb's MCP tools for one engine over stdio.",
    )
    parsed.add_argument(
        "--engine",
        choices=ENGINES,
        default="clickhouse",
        help="which engine to connect to; credentials come from the environment",
    )
    parsed.add_argument(
        "--memory",
        action="store_true",
        help=(
            "serve the exemplar memory tools, connecting to AGENTDB_MEMORY_DSN. "
            "Off by default: the tools are absent rather than advertised and broken "
            "when there is no store behind them"
        ),
    )
    return parsed


async def build_adapter(engine: str) -> Adapter:
    """Connect to ``engine`` using the ``AGENTDB_*`` variables already in the environment."""
    if engine == "clickhouse":
        client = await build_client(ClickHouseTarget.from_env())
        return ClickHouseAdapter(client=client)
    target = DatabricksTarget.from_env()
    return DatabricksAdapter(
        client=await build_databricks_client(target),
        catalog=target.catalog,
    )


def build_store(config: Config) -> ExemplarStore:
    """Open the exemplar store and make sure its schema is there (SPEC §10.2).

    Applying the DDL on startup rather than in a migration step is deliberate:
    the store is agentdb's own state, it is created idempotently, and a reader
    following the README should not have to run anything between
    ``docker compose up`` and a working server.
    """
    store = ExemplarStore(connect(config.memory_dsn), config=config)
    store.ensure_schema()
    return store


async def serve(engine: str, config: Config | None = None, *, memory: bool = False) -> None:
    """Connect, build the catalog, and serve until the client disconnects."""
    effective = config or Config()
    adapter = await build_adapter(engine)
    store = build_store(effective) if memory else None
    await serve_stdio(build_catalog(adapter, effective, store=store))


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``agentdb`` console script."""
    arguments = parser().parse_args(argv)
    asyncio.run(serve(arguments.engine, memory=arguments.memory))
    return 0
