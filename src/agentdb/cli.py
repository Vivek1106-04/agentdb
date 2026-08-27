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
import sys
from collections.abc import Callable, Sequence

from agentdb.adapters.base import Adapter
from agentdb.adapters.clickhouse import ClickHouseAdapter
from agentdb.adapters.clickhouse_client import ClickHouseTarget, build_client
from agentdb.adapters.databricks import DatabricksAdapter
from agentdb.adapters.databricks_client import DatabricksTarget
from agentdb.adapters.databricks_client import build_client as build_databricks_client
from agentdb.config import Config
from agentdb.core.memory.postgres import connect
from agentdb.core.memory.store import ExemplarStore
from agentdb.demo import CASES, run_demo
from agentdb.server import build_catalog
from agentdb.server.transports import serve_stdio

ENGINES = ("clickhouse", "databricks")

Writer = Callable[[str], None]
"""Where the demo's output goes. Injected so a test reads it without a process."""


COMMANDS = ("serve", "demo")
DEFAULT_COMMAND = "serve"


def parser() -> argparse.ArgumentParser:
    """The argument parser, exposed so ``--help`` is testable without a process."""
    parsed = argparse.ArgumentParser(
        prog="agentdb",
        description="Serve agentdb's MCP tools for one engine over stdio.",
    )
    commands = parsed.add_subparsers(dest="command", required=True)

    serve_command = commands.add_parser("serve", help="serve the MCP tools over stdio")
    serve_command.add_argument(
        "--engine",
        choices=ENGINES,
        default="clickhouse",
        help="which engine to connect to; credentials come from the environment",
    )
    serve_command.add_argument(
        "--memory",
        action="store_true",
        help=(
            "serve the exemplar memory tools, connecting to AGENTDB_MEMORY_DSN. "
            "Off by default: the tools are absent rather than advertised and broken "
            "when there is no store behind them"
        ),
    )

    demo_command = commands.add_parser(
        "demo", help="run the README's before/after against the connected engine"
    )
    demo_command.add_argument("--engine", choices=ENGINES, default="clickhouse")
    return parsed


def _with_default_command(argv: Sequence[str] | None) -> list[str]:
    """Let ``agentdb --engine clickhouse`` keep meaning ``agentdb serve --engine clickhouse``.

    Every MCP client config in the wild spells the server that way, and a
    subcommand added years later is not a reason to break them.
    """
    arguments = list(argv if argv is not None else sys.argv[1:])
    if arguments and (arguments[0] in COMMANDS or arguments[0] in ("-h", "--help")):
        return arguments
    return [DEFAULT_COMMAND, *arguments]


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


async def demo(engine: str, *, write: Writer = print) -> None:
    """Run the README's before/after panel against a live engine."""
    adapter = await build_adapter(engine)
    try:
        write(await run_demo(adapter, CASES[engine]))
    finally:
        client = getattr(adapter, "client", None)
        closer = getattr(client, "close", None)
        if closer is not None:
            await closer()


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``agentdb`` console script."""
    arguments = parser().parse_args(_with_default_command(argv))
    if arguments.command == "demo":
        asyncio.run(demo(arguments.engine))
        return 0
    asyncio.run(serve(arguments.engine, memory=arguments.memory))
    return 0
