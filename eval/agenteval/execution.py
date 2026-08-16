"""How a system under test reaches a database.

The harness has to run SQL to score it, but it must not care *whose* database
client is doing the running — a third-party MCP server executes queries through
its own connection, and agenteval only ever sees what came back. This protocol
is deliberately the smallest surface that supports the A-family arms: show the
agent a schema, run what it emits, report what happened.

Declared here rather than imported from ``agentdb.adapters``: agenteval depends
on no part of agentdb (SPEC §4.1.6), and that isolation is CI-enforced.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agenteval.systems.base import EmittedQuery
from agenteval.tasks import Engine


@runtime_checkable
class QueryExecutor(Protocol):
    """Read-only access to one engine, for one benchmark run.

    Implementations must be read-only *at the connection or role level*, never
    by inspecting the SQL string — a benchmark that lets a model's output decide
    whether it is safe has no safety property at all (SPEC §4.1.3).
    """

    @property
    def engine(self) -> Engine:
        """Which engine this executor talks to. Read-only: an executor that
        changed engines mid-run would silently mix two sets of numbers."""
        ...

    async def schema_text(self, namespace: str) -> str:
        """The ``CREATE TABLE`` DDL for ``namespace``, as the A0 arm shows it."""
        ...

    async def run(self, sql: str) -> EmittedQuery:
        """Execute ``sql`` and report the outcome.

        Never raises for a *query* failure: an engine rejecting a query is a
        measurement, not an exception. Connection-level faults may still raise.
        """
        ...

    async def aclose(self) -> None:
        """Release whatever the executor holds open.

        Every run used to leak its connection pool — MCP sessions were closed and
        engines were not — which printed an unclosed-session error over the end of
        each run. Implementations that hold nothing may do nothing, but they must
        say so rather than leave the caller guessing.
        """
        ...
