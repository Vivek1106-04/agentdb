"""A live ClickHouse :class:`~agenteval.execution.QueryExecutor`.

Two things here are load-bearing for the benchmark's credibility.

**Read-only is not this module's job.** The harness connects as a role whose
profile carries ``readonly = 1`` and the per-query ceilings (SPEC §13.3). No SQL
string is inspected, because string filtering is not a security boundary and a
benchmark that pretended otherwise would be publishing a false safety claim
alongside its numbers.

**Every query is attributable.** Each execution sets ``log_comment`` to
``agentdb:{context}:{turn}``, so ``system.query_log`` can be filtered back to a
single graded attempt. That is what lets a reader check a reported bytes-read
figure against the server's own record instead of trusting the harness.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Protocol
from uuid import uuid4

from agenteval.engines.errors import clickhouse_error_class
from agenteval.systems.base import EmittedQuery
from agenteval.tasks import Engine

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

LOG_COMMENT_PREFIX = "agentdb"
"""``WHERE log_comment LIKE 'agentdb:%'`` gives full agent auditability (SPEC §8.4)."""


class SchemaError(RuntimeError):
    """The schema could not be read. Fatal: no arm can run without it."""


class QueryResult(Protocol):
    """The slice of a clickhouse-connect result this module reads."""

    column_names: Sequence[str]
    result_rows: Sequence[Sequence[Any]]
    summary: Mapping[str, str]


class ClickHouseClient(Protocol):
    """The two client methods the harness uses, so the driver stays swappable."""

    async def query(self, query: str, *, settings: Mapping[str, Any]) -> QueryResult: ...

    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ClickHouseLimits:
    """Per-query ceilings. Lowered from the role profile, never raised past it."""

    max_execution_time: int = 30
    max_result_rows: int = 10_000


@dataclass(frozen=True, slots=True)
class ClickHouseExecutor:
    """Runs benchmark SQL against one ClickHouse server."""

    client: ClickHouseClient
    context_id: str = "agenteval"
    limits: ClickHouseLimits = field(default_factory=ClickHouseLimits)
    turn_id: Callable[[], str] = lambda: uuid4().hex[:12]
    engine: Engine = "clickhouse"

    async def schema_text(self, namespace: str) -> str:
        """The ``CREATE TABLE`` statements for ``namespace``, in table-name order."""
        database = _quote_identifier(namespace)
        listing = await self._query(f"SHOW TABLES FROM {database}")
        names = sorted(str(row[0]) for row in listing.result_rows)
        if not names:
            raise SchemaError(f"database {namespace!r} has no tables")

        statements = [
            str(
                (
                    await self._query(f"SHOW CREATE TABLE {database}.{_quote_identifier(n)}")
                ).result_rows[0][0]
            )
            for n in names
        ]
        return "\n\n".join(statements)

    async def run(self, sql: str, namespace: str) -> EmittedQuery:
        """Execute ``sql`` against ``namespace``, reporting rejection rather than raising.

        The database is set per query rather than on the connection, so one run
        can cross ``clickbench_nl`` (``agentdb``) and ``tpch_nl`` (``tpch``)
        without reconnecting — and so an unqualified gold query lands in the
        database its task declares rather than in whichever one the operator
        exported last.
        """
        started = perf_counter()
        try:
            result = await self._query(sql, namespace)
        except Exception as exc:
            error_class = clickhouse_error_class(str(exc))
            if error_class is None:
                raise
            return EmittedQuery(
                sql=sql,
                succeeded=False,
                error_class=error_class,
                error_text=str(exc),
                duration_ms=_elapsed_ms(started),
            )

        rows = tuple(tuple(row) for row in result.result_rows)
        return EmittedQuery(
            sql=sql,
            succeeded=True,
            columns=tuple(result.column_names),
            rows=rows,
            row_count=len(rows),
            duration_ms=_elapsed_ms(started),
            rows_read=_summary_int(result.summary, "read_rows"),
            bytes_read=_summary_int(result.summary, "read_bytes"),
        )

    async def aclose(self) -> None:
        """Close the driver's connection pool."""
        await self.client.close()

    async def _query(self, sql: str, namespace: str | None = None) -> QueryResult:
        return await self.client.query(sql, settings=self._settings(namespace))

    def _settings(self, namespace: str | None = None) -> dict[str, Any]:
        settings: dict[str, Any] = {
            "log_comment": f"{LOG_COMMENT_PREFIX}:{self.context_id}:{self.turn_id()}",
            "max_execution_time": self.limits.max_execution_time,
            "max_result_rows": self.limits.max_result_rows,
        }
        if namespace is not None:
            # A URL parameter of the HTTP interface, not a server setting: it
            # scopes unqualified names in this statement alone.
            settings["database"] = _identifier(namespace)
        return settings


def _identifier(name: str) -> str:
    """Validate a database name bound for the wire, unquoted.

    The ``database`` parameter takes a bare name; backticks would become part of
    it. Validation still runs, for the reason :func:`_quote_identifier` gives.
    """
    if not _IDENTIFIER.match(name):
        raise SchemaError(f"{name!r} is not a valid ClickHouse identifier")
    return name


def _quote_identifier(name: str) -> str:
    """Backtick a database or table name, refusing anything that is not one.

    Names come from task files and from ``SHOW TABLES``, never from a model, so
    this is a corruption check rather than an injection defence — but a
    benchmark that let a malformed namespace reach the server would be reporting
    on a database nobody named.
    """
    if not _IDENTIFIER.match(name):
        raise SchemaError(f"{name!r} is not a valid ClickHouse identifier")
    return f"`{name}`"


def _elapsed_ms(started: float) -> int:
    return round((perf_counter() - started) * 1000)


def _summary_int(summary: Mapping[str, str], key: str) -> int | None:
    """Read one counter from the server's own summary, tolerating its absence."""
    raw = summary.get(key)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None
