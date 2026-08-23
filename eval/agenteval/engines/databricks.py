"""A live Databricks :class:`~agenteval.execution.QueryExecutor`.

The Databricks counterpart of the ClickHouse executor, and deliberately its
twin: same interface, same grading, same trace shape, so a cross-engine number
is a comparison rather than two unrelated experiments.

Two things are load-bearing for credibility, and both differ in mechanism from
ClickHouse while matching it in intent.

**Read-only is not this module's job.** The harness authenticates as a principal
holding ``SELECT`` and nothing more (SPEC §13.3). No SQL string is inspected;
string filtering is not a security boundary.

**Every statement is attributable.** Databricks has no ``log_comment``, so
attribution rides on two mechanisms: the API's own ``statement_id``, recorded on
the result, and a ``/* agentdb:{context}:{turn} */`` prefix that
``system.query.history`` keeps verbatim. Either one joins a reported figure back
to the warehouse's own record.

This module imports nothing from ``agentdb``. The harness must be able to score a
system this project did not write, including on an engine adapter it does not
share (SPEC §4.1.6).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Protocol
from uuid import uuid4

from agenteval.engines.clickhouse import SchemaError
from agenteval.engines.errors import databricks_error_class
from agenteval.systems.base import EmittedQuery
from agenteval.tasks import Engine

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

TAG_PREFIX = "agentdb"
"""``WHERE statement_text ILIKE '%/* agentdb:%'`` gives the same audit trail
ClickHouse gets from ``log_comment`` (SPEC §8.2)."""

LIST_TABLES = """
SELECT table_name
FROM system.information_schema.tables
WHERE table_catalog = :catalog AND table_schema = :schema
ORDER BY table_name
""".strip()


class StatementResult(Protocol):
    """The slice of a statement response this module reads."""

    @property
    def columns(self) -> Sequence[str]: ...

    @property
    def rows(self) -> Sequence[Sequence[Any]]: ...

    @property
    def statement_id(self) -> str | None: ...

    @property
    def rows_read(self) -> int | None: ...

    @property
    def bytes_read(self) -> int | None: ...


class DatabricksClient(Protocol):
    """The client methods the harness uses, so the transport stays swappable."""

    async def statement(
        self,
        sql: str,
        *,
        parameters: Mapping[str, Any],
        row_limit: int | None = None,
        timeout_s: int | None = None,
        catalog: str | None = None,
        schema: str | None = None,
    ) -> StatementResult: ...

    async def query_info(self, statement_id: str) -> Mapping[str, Any] | None:
        """The warehouse's own record of one execution, or ``None``."""


@dataclass(frozen=True, slots=True)
class DatabricksLimits:
    """Per-statement ceilings, matched to the ClickHouse arm so arms stay comparable."""

    timeout_s: int = 30
    max_result_rows: int = 10_000


@dataclass(frozen=True, slots=True)
class DatabricksExecutor:
    """Runs benchmark SQL against one Databricks SQL warehouse."""

    client: DatabricksClient
    catalog: str = "samples"
    context_id: str = "agenteval"
    limits: DatabricksLimits = field(default_factory=DatabricksLimits)
    turn_id: Callable[[], str] = lambda: uuid4().hex[:12]
    engine: Engine = "databricks"
    collect_pruning: bool = False
    """Whether to ask the query-history API what each statement pruned.

    Costs one extra API call per emitted query and answers nothing about
    accuracy, so it is opt-in — but it is the only place Databricks pruning
    evidence exists, so the physical-design arms want it."""

    async def schema_text(self, namespace: str) -> str:
        """The ``CREATE TABLE`` statements for ``namespace``, in table-name order.

        Every statement is fully qualified. A schema dump that showed two-part
        names would be teaching the model the habit that
        ``UNQUALIFIED_RELATION`` exists to catch.
        """
        catalog, schema = self._split(namespace)
        listing = await self._query(LIST_TABLES, {"catalog": catalog, "schema": schema})
        names = sorted(str(row[0]) for row in listing.rows)
        if not names:
            raise SchemaError(f"schema {catalog}.{schema} has no tables")

        statements = []
        for name in names:
            reference = ".".join(_quote(part) for part in (catalog, schema, name))
            result = await self._query(f"SHOW CREATE TABLE {reference}", {})
            statements.append(str(result.rows[0][0]) if result.rows else "")
        return "\n\n".join(statement for statement in statements if statement)

    async def run(self, sql: str, namespace: str) -> EmittedQuery:
        """Execute ``sql`` against ``namespace``, reporting rejection rather than raising.

        The catalog and schema come from the task, not from however the client
        was constructed, so a two-part name in a model's answer resolves against
        the schema the task names — the one the grounding described.
        """
        started = perf_counter()
        try:
            result = await self._query(sql, {}, namespace=namespace)
        except Exception as exc:
            error_class = databricks_error_class(str(exc))
            if error_class is None:
                raise
            return EmittedQuery(
                sql=sql,
                succeeded=False,
                error_class=error_class,
                error_text=str(exc),
                duration_ms=_elapsed_ms(started),
            )

        rows = tuple(tuple(row) for row in result.rows)
        files_read, files_pruned = await self._pruning(result.statement_id)
        return EmittedQuery(
            sql=sql,
            succeeded=True,
            columns=tuple(result.columns),
            rows=rows,
            row_count=len(rows),
            duration_ms=_elapsed_ms(started),
            rows_read=result.rows_read,
            bytes_read=result.bytes_read,
            statement_id=result.statement_id,
            files_read=files_read,
            files_pruned=files_pruned,
        )

    async def aclose(self) -> None:
        """Nothing to release.

        The Statement Execution API is called per statement through the SDK,
        which owns its own HTTP session and outlives no run of ours. Stated
        rather than omitted, so a reader does not have to check whether this was
        forgotten.
        """
        return

    async def _pruning(self, statement_id: str | None) -> tuple[int | None, int | None]:
        """What the warehouse measured itself pruning, if it will say.

        Off by default. Every lookup is a second API call per emitted query, and
        a benchmark measuring accuracy has no use for it; the arms that reason
        about physical design do, and they turn it on.

        Zero files read is *not* recorded as pruning. Three unrelated situations
        produce it — the result cache answered, Delta metadata answered, or the
        warehouse reported nothing — and a ratio built from any of them says the
        query pruned everything perfectly (SPEC §8.2).
        """
        if not self.collect_pruning or not statement_id:
            return None, None
        entry = await self.client.query_info(statement_id)
        metrics = entry.get("metrics") if entry else None
        if not isinstance(metrics, Mapping) or metrics.get("result_from_cache") is True:
            return None, None
        read = _count(metrics.get("read_files_count"))
        pruned = _count(metrics.get("pruned_files_count"))
        if not read and not pruned:
            return None, None
        return read, pruned

    def _split(self, namespace: str) -> tuple[str, str]:
        """``schema`` or ``catalog.schema`` into both parts."""
        catalog, separator, schema = namespace.partition(".")
        if not separator:
            return self.catalog, namespace
        return catalog, schema

    async def _query(
        self, sql: str, parameters: Mapping[str, Any], namespace: str | None = None
    ) -> StatementResult:
        tagged = f"/* {TAG_PREFIX}:{self.context_id}:{self.turn_id()} */\n{sql}"
        catalog, schema = self._split(namespace) if namespace is not None else (None, None)
        return await self.client.statement(
            tagged,
            parameters=parameters,
            row_limit=self.limits.max_result_rows,
            timeout_s=self.limits.timeout_s,
            catalog=catalog,
            schema=schema,
        )


def _quote(name: str) -> str:
    """Backtick a name part, refusing anything that is not an identifier.

    Names come from task files and from ``information_schema``, never from a
    model, so this is a corruption check — but a benchmark that let a malformed
    namespace reach the warehouse would be reporting on a table nobody named.
    """
    if not _IDENTIFIER.match(name):
        raise SchemaError(f"{name!r} is not a valid Databricks identifier")
    return f"`{name}`"


def _count(value: object) -> int | None:
    """A metric counter, or ``None`` when it is absent or unreadable."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return int(value)
    text = str(value).strip()
    return int(text) if text.isdigit() else None


def _elapsed_ms(started: float) -> int:
    return round((perf_counter() - started) * 1000)
