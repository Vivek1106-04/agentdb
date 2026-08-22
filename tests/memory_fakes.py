"""A fake DB-API connection for the exemplar store's unit tier.

The store speaks a structural :class:`~agentdb.core.memory.store.Connection`
protocol precisely so its logic — bi-temporal window closing, re-validation,
hybrid ranking hand-off — is testable without Postgres. What this fake does not
prove is that the SQL is valid; that is the e2e tier's job, against the real
pgvector container.

Result sets are keyed by a distinctive substring of the statement rather than by
call order, so a test states which query it is answering and stays readable when
the store issues one more.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from datetime import datetime

from agentdb.core.memory.models import Outcome, Provenance

Row = Sequence[object]
ResultSet = Sequence[Row]


class FakeCursor:
    """One statement's worth of interaction with :class:`FakeConnection`."""

    def __init__(self, connection: FakeConnection) -> None:
        self._connection = connection
        self._rows: ResultSet = ()
        self.closed = False

    def execute(self, query: str, params: Sequence[object] | None = None) -> object:
        self._rows = self._connection.dispatch(query, params)
        return self

    def fetchall(self) -> ResultSet:
        return self._rows

    def fetchone(self) -> Row | None:
        return self._rows[0] if self._rows else None

    def close(self) -> None:
        self.closed = True


class FakeConnection:
    """Records every statement and answers the ones a test scripted."""

    def __init__(self, responses: Mapping[str, Sequence[ResultSet]] | None = None) -> None:
        self._responses = {key: deque(values) for key, values in (responses or {}).items()}
        self.statements: list[tuple[str, Sequence[object] | None]] = []
        self.commits = 0

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def commit(self) -> None:
        self.commits += 1

    def dispatch(self, query: str, params: Sequence[object] | None) -> ResultSet:
        self.statements.append((query, params))
        for key, queued in self._responses.items():
            if key in query:
                # The last scripted result repeats: a test that cares about one
                # answer should not have to script every later repetition of it.
                return queued.popleft() if len(queued) > 1 else queued[0]
        return ()

    def executed(self, needle: str) -> list[tuple[str, Sequence[object] | None]]:
        """Every statement containing ``needle``, in the order they were issued."""
        return [call for call in self.statements if needle in call[0]]


def version_row(
    *,
    id: int = 1,  # noqa: A002 — mirrors the column name
    engine: str = "clickhouse",
    namespace: str = "agentdb",
    fingerprint: str = "fp-1",
    layout_json: object = "",
    observed_at: datetime,
    superseded_at: datetime | None = None,
) -> Row:
    """A row in ``agentdb_schema_version`` column order."""
    return (id, engine, namespace, fingerprint, layout_json, observed_at, superseded_at)


def exemplar_row(
    *,
    id: int = 1,  # noqa: A002 — mirrors the column name
    engine: str = "clickhouse",
    namespace: str = "agentdb",
    question: str = "how many hits per counter?",
    sql: str = "SELECT CounterID, count() FROM hits GROUP BY CounterID",
    normalized_sql: str = "SELECT CounterID, count() FROM hits GROUP BY CounterID",
    relations: Sequence[str] = ("hits",),
    columns: Sequence[str] = ("CounterID",),
    schema_version_id: int = 1,
    outcome: Outcome = Outcome.SUCCESS,
    provenance: Provenance = Provenance.AGENT,
    valid_from: datetime,
    valid_to: datetime | None = None,
    tx_from: datetime,
    tx_to: datetime | None = None,
    rows_returned: int | None = None,
    bytes_read: int | None = None,
    duration_ms: int | None = None,
    error_class: str | None = None,
    error_text: str | None = None,
    embedding: object = None,
) -> Row:
    """A row in :data:`~agentdb.core.memory.store.EXEMPLAR_COLUMNS` order."""
    return (
        id,
        engine,
        namespace,
        question,
        sql,
        normalized_sql,
        list(relations),
        list(columns),
        schema_version_id,
        outcome.value,
        provenance.value,
        valid_from,
        valid_to,
        tx_from,
        tx_to,
        rows_returned,
        bytes_read,
        duration_ms,
        error_class,
        error_text,
        embedding,
    )
