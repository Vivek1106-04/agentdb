"""The exemplar store itself: writes, invalidation, retrieval, history (SPEC §10).

Three things this module refuses to do, each for a reason the benchmark depends
on:

* **It does not mutate history.** A correction closes a transaction-time window
  and appends a new row; an invalidation closes a valid-time window. Nothing is
  UPDATEd except a ``NULL`` end-bound, and nothing is ever deleted, which is what
  makes ``explain_exemplar_history`` answerable at all.
* **It does not rank in SQL.** pgvector picks the candidate pool; the hybrid
  score of SPEC §10.4 runs in :mod:`agentdb.core.memory.ranking`, because every
  weight in it is an ablation arm and an arm has to be re-runnable over a fixed
  pool without a database.
* **It does not speak the** :class:`~agentdb.adapters.base.Adapter` **protocol.**
  Postgres is agentdb's private state store, not a measured engine, and the
  import-linter contract of SPEC §12 keeps it from becoming one.

The connection is a structural protocol rather than a psycopg type: the store is
then unit-testable at 100% without a live database, and the e2e tier proves the
SQL against the real one.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import resources
from typing import Protocol, SupportsInt, cast

from agentdb.adapters import Engine
from agentdb.config import Config
from agentdb.core.memory.embedding import Embedder, HashingEmbedder
from agentdb.core.memory.fingerprint import (
    NamespaceSnapshot,
    fingerprint,
    invalidation_reason,
    snapshot_from_json,
    snapshot_to_json,
)
from agentdb.core.memory.models import (
    Exemplar,
    ExemplarDraft,
    Outcome,
    Provenance,
    SchemaVersion,
    ScoredExemplar,
)
from agentdb.core.memory.ranking import rank


class MemoryStoreError(RuntimeError):
    """Raised when the store is asked for something its state cannot support."""


class Cursor(Protocol):
    """The cursor surface the store uses. Satisfied by ``psycopg.Cursor``."""

    def execute(self, query: str, params: Sequence[object] | None = None) -> object: ...

    def fetchall(self) -> Sequence[Sequence[object]]: ...

    def fetchone(self) -> Sequence[object] | None: ...

    def close(self) -> None: ...


class Connection(Protocol):
    """The connection surface the store uses. Satisfied by ``psycopg.Connection``."""

    def cursor(self) -> Cursor: ...

    def commit(self) -> None: ...


@dataclass(frozen=True, slots=True)
class Invalidation:
    """One exemplar that stopped being true, and the mechanical reason why."""

    exemplar_id: int
    reason: str


@dataclass(frozen=True, slots=True)
class SyncResult:
    """What observing a namespace's schema did to the store (SPEC §10.3)."""

    version: SchemaVersion
    changed: bool
    invalidated: tuple[Invalidation, ...] = ()


@dataclass(frozen=True, slots=True)
class Revision:
    """One transaction-time revision of an exemplar, for history (SPEC §10.3)."""

    exemplar: Exemplar
    fingerprint: str
    reason: str | None
    """Why valid time closed, recomputed against the schema that superseded it.

    ``None`` while the exemplar is still valid, and ``None`` too when the store
    holds no schema version covering the moment it was invalidated — an honest
    absence beats a guessed explanation.
    """


EXEMPLAR_COLUMNS = (
    "id, engine, namespace, question, sql, normalized_sql, relations, columns, "
    "schema_version_id, outcome, provenance, valid_from, valid_to, tx_from, tx_to, "
    "rows_returned, bytes_read, duration_ms, error_class, error_text, embedding"
)
"""Column order every ``SELECT`` in this module uses, and :func:`_exemplar` reads."""


def utcnow() -> datetime:
    """The store's clock. Injectable so tests can place events on both axes."""
    return datetime.now(UTC)


class ExemplarStore:
    """Bi-temporal query memory over Postgres + pgvector (SPEC §10)."""

    def __init__(
        self,
        connection: Connection,
        *,
        config: Config | None = None,
        embedder: Embedder | None = None,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self._connection = connection
        self._config = config or Config()
        self._embedder = embedder or HashingEmbedder()
        self._clock = clock

    # -- schema ------------------------------------------------------------

    def ensure_schema(self) -> None:
        """Apply the store's DDL. Idempotent, so a fresh clone needs no migration."""
        ddl = resources.files(__package__).joinpath("schema.sql").read_text(encoding="utf-8")
        self._run(ddl)
        self._connection.commit()

    # -- schema versions and invalidation ----------------------------------

    def sync(self, state: NamespaceSnapshot) -> SyncResult:
        """Record ``state`` as the current schema version, invalidating what it broke.

        Called on every ``describe_relation`` / ``physical_layout`` (SPEC §10.3).
        When the fingerprint is unchanged this is one ``SELECT`` and no writes,
        which is what makes it affordable on that path.
        """
        digest = fingerprint(state)
        current = self._current_version(state.engine, state.namespace)
        if current is not None and current.fingerprint == digest:
            return SyncResult(version=current, changed=False)

        now = self._clock()
        version = self._supersede(state, digest, now)
        invalidated = self._invalidate(state, now)
        self._connection.commit()
        return SyncResult(version=version, changed=True, invalidated=invalidated)

    def _current_version(self, engine: Engine, namespace: str) -> SchemaVersion | None:
        rows = self._query(
            "SELECT id, engine, namespace, fingerprint, layout_json, observed_at, superseded_at "
            "FROM agentdb_schema_version "
            "WHERE engine = %s AND namespace = %s AND superseded_at IS NULL "
            "ORDER BY observed_at DESC LIMIT 1",
            (engine, namespace),
        )
        return _version(rows[0]) if rows else None

    def _supersede(self, state: NamespaceSnapshot, digest: str, now: datetime) -> SchemaVersion:
        """Close the open version and open one for ``digest``.

        A schema that reverts to a shape it already held reactivates that row
        rather than colliding with its own uniqueness constraint — the store is
        recording *which* shape is current, not how many times it has been seen.
        """
        self._run(
            "UPDATE agentdb_schema_version SET superseded_at = %s "
            "WHERE engine = %s AND namespace = %s AND superseded_at IS NULL",
            (now, state.engine, state.namespace),
        )
        rows = self._query(
            "INSERT INTO agentdb_schema_version "
            "(engine, namespace, fingerprint, layout_json, observed_at) "
            "VALUES (%s, %s, %s, %s, %s) "
            "ON CONFLICT (engine, namespace, fingerprint) DO UPDATE "
            "SET observed_at = EXCLUDED.observed_at, superseded_at = NULL "
            "RETURNING id, engine, namespace, fingerprint, layout_json, observed_at, superseded_at",
            (
                state.engine,
                state.namespace,
                digest,
                json.dumps(snapshot_to_json(state)),
                now,
            ),
        )
        return _version(rows[0])

    def _invalidate(self, state: NamespaceSnapshot, now: datetime) -> tuple[Invalidation, ...]:
        """Stamp ``valid_to`` on every live exemplar the new schema broke.

        Re-validation is a set comparison against the names the exemplar
        recorded, never a re-execution: an invalidation sweep that ran queries
        would make every schema change cost a workload.
        """
        rows = self._query(
            f"SELECT {EXEMPLAR_COLUMNS} FROM agentdb_exemplar "
            "WHERE engine = %s AND namespace = %s AND valid_to IS NULL AND tx_to IS NULL",
            (state.engine, state.namespace),
        )
        previous = self._snapshots_by_version(row[8] for row in rows)

        invalidated: list[Invalidation] = []
        for row in rows:
            exemplar = _exemplar(row)
            reason = invalidation_reason(
                exemplar.relations,
                exemplar.columns,
                state,
                previous.get(exemplar.schema_version_id),
            )
            if reason is None:
                continue
            self._run(
                "UPDATE agentdb_exemplar SET valid_to = %s WHERE id = %s",
                (now, exemplar.id),
            )
            invalidated.append(Invalidation(exemplar_id=exemplar.id, reason=reason))
        return tuple(invalidated)

    def _snapshots_by_version(self, ids: Iterable[object]) -> Mapping[int, NamespaceSnapshot]:
        version_ids = sorted({_int(value) for value in ids})
        if not version_ids:
            return {}
        rows = self._query(
            "SELECT id, layout_json FROM agentdb_schema_version WHERE id = ANY(%s)",
            (version_ids,),
        )
        return {_int(row[0]): snapshot_from_json(_json(row[1])) for row in rows}

    # -- writes ------------------------------------------------------------

    def record(self, draft: ExemplarDraft) -> Exemplar:
        """Record ``draft`` against the current schema version.

        A draft whose normalized SQL is already remembered is a *correction*: the
        standing record's transaction-time window closes and the new one opens.
        The old row stays queryable, because "what did agentdb believe last
        month" is the question bi-temporality exists to answer.
        """
        version = self._current_version(draft.engine, draft.namespace)
        if version is None:
            raise MemoryStoreError(
                f"no schema version for {draft.engine}/{draft.namespace}: call sync() first"
            )

        now = self._clock()
        self._run(
            "UPDATE agentdb_exemplar SET tx_to = %s "
            "WHERE engine = %s AND namespace = %s AND normalized_sql = %s AND tx_to IS NULL",
            (now, draft.engine, draft.namespace, draft.normalized_sql),
        )
        rows = self._query(
            "INSERT INTO agentdb_exemplar "
            "(engine, namespace, question, sql, normalized_sql, relations, columns, "
            "schema_version_id, outcome, provenance, valid_from, tx_from, "
            "rows_returned, bytes_read, duration_ms, error_class, error_text, embedding) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
            "%s::vector) "
            f"RETURNING {EXEMPLAR_COLUMNS}",
            (
                draft.engine,
                draft.namespace,
                draft.question,
                draft.sql,
                draft.normalized_sql,
                list(draft.relations),
                list(draft.columns),
                version.id,
                draft.outcome.value,
                draft.provenance.value,
                now,
                now,
                draft.rows_returned,
                draft.bytes_read,
                draft.duration_ms,
                draft.error_class,
                draft.error_text,
                _vector(self._embedder.embed(draft.question)),
            ),
        )
        self._connection.commit()
        return _exemplar(rows[0])

    # -- retrieval ---------------------------------------------------------

    def retrieve(
        self,
        question: str,
        *,
        engine: Engine,
        namespace: str,
        relations: Sequence[str] = (),
        k: int | None = None,
        failures_only: bool = False,
    ) -> tuple[ScoredExemplar, ...]:
        """Return the best exemplars for ``question``, hybrid-ranked (SPEC §10.4).

        ``failures_only`` serves the negative exemplars of arm ``A5_negmemory``:
        previously-failed queries with their error class, offered as explicit "do
        not do this" context. They are a separate call rather than a mixed pool
        because the two sets are rendered differently and ablated separately.

        Only currently-valid, currently-current rows are candidates. An exemplar
        the schema has invalidated never reaches the ranking — which is the whole
        point of the store, and is enforced in the ``WHERE`` clause rather than
        left to a weight.
        """
        question_embedding = self._embedder.embed(question)
        outcome_filter = "outcome <> 'success'" if failures_only else "outcome = 'success'"
        rows = self._query(
            f"SELECT {EXEMPLAR_COLUMNS} FROM agentdb_exemplar "
            "WHERE engine = %s AND namespace = %s AND valid_to IS NULL AND tx_to IS NULL "
            f"AND {outcome_filter} "
            # Relation overlap first, then vector distance: a lexical embedder
            # can rank an exemplar over exactly the right tables below one that
            # merely shares vocabulary, and the pool has to contain both for the
            # w_rel ablation to be able to move anything.
            "ORDER BY (relations && %s::text[]) DESC, embedding <=> %s::vector "
            "LIMIT %s",
            (
                engine,
                namespace,
                list(relations),
                _vector(question_embedding),
                self._config.exemplar_candidate_pool,
            ),
        )
        return rank(
            [_exemplar(row) for row in rows],
            question_embedding=question_embedding,
            relations=relations,
            now=self._clock(),
            weights=self._config.retrieval_weights,
            tau_days=self._config.exemplar_recency_tau_days,
            limit=self._config.exemplar_top_k if k is None else k,
        )

    def history(
        self, *, engine: Engine, namespace: str, normalized_sql: str
    ) -> tuple[Revision, ...]:
        """Every revision of one remembered query, oldest first (SPEC §10.3).

        This is what ``explain_exemplar_history`` renders: when agentdb learned
        the query, when it stopped being true, and — recomputed against the
        schema version that superseded it — what changed underneath it.
        """
        rows = self._query(
            f"SELECT {EXEMPLAR_COLUMNS} FROM agentdb_exemplar "
            "WHERE engine = %s AND namespace = %s AND normalized_sql = %s "
            "ORDER BY tx_from, id",
            (engine, namespace, normalized_sql),
        )
        versions = self._version_timeline(engine, namespace)
        return tuple(self._revision(_exemplar(row), versions) for row in rows)

    def _revision(
        self,
        exemplar: Exemplar,
        versions: Sequence[tuple[SchemaVersion, NamespaceSnapshot]],
    ) -> Revision:
        written_under = next(
            (pair for pair in versions if pair[0].id == exemplar.schema_version_id), None
        )
        digest = written_under[0].fingerprint if written_under is not None else ""
        if exemplar.valid_to is None:
            return Revision(exemplar=exemplar, fingerprint=digest, reason=None)

        superseding = _version_at(versions, exemplar.valid_to)
        if superseding is None:
            return Revision(exemplar=exemplar, fingerprint=digest, reason=None)
        reason = invalidation_reason(
            exemplar.relations,
            exemplar.columns,
            superseding,
            written_under[1] if written_under is not None else None,
        )
        return Revision(exemplar=exemplar, fingerprint=digest, reason=reason)

    def _version_timeline(
        self, engine: Engine, namespace: str
    ) -> tuple[tuple[SchemaVersion, NamespaceSnapshot], ...]:
        rows = self._query(
            "SELECT id, engine, namespace, fingerprint, layout_json, observed_at, superseded_at "
            "FROM agentdb_schema_version WHERE engine = %s AND namespace = %s "
            "ORDER BY observed_at, id",
            (engine, namespace),
        )
        return tuple((_version(row), snapshot_from_json(_json(row[4]))) for row in rows)

    # -- plumbing ----------------------------------------------------------

    def _query(
        self, sql: str, params: Sequence[object] | None = None
    ) -> Sequence[Sequence[object]]:
        cursor = self._connection.cursor()
        try:
            cursor.execute(sql, params)
            return cursor.fetchall()
        finally:
            cursor.close()

    def _run(self, sql: str, params: Sequence[object] | None = None) -> None:
        cursor = self._connection.cursor()
        try:
            cursor.execute(sql, params)
        finally:
            cursor.close()


def _version_at(
    versions: Sequence[tuple[SchemaVersion, NamespaceSnapshot]], moment: datetime
) -> NamespaceSnapshot | None:
    """The snapshot that was current at ``moment``, or ``None`` if none covers it."""
    for version, state in versions:
        starts_before = version.observed_at <= moment
        ends_after = version.superseded_at is None or version.superseded_at > moment
        if starts_before and ends_after:
            return state
    return None


def _version(row: Sequence[object]) -> SchemaVersion:
    return SchemaVersion(
        id=_int(row[0]),
        engine=_engine(row[1]),
        namespace=str(row[2]),
        fingerprint=str(row[3]),
        layout_json=_json(row[4]),
        observed_at=_moment(row[5]),
        superseded_at=None if row[6] is None else _moment(row[6]),
    )


def _exemplar(row: Sequence[object]) -> Exemplar:
    return Exemplar(
        id=_int(row[0]),
        engine=_engine(row[1]),
        namespace=str(row[2]),
        question=str(row[3]),
        sql=str(row[4]),
        normalized_sql=str(row[5]),
        relations=_strings(row[6]),
        columns=_strings(row[7]),
        schema_version_id=_int(row[8]),
        outcome=Outcome(str(row[9])),
        provenance=Provenance(str(row[10])),
        valid_from=_moment(row[11]),
        valid_to=None if row[12] is None else _moment(row[12]),
        tx_from=_moment(row[13]),
        tx_to=None if row[14] is None else _moment(row[14]),
        rows_returned=None if row[15] is None else _int(row[15]),
        bytes_read=None if row[16] is None else _int(row[16]),
        duration_ms=None if row[17] is None else _int(row[17]),
        error_class=None if row[18] is None else str(row[18]),
        error_text=None if row[19] is None else str(row[19]),
        embedding=_parse_vector(row[20]),
    )


def _moment(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    raise MemoryStoreError(f"expected a timestamp from the store, got {type(value).__name__}")


def _int(value: object) -> int:
    """Read an integer column. The driver hands back ``object``; the DDL says otherwise."""
    return int(cast(SupportsInt, value))


def _engine(value: object) -> Engine:
    """Read the ``engine`` column, which this module is the only writer of."""
    return cast(Engine, value)


def _strings(value: object) -> tuple[str, ...]:
    """Read a ``TEXT[]`` column."""
    return tuple(str(item) for item in cast(Sequence[object], value))


def _json(value: object) -> Mapping[str, object]:
    """Decode a ``JSONB`` column, whether the driver parsed it or handed back text."""
    if isinstance(value, str):
        decoded: Mapping[str, object] = json.loads(value)
        return decoded
    if isinstance(value, Mapping):
        return value
    raise MemoryStoreError(f"expected JSON from the store, got {type(value).__name__}")


def _vector(embedding: Sequence[float]) -> str:
    """Render an embedding as pgvector's own text form.

    Text rather than a registered type adapter so the store keeps working
    through any driver that speaks the DB-API — including the fake the unit
    tests use, which is what lets this module be covered without a database.
    """
    return "[" + ",".join(repr(float(value)) for value in embedding) + "]"


def _parse_vector(value: object) -> tuple[float, ...]:
    """Read back what :func:`_vector` wrote, or what pgvector returned."""
    if value is None:
        return ()
    if isinstance(value, str):
        stripped = value.strip().strip("[]")
        if not stripped:
            return ()
        return tuple(float(part) for part in stripped.split(","))
    return tuple(float(part) for part in cast(Sequence[float], value))
