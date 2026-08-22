"""The exemplar store against live Postgres + pgvector (SPEC §10, §12).

The unit tier proves the store's logic against a fake connection; it would pass
against a typo in the SQL. This is where the SQL itself is proven: the DDL
applies, ``VECTOR(1536)`` accepts what :func:`_vector` writes, ``ON CONFLICT``
does what the revert case needs, and ``relations && ...`` selects the pool.

Run with::

    make up
    uv sync --extra memory
    uv run pytest -m e2e tests/e2e/test_memory_store_live.py

It skips loudly when nothing is listening on the configured DSN, naming the
command that starts the container. Every test namespaces its rows under a fresh
UUID and drops nothing: the store's whole premise is that history is never
deleted, and a test that truncated it would be testing a different system.
"""

from __future__ import annotations

from collections.abc import Iterator
from uuid import uuid4

import pytest

from agentdb.adapters import ColumnDef, PhysicalLayout, RelationDetail, RelationRef
from agentdb.core.memory import ExemplarDraft, Outcome, normalize_sql, snapshot
from agentdb.core.memory.postgres import connect
from agentdb.core.memory.store import Connection, ExemplarStore

pytestmark = pytest.mark.e2e

QUESTION = "how many hits did each counter get?"
SQL = "SELECT CounterID, count() FROM hits GROUP BY CounterID"


@pytest.fixture(scope="module")
def connection() -> Iterator[Connection]:
    try:
        live = connect()
    except Exception as exc:
        pytest.skip(f"no exemplar store reachable ({exc}); start one with: make up")
    yield live
    live.commit()


@pytest.fixture
def store(connection: Connection) -> ExemplarStore:
    memory = ExemplarStore(connection)
    memory.ensure_schema()
    return memory


@pytest.fixture
def namespace() -> str:
    """A private namespace per test — the store is append-only by design."""
    return f"e2e_{uuid4().hex[:12]}"


def hits_snapshot(namespace: str, *, columns: tuple[str, ...], order_by: tuple[str, ...]):  # type: ignore[no-untyped-def]  # snapshot type is internal
    ref = RelationRef(namespace=namespace, name="hits")
    detail = RelationDetail(
        ref=ref,
        columns=tuple(
            ColumnDef(name=name, data_type="UInt32", is_nullable=False) for name in columns
        ),
        create_statement=f"CREATE TABLE {namespace}.hits (...)",
    )
    layout = PhysicalLayout(
        engine="clickhouse",
        ref=ref,
        create_statement=detail.create_statement,
        table_engine="MergeTree",
        order_by=order_by,
    )
    return snapshot("clickhouse", namespace, [detail], [layout])


def draft(namespace: str, **overrides: object) -> ExemplarDraft:
    fields: dict[str, object] = {
        "engine": "clickhouse",
        "namespace": namespace,
        "question": QUESTION,
        "sql": SQL,
        "normalized_sql": normalize_sql(SQL, "clickhouse"),
        "relations": ("hits",),
        "columns": ("CounterID",),
        "outcome": Outcome.SUCCESS,
        "bytes_read": 4_096,
    }
    fields.update(overrides)
    return ExemplarDraft(**fields)  # type: ignore[arg-type]  # test factory, keyed by field name


def test_the_ddl_applies_twice_without_complaint(store: ExemplarStore) -> None:
    store.ensure_schema()


def test_a_recorded_exemplar_comes_back_from_the_vector_pool(
    store: ExemplarStore, namespace: str
) -> None:
    store.sync(hits_snapshot(namespace, columns=("CounterID",), order_by=("CounterID",)))
    recorded = store.record(draft(namespace))

    found = store.retrieve(QUESTION, engine="clickhouse", namespace=namespace, relations=("hits",))

    assert [item.exemplar.id for item in found] == [recorded.id]
    assert len(found[0].exemplar.embedding) == 1536
    assert found[0].components["rel"] == 1.0


def test_a_schema_change_invalidates_the_exemplar_it_broke(
    store: ExemplarStore, namespace: str
) -> None:
    store.sync(
        hits_snapshot(namespace, columns=("CounterID", "EventDate"), order_by=("CounterID",))
    )
    store.record(draft(namespace, columns=("EventDate",)))

    result = store.sync(hits_snapshot(namespace, columns=("CounterID",), order_by=("CounterID",)))

    assert result.changed
    assert [item.reason for item in result.invalidated] == [
        "column 'EventDate' no longer exists on any relation this exemplar names"
    ]
    assert store.retrieve(QUESTION, engine="clickhouse", namespace=namespace) == ()


def test_a_sort_key_change_invalidates_nothing_a_column_change_would_not(
    store: ExemplarStore, namespace: str
) -> None:
    """The fingerprint moves; the exemplar still type-checks, so it survives."""
    store.sync(
        hits_snapshot(namespace, columns=("CounterID", "EventDate"), order_by=("CounterID",))
    )
    store.record(draft(namespace))

    result = store.sync(
        hits_snapshot(
            namespace, columns=("CounterID", "EventDate"), order_by=("EventDate", "CounterID")
        )
    )

    assert result.changed
    assert result.invalidated == ()
    assert len(store.retrieve(QUESTION, engine="clickhouse", namespace=namespace)) == 1


def test_a_reverted_schema_reactivates_its_own_version_row(
    store: ExemplarStore, namespace: str
) -> None:
    """``ON CONFLICT`` rather than a uniqueness violation on the way back."""
    original = hits_snapshot(namespace, columns=("CounterID",), order_by=("CounterID",))
    store.sync(original)
    store.sync(hits_snapshot(namespace, columns=("CounterID", "URL"), order_by=("CounterID",)))

    back = store.sync(original)

    assert back.changed
    assert back.version.is_current


def test_a_correction_appends_and_history_explains_the_break(
    store: ExemplarStore, namespace: str
) -> None:
    store.sync(
        hits_snapshot(namespace, columns=("CounterID", "EventDate"), order_by=("CounterID",))
    )
    store.record(draft(namespace, columns=("EventDate",)))
    store.record(draft(namespace, columns=("EventDate",), bytes_read=2_048))
    store.sync(hits_snapshot(namespace, columns=("CounterID",), order_by=("CounterID",)))

    history = store.history(
        engine="clickhouse",
        namespace=namespace,
        normalized_sql=normalize_sql(SQL, "clickhouse"),
    )

    assert len(history) == 2
    assert history[0].exemplar.tx_to is not None, "the corrected revision keeps its own row"
    assert history[-1].exemplar.valid_to is not None
    assert "EventDate" in (history[-1].reason or "")


def test_negative_exemplars_are_served_separately(store: ExemplarStore, namespace: str) -> None:
    store.sync(hits_snapshot(namespace, columns=("CounterID",), order_by=("CounterID",)))
    store.record(draft(namespace))
    store.record(
        draft(
            namespace,
            sql="SELECT UserID FROM hits",
            normalized_sql="SELECT UserID FROM hits",
            outcome=Outcome.ERROR,
            error_class="semantic",
            error_text="Code: 47. UNKNOWN_IDENTIFIER",
        )
    )

    positives = store.retrieve(QUESTION, engine="clickhouse", namespace=namespace)
    negatives = store.retrieve(
        QUESTION, engine="clickhouse", namespace=namespace, failures_only=True
    )

    assert [item.exemplar.outcome for item in positives] == [Outcome.SUCCESS]
    assert [item.exemplar.error_class for item in negatives] == ["semantic"]
