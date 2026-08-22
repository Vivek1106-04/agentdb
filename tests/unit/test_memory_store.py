"""The exemplar store's behaviour against a scripted connection (SPEC §10).

What is under test is the part that would be wrong silently: which time window
closes on which event, what reaches the ranking, and whether an invalidated
exemplar can still be served. The SQL's validity is the e2e tier's problem —
these tests would happily pass against a typo, which is why
``tests/e2e/test_memory_store_e2e.py`` exists.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from agentdb.adapters import ColumnDef, RelationDetail, RelationRef
from agentdb.config import Config
from agentdb.core.memory import (
    ExemplarDraft,
    NamespaceSnapshot,
    Outcome,
    fingerprint,
    snapshot,
    snapshot_to_json,
)
from agentdb.core.memory.store import (
    EXEMPLAR_COLUMNS,
    ExemplarStore,
    MemoryStoreError,
    utcnow,
)
from tests.memory_fakes import FakeConnection, Row, exemplar_row, version_row

NOW = datetime(2026, 8, 22, 12, 0, tzinfo=UTC)
EARLIER = NOW - timedelta(days=10)

HITS = RelationRef(namespace="agentdb", name="hits")


def hits_snapshot(*, columns: tuple[str, ...] = ("CounterID", "EventDate")) -> NamespaceSnapshot:
    return snapshot(
        "clickhouse",
        "agentdb",
        [
            RelationDetail(
                ref=HITS,
                columns=tuple(
                    ColumnDef(name=name, data_type="UInt32", is_nullable=False) for name in columns
                ),
                create_statement="CREATE TABLE agentdb.hits (...)",
            )
        ],
    )


def store(connection: FakeConnection, **kwargs: object) -> ExemplarStore:
    return ExemplarStore(connection, clock=lambda: NOW, **kwargs)  # type: ignore[arg-type]  # kwargs


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------


def test_ensure_schema_applies_the_ddl_and_commits() -> None:
    connection = FakeConnection()

    store(connection).ensure_schema()

    ddl = connection.statements[0][0]
    assert "CREATE TABLE IF NOT EXISTS agentdb_exemplar" in ddl
    assert "USING hnsw (embedding vector_cosine_ops)" in ddl
    assert connection.commits == 1


# --------------------------------------------------------------------------
# schema versions and invalidation
# --------------------------------------------------------------------------


def test_an_unchanged_fingerprint_writes_nothing() -> None:
    state = hits_snapshot()
    connection = FakeConnection(
        {
            "ORDER BY observed_at DESC LIMIT 1": [
                [
                    version_row(
                        fingerprint=fingerprint(state),
                        layout_json=json.dumps(snapshot_to_json(state)),
                        observed_at=EARLIER,
                    )
                ]
            ]
        }
    )

    result = store(connection).sync(state)

    assert not result.changed
    assert result.invalidated == ()
    assert connection.executed("UPDATE") == []
    assert connection.commits == 0


def test_a_first_observation_opens_a_version() -> None:
    state = hits_snapshot()
    connection = FakeConnection(
        {
            "INSERT INTO agentdb_schema_version": [
                [
                    version_row(
                        id=4,
                        fingerprint=fingerprint(state),
                        layout_json=json.dumps(snapshot_to_json(state)),
                        observed_at=NOW,
                    )
                ]
            ]
        }
    )

    result = store(connection).sync(state)

    assert result.changed
    assert result.version.id == 4
    assert result.version.is_current
    assert connection.commits == 1


def test_a_broken_exemplar_is_stamped_and_a_surviving_one_is_left_alone() -> None:
    before = hits_snapshot()
    after = hits_snapshot(columns=("CounterID",))
    connection = FakeConnection(
        {
            "ORDER BY observed_at DESC LIMIT 1": [
                [
                    version_row(
                        fingerprint=fingerprint(before),
                        layout_json=json.dumps(snapshot_to_json(before)),
                        observed_at=EARLIER,
                    )
                ]
            ],
            "INSERT INTO agentdb_schema_version": [
                [
                    version_row(
                        id=2,
                        fingerprint=fingerprint(after),
                        layout_json=json.dumps(snapshot_to_json(after)),
                        observed_at=NOW,
                    )
                ]
            ],
            "valid_to IS NULL AND tx_to IS NULL": [
                [
                    exemplar_row(
                        id=11, columns=("EventDate",), valid_from=EARLIER, tx_from=EARLIER
                    ),
                    exemplar_row(
                        id=12, columns=("CounterID",), valid_from=EARLIER, tx_from=EARLIER
                    ),
                ]
            ],
            "WHERE id = ANY(%s)": [[(1, json.dumps(snapshot_to_json(before)))]],
        }
    )

    result = store(connection).sync(after)

    assert [item.exemplar_id for item in result.invalidated] == [11]
    assert (
        result.invalidated[0].reason
        == "column 'EventDate' no longer exists on any relation this exemplar names"
    )
    stamped = connection.executed("SET valid_to")
    assert len(stamped) == 1
    assert stamped[0][1] == (NOW, 11)


def test_the_previous_version_is_closed_before_the_new_one_opens() -> None:
    state = hits_snapshot()
    connection = FakeConnection(
        {
            "INSERT INTO agentdb_schema_version": [
                [
                    version_row(
                        id=2, layout_json=json.dumps(snapshot_to_json(state)), observed_at=NOW
                    )
                ]
            ]
        }
    )

    store(connection).sync(state)

    closing = connection.executed("SET superseded_at")
    assert closing[0][1] == (NOW, "clickhouse", "agentdb")
    assert connection.statements.index(closing[0]) < next(
        index
        for index, call in enumerate(connection.statements)
        if "INSERT INTO agentdb_schema_version" in call[0]
    )


# --------------------------------------------------------------------------
# writes
# --------------------------------------------------------------------------


def draft(**overrides: object) -> ExemplarDraft:
    fields: dict[str, object] = {
        "engine": "clickhouse",
        "namespace": "agentdb",
        "question": "how many hits per counter?",
        "sql": "SELECT CounterID, count() FROM hits GROUP BY CounterID",
        "normalized_sql": "SELECT CounterID, count() FROM hits GROUP BY CounterID",
        "relations": ("hits",),
        "columns": ("CounterID",),
        "outcome": Outcome.SUCCESS,
    }
    fields.update(overrides)
    return ExemplarDraft(**fields)  # type: ignore[arg-type]  # test factory, keyed by field name


def recording_connection() -> FakeConnection:
    return FakeConnection(
        {
            "ORDER BY observed_at DESC LIMIT 1": [
                [version_row(id=3, observed_at=EARLIER, layout_json="{}")]
            ],
            "INSERT INTO agentdb_exemplar": [
                [exemplar_row(id=21, schema_version_id=3, valid_from=NOW, tx_from=NOW)]
            ],
        }
    )


def test_recording_without_a_schema_version_fails_rather_than_guessing() -> None:
    with pytest.raises(MemoryStoreError, match="call sync"):
        store(FakeConnection()).record(draft())


def test_a_recorded_exemplar_binds_to_the_current_schema_version() -> None:
    connection = recording_connection()

    exemplar = store(connection).record(draft())

    assert exemplar.id == 21
    assert exemplar.schema_version_id == 3
    assert exemplar.is_valid and exemplar.is_current
    assert connection.commits == 1


def test_a_correction_closes_the_standing_record_rather_than_overwriting_it() -> None:
    connection = recording_connection()

    store(connection).record(draft())

    closing = connection.executed("SET tx_to")
    assert len(closing) == 1
    assert closing[0][1] == (
        NOW,
        "clickhouse",
        "agentdb",
        "SELECT CounterID, count() FROM hits GROUP BY CounterID",
    )


def test_the_question_is_embedded_on_the_way_in() -> None:
    connection = recording_connection()

    store(connection).record(draft())

    params = connection.executed("INSERT INTO agentdb_exemplar")[0][1]
    assert params is not None
    vector = params[-1]
    assert isinstance(vector, str)
    assert vector.startswith("[") and vector.endswith("]")
    assert len(vector.split(",")) == 1536


def test_a_failed_query_is_recordable_with_its_error_class() -> None:
    connection = FakeConnection(
        {
            "ORDER BY observed_at DESC LIMIT 1": [
                [version_row(id=3, observed_at=EARLIER, layout_json="{}")]
            ],
            "INSERT INTO agentdb_exemplar": [
                [
                    exemplar_row(
                        id=22,
                        outcome=Outcome.ERROR,
                        error_class="semantic",
                        error_text="Code: 47 UNKNOWN_IDENTIFIER",
                        schema_version_id=3,
                        valid_from=NOW,
                        tx_from=NOW,
                    )
                ]
            ],
        }
    )

    exemplar = store(connection).record(
        draft(outcome=Outcome.ERROR, error_class="semantic", error_text="Code: 47")
    )

    assert exemplar.is_negative
    assert exemplar.error_class == "semantic"


# --------------------------------------------------------------------------
# retrieval
# --------------------------------------------------------------------------


def retrieval_connection(*rows: Row) -> FakeConnection:
    return FakeConnection({"ORDER BY (relations &&": [list(rows)]})


def test_retrieval_ranks_the_pool_and_honours_k() -> None:
    connection = retrieval_connection(
        exemplar_row(id=31, relations=("visits",), valid_from=EARLIER, tx_from=EARLIER),
        exemplar_row(id=32, relations=("hits",), valid_from=EARLIER, tx_from=NOW),
    )

    found = store(connection).retrieve(
        "how many hits per counter?",
        engine="clickhouse",
        namespace="agentdb",
        relations=("hits",),
        k=1,
    )

    assert [item.exemplar.id for item in found] == [32]
    assert set(found[0].components) == {"sem", "rel", "success", "recency", "cost"}


def test_retrieval_asks_only_for_live_successful_rows_of_one_namespace() -> None:
    connection = retrieval_connection()

    store(connection).retrieve("q", engine="clickhouse", namespace="agentdb")

    query, params = connection.executed("ORDER BY (relations &&")[0]
    assert "valid_to IS NULL AND tx_to IS NULL" in query
    assert "outcome = 'success'" in query
    assert params is not None
    assert params[0] == "clickhouse"
    assert params[-1] == Config().exemplar_candidate_pool


def test_negative_exemplars_are_a_separate_call_not_a_mixed_pool() -> None:
    connection = retrieval_connection()

    store(connection).retrieve("q", engine="clickhouse", namespace="agentdb", failures_only=True)

    query, _ = connection.executed("ORDER BY (relations &&")[0]
    assert "outcome <> 'success'" in query


def test_k_defaults_to_the_configured_context_budget() -> None:
    connection = retrieval_connection(
        *(
            exemplar_row(id=index, valid_from=EARLIER, tx_from=EARLIER)
            for index in range(1, Config().exemplar_top_k + 4)
        )
    )

    found = store(connection).retrieve("q", engine="clickhouse", namespace="agentdb")

    assert len(found) == Config().exemplar_top_k


def test_a_stored_embedding_is_read_back_into_the_ranking() -> None:
    connection = retrieval_connection(
        exemplar_row(id=41, embedding="[1.0,0.0]", valid_from=EARLIER, tx_from=EARLIER),
    )

    found = store(connection).retrieve("q", engine="clickhouse", namespace="agentdb")

    assert found[0].exemplar.embedding == (1.0, 0.0)


# --------------------------------------------------------------------------
# history
# --------------------------------------------------------------------------


def history_connection(*, revisions: list[Row], versions: list[Row]) -> FakeConnection:
    return FakeConnection(
        {
            "ORDER BY tx_from, id": [revisions],
            "ORDER BY observed_at, id": [versions],
        }
    )


def test_history_explains_what_changed_underneath_an_invalidated_query() -> None:
    before = hits_snapshot()
    after = hits_snapshot(columns=("CounterID",))
    connection = history_connection(
        revisions=[
            exemplar_row(
                id=51,
                columns=("EventDate",),
                schema_version_id=1,
                valid_from=EARLIER,
                valid_to=NOW,
                tx_from=EARLIER,
            )
        ],
        versions=[
            version_row(
                id=1,
                fingerprint="fp-before",
                layout_json=json.dumps(snapshot_to_json(before)),
                observed_at=EARLIER,
                superseded_at=NOW,
            ),
            version_row(
                id=2,
                fingerprint="fp-after",
                layout_json=json.dumps(snapshot_to_json(after)),
                observed_at=NOW,
            ),
        ],
    )

    history = store(connection).history(
        engine="clickhouse", namespace="agentdb", normalized_sql="SELECT ?"
    )

    assert history[0].fingerprint == "fp-before"
    assert history[0].reason is not None
    assert "EventDate" in history[0].reason


def test_a_still_valid_revision_has_no_reason() -> None:
    state = hits_snapshot()
    connection = history_connection(
        revisions=[exemplar_row(id=52, schema_version_id=1, valid_from=EARLIER, tx_from=EARLIER)],
        versions=[
            version_row(
                id=1,
                layout_json=json.dumps(snapshot_to_json(state)),
                observed_at=EARLIER,
            )
        ],
    )

    history = store(connection).history(
        engine="clickhouse", namespace="agentdb", normalized_sql="SELECT ?"
    )

    assert history[0].reason is None


def test_an_invalidation_no_stored_version_covers_is_reported_as_unexplained() -> None:
    """An honest absence beats a guessed explanation."""
    state = hits_snapshot()
    connection = history_connection(
        revisions=[
            exemplar_row(
                id=53,
                schema_version_id=1,
                valid_from=EARLIER,
                valid_to=EARLIER - timedelta(days=5),
                tx_from=EARLIER,
            )
        ],
        versions=[
            version_row(
                id=1,
                layout_json=json.dumps(snapshot_to_json(state)),
                observed_at=EARLIER,
            )
        ],
    )

    history = store(connection).history(
        engine="clickhouse", namespace="agentdb", normalized_sql="SELECT ?"
    )

    assert history[0].reason is None


def test_a_revision_whose_version_row_is_gone_still_lists() -> None:
    after = hits_snapshot(columns=("CounterID",))
    connection = history_connection(
        revisions=[
            exemplar_row(
                id=54,
                columns=("EventDate",),
                schema_version_id=99,
                valid_from=EARLIER,
                valid_to=NOW,
                tx_from=EARLIER,
            )
        ],
        versions=[
            version_row(
                id=2,
                fingerprint="fp-after",
                layout_json=json.dumps(snapshot_to_json(after)),
                observed_at=EARLIER,
            )
        ],
    )

    history = store(connection).history(
        engine="clickhouse", namespace="agentdb", normalized_sql="SELECT ?"
    )

    assert history[0].fingerprint == ""
    assert history[0].reason is not None


# --------------------------------------------------------------------------
# reading what the driver returned
# --------------------------------------------------------------------------


def test_a_column_the_ddl_says_is_a_timestamp_but_is_not_fails_loudly() -> None:
    connection = retrieval_connection(
        exemplar_row(id=61, valid_from="not-a-timestamp", tx_from=NOW),  # type: ignore[arg-type]
    )

    with pytest.raises(MemoryStoreError, match="expected a timestamp"):
        store(connection).retrieve("q", engine="clickhouse", namespace="agentdb")


def test_jsonb_is_read_whether_the_driver_parsed_it_or_not() -> None:
    state = hits_snapshot()
    parsed = FakeConnection(
        {
            "ORDER BY observed_at DESC LIMIT 1": [
                [
                    version_row(
                        fingerprint=fingerprint(state),
                        layout_json=snapshot_to_json(state),
                        observed_at=EARLIER,
                    )
                ]
            ]
        }
    )

    assert not store(parsed).sync(state).changed


def test_a_layout_column_that_is_neither_json_nor_a_mapping_fails_loudly() -> None:
    state = hits_snapshot()
    connection = FakeConnection(
        {
            "ORDER BY observed_at DESC LIMIT 1": [
                [version_row(fingerprint="fp", layout_json=17, observed_at=EARLIER)]
            ]
        }
    )

    with pytest.raises(MemoryStoreError, match="expected JSON"):
        store(connection).sync(state)


def test_an_embedding_reads_back_from_every_shape_a_driver_may_hand_over() -> None:
    connection = retrieval_connection(
        exemplar_row(id=71, embedding=None, valid_from=EARLIER, tx_from=EARLIER),
        exemplar_row(id=72, embedding="[]", valid_from=EARLIER, tx_from=EARLIER),
        exemplar_row(id=73, embedding=[0.5, 0.25], valid_from=EARLIER, tx_from=EARLIER),
    )

    found = store(connection).retrieve("q", engine="clickhouse", namespace="agentdb", k=3)
    embeddings = {item.exemplar.id: item.exemplar.embedding for item in found}

    assert embeddings[71] == ()
    assert embeddings[72] == ()
    assert embeddings[73] == (0.5, 0.25)


def test_every_cursor_is_closed_even_though_the_store_holds_no_transaction() -> None:
    connection = FakeConnection()

    store(connection).ensure_schema()

    assert connection.statements  # the fake records through the cursor it closed


def test_the_default_clock_is_the_wall_clock() -> None:
    assert utcnow().tzinfo is UTC


def test_the_column_list_and_the_row_reader_agree() -> None:
    """A column added to one and not the other would shift every field silently."""
    assert len(EXEMPLAR_COLUMNS.split(",")) == len(exemplar_row(valid_from=NOW, tx_from=NOW))
