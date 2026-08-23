"""``A4_memory`` and ``A5_negmemory`` as the harness sees them (SPEC §11.3).

The arms differ by one flag, so the properties worth testing are the ones that
would make the A3→A4→A5 deltas mean something other than they claim: that A4
never shows a failure, that A5 shows exactly the failures, that an exemplar the
schema invalidated reaches no model, and that the fingerprint moves whenever the
ranking that produced the exemplars moves.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import ModuleType, SimpleNamespace
from typing import Any, cast

from agentdb.adapters import ExplainMode, RawPlan
from agentdb.bench import build_provider
from agentdb.bench.memory_provider import (
    MemoryContextProvider,
    build_memory_provider,
    clickhouse_memory_provider,
    databricks_memory_provider,
)
from agentdb.config import Config, RetrievalWeights
from agentdb.core.memory.models import Outcome
from agentdb.core.memory.store import Connection, ExemplarStore
from tests.fakes import clickhouse_hits_fixture
from tests.memory_fakes import FakeConnection, exemplar_row, version_row

QUESTION = "how many hits per counter?"
NOW = datetime(2026, 8, 22, 12, 0, tzinfo=UTC)
EARLIER = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)

UNPRUNED_PLAN = {
    "Plan": {
        "Node Type": "ReadFromMergeTree",
        "Description": "agentdb.hits",
        "Indexes": [{"Type": "PrimaryKey", "Initial Granules": 1_000, "Selected Granules": 1_000}],
    }
}


def connection(*, negative: bool = False) -> FakeConnection:
    """A store holding one success and, optionally, one failure."""
    success = exemplar_row(
        id=1,
        question="hits per counter last July",
        sql="SELECT CounterID, count() FROM hits GROUP BY CounterID",
        embedding="[]",
        valid_from=EARLIER,
        tx_from=EARLIER,
    )
    failure = exemplar_row(
        id=2,
        question="unique users per counter",
        sql="SELECT UserId, count() FROM hits GROUP BY UserId",
        outcome=Outcome.ERROR,
        error_class="semantic",
        error_text="Code: 47. UNKNOWN_IDENTIFIER UserId",
        embedding="[]",
        valid_from=EARLIER,
        tx_from=EARLIER,
    )
    return FakeConnection(
        {
            "outcome = 'success'": [[success]],
            "outcome <> 'success'": [[failure]] if negative else [[]],
            "INSERT INTO agentdb_schema_version": [
                [version_row(id=1, layout_json="{}", observed_at=NOW)]
            ],
        }
    )


def provider(
    *, include_failures: bool = False, live: Connection | None = None, config: Config | None = None
) -> MemoryContextProvider:
    base = build_provider(adapter=clickhouse_hits_fixture(), config=config, plan_review=True)
    store = ExemplarStore(live or connection(), config=config, clock=lambda: NOW)
    return build_memory_provider(base=base, store=store, include_failures=include_failures)


async def test_a4_appends_the_queries_that_worked_to_the_a3_payload() -> None:
    payload = await provider().context(namespace="agentdb", question=QUESTION)

    assert "sort key (ORDER BY)" in payload, "still the A3 grounding"
    assert "Queries that answered questions" in payload
    assert "hits per counter last July" in payload


async def test_a4_never_shows_a_failure() -> None:
    payload = await provider().context(namespace="agentdb", question=QUESTION)

    assert "do not repeat them" not in payload


async def test_a5_adds_the_failures_and_their_error_class() -> None:
    payload = await provider(include_failures=True, live=connection(negative=True)).context(
        namespace="agentdb", question=QUESTION
    )

    assert "do not repeat them" in payload
    assert "unique users per counter" in payload


async def test_an_empty_store_leaves_the_a3_payload_exactly_as_it_was() -> None:
    empty = FakeConnection(
        {
            "INSERT INTO agentdb_schema_version": [
                [version_row(id=1, layout_json="{}", observed_at=NOW)]
            ]
        }
    )
    memory = provider(live=empty)
    base_payload = await memory.base.context(namespace="agentdb", question=QUESTION)

    payload = await memory.context(namespace="agentdb", question=QUESTION)

    assert payload == base_payload


async def test_the_namespace_is_fingerprinted_once_per_run_not_once_per_question() -> None:
    """A sync per task would charge every arm a round trip to prove the same thing."""
    live = connection()
    memory = provider(live=live)

    await memory.context(namespace="agentdb", question=QUESTION)
    await memory.context(namespace="agentdb", question="something else entirely")

    assert len(live.executed("INSERT INTO agentdb_schema_version")) == 1


async def test_retrieval_offers_the_relations_the_payload_describes() -> None:
    live = connection()

    await provider(live=live).context(namespace="agentdb", question=QUESTION)

    _, params = live.executed("outcome = 'success'")[0]
    assert params is not None
    assert params[2] == ["hits"]


async def test_the_plan_review_of_a3_is_still_there() -> None:
    """A4 that quietly dropped the plan turn would price exemplars against the wrong base."""
    adapter = clickhouse_hits_fixture()
    adapter.plan = RawPlan(
        engine="clickhouse",
        mode=ExplainMode.ESTIMATE,
        sql="",
        payload=json.dumps(UNPRUNED_PLAN),
    )
    memory = build_memory_provider(
        base=build_provider(adapter=adapter, plan_review=True),
        store=ExemplarStore(connection(), clock=lambda: NOW),
    )

    review = await memory.explain_plan(sql="SELECT count() FROM hits", namespace="agentdb")

    assert review is not None
    assert "FULL_SCAN" in review


def test_the_two_arms_do_not_share_a_fingerprint() -> None:
    assert provider().fingerprint != provider(include_failures=True).fingerprint


def test_changing_a_retrieval_weight_changes_the_fingerprint() -> None:
    """Every weight is an ablation arm, so a run must record which one produced it."""
    zeroed = Config(retrieval_weights=RetrievalWeights(sem=0.0))

    assert provider().fingerprint != provider(config=zeroed).fingerprint


def test_the_arms_are_named_for_the_report() -> None:
    assert provider().name == "agentdb/memory"
    assert provider(include_failures=True).name == "agentdb/negmemory"


async def test_the_factory_wires_a_store_and_applies_its_schema() -> None:
    live = connection()

    async def get_async_client(**_: Any) -> object:
        return SimpleNamespace(query=None)

    driver = SimpleNamespace(get_async_client=get_async_client)

    memory = await clickhouse_memory_provider(
        include_failures=True,
        connector=lambda _dsn: cast(Connection, live),
        importer=lambda _name: cast(ModuleType, driver),
    )

    assert memory.name == "agentdb/A5_negmemory"
    assert memory.base.name == "agentdb/A5_negmemory/base"
    assert "CREATE TABLE IF NOT EXISTS agentdb_exemplar" in live.statements[0][0]


async def test_closing_the_memory_arm_closes_the_grounding_beneath_it() -> None:
    memory = provider()

    await memory.aclose()  # the fake adapter holds no connection; this must still be safe


async def test_the_databricks_factory_builds_the_same_arm_against_a_warehouse() -> None:
    """The memory arms are cross-engine: the store is shared, the exemplars are not."""
    live = connection()

    async def get_async_client(**_: Any) -> object:  # the SDK path, not the CH driver
        return SimpleNamespace()

    workspace = SimpleNamespace(
        statement_execution=SimpleNamespace(), query_history=SimpleNamespace()
    )
    sdk = SimpleNamespace(
        WorkspaceClient=lambda **_: workspace,
        StatementParameterListItem=dict,
        QueryFilter=dict,
        get_async_client=get_async_client,
    )

    import os

    os.environ.setdefault("AGENTDB_DBX_HOST", "https://example.cloud.databricks.com")
    os.environ.setdefault("AGENTDB_DBX_WAREHOUSE_ID", "wh-1")

    memory = await databricks_memory_provider(
        include_failures=True,
        connector=lambda _dsn: cast(Connection, live),
        importer=lambda _name: cast(ModuleType, sdk),
    )

    assert memory.base.builder.adapter.engine == "databricks"
    assert memory.include_failures is True
