"""``A6_full``: the advisor's findings as context (SPEC §11.3, §9).

Two things decide whether this arm means anything. The advice has to reach the
model — as the *fact*, not as a migration script it cannot run — and the demand
signal behind it has to come from somewhere that is neither this project's task
suite nor a query log holding this project's own gold executions.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import ModuleType, SimpleNamespace
from typing import Any, cast

from agentdb.adapters import ColumnProfile
from agentdb.bench import build_memory_provider, build_provider
from agentdb.bench.advised_provider import (
    AdvisedContextProvider,
    build_advised_provider,
    clickhouse_advised_provider,
    load_workload,
)
from agentdb.core.advisor.render import HEADER
from agentdb.core.memory.store import Connection, ExemplarStore
from tests.fakes import FakeAdapter, clickhouse_hits_fixture
from tests.memory_fakes import FakeConnection, version_row

QUESTION = "how many hits per counter?"
NOW = datetime(2026, 8, 23, 12, 0, tzinfo=UTC)


def store_connection() -> FakeConnection:
    return FakeConnection(
        {
            "INSERT INTO agentdb_schema_version": [
                [version_row(id=1, layout_json="{}", observed_at=NOW)]
            ]
        }
    )


def advised(
    *,
    adapter: FakeAdapter | None = None,
    workload: str = "clickbench",
    level: str = "layout",
) -> AdvisedContextProvider:
    live = adapter or clickhouse_hits_fixture()
    memory = build_memory_provider(
        base=build_provider(adapter=live, level=level),
        store=ExemplarStore(store_connection(), clock=lambda: NOW),
        include_failures=True,
    )
    return build_advised_provider(base=memory, workload=workload, name="agentdb/A6_full")


# --------------------------------------------------------------------------
# the reference workload
# --------------------------------------------------------------------------


def test_the_shipped_workloads_parse_into_statements() -> None:
    clickbench = load_workload("clickbench")
    tpch = load_workload("tpch")

    assert len(clickbench) > 20
    assert len(tpch) > 10
    assert all(statement.upper().startswith("SELECT") for statement in clickbench)
    assert not any("--" in statement for statement in clickbench), "comments stripped"


def test_a_workload_can_come_from_a_path_the_operator_names(tmp_path: object) -> None:
    from pathlib import Path

    path = cast(Path, tmp_path) / "mine.sql"
    path.write_text("-- my own\nSELECT count() FROM hits WHERE UserID = 7;\n", encoding="utf-8")

    assert load_workload(str(path)) == ("SELECT count() FROM hits WHERE UserID = 7",)


def test_the_workload_is_part_of_the_arms_fingerprint(tmp_path: object) -> None:
    """A run whose demand signal changed is not the same arm."""
    from pathlib import Path

    other = cast(Path, tmp_path) / "other.sql"
    other.write_text("SELECT count() FROM hits WHERE URL LIKE '%x%';\n", encoding="utf-8")

    assert advised().fingerprint != advised(workload=str(other)).fingerprint


# --------------------------------------------------------------------------
# what reaches the model
# --------------------------------------------------------------------------


async def test_the_advice_block_follows_the_a5_payload() -> None:
    provider = advised()

    payload = await provider.context(namespace="agentdb", question=QUESTION)

    assert "sort key (ORDER BY)" in payload, "still the A2 grounding underneath"
    assert HEADER in payload
    assert payload.index("sort key (ORDER BY)") < payload.index(HEADER)


async def test_the_agent_is_given_the_fact_and_not_the_migration() -> None:
    """An agent answering a question cannot run ALTER TABLE, and knows it."""
    payload = await advised().context(namespace="agentdb", question=QUESTION)

    advice = payload[payload.index(HEADER) :]
    assert "ALTER TABLE" not in advice
    assert "CREATE TABLE" not in advice
    assert "MATERIALIZE" not in advice


async def test_every_finding_says_how_confident_it_is() -> None:
    payload = await advised().context(namespace="agentdb", question=QUESTION)

    advice = payload[payload.index(HEADER) :]
    lines = [line for line in advice.splitlines() if line.startswith("- ")]
    assert lines
    assert all(line.startswith("- [") for line in lines), "each finding is labelled"


async def test_advice_is_computed_once_per_namespace_not_once_per_question() -> None:
    adapter = clickhouse_hits_fixture()
    provider = advised(adapter=adapter)

    first = await provider.context(namespace="agentdb", question=QUESTION)
    second = await provider.context(namespace="agentdb", question="something else")

    assert first[first.index(HEADER) :] == second[second.index(HEADER) :]
    assert len(adapter.calls_named("physical_layout")) == 1


async def test_a_namespace_the_workload_never_mentions_earns_no_advice_block() -> None:
    """A reference workload for another table is not evidence about this one."""
    adapter = clickhouse_hits_fixture()

    provider = advised(adapter=adapter, workload="tpch")
    payload = await provider.context(namespace="agentdb", question=QUESTION)

    assert HEADER not in payload


async def test_a_payload_carrying_no_layout_carries_no_advice() -> None:
    """Below A2 there is no physical design to reason about, so A6 adds nothing."""
    payload = await advised(level="stats").context(namespace="agentdb", question=QUESTION)

    assert HEADER not in payload


async def test_the_plan_review_of_the_arms_below_is_still_delegated() -> None:
    provider = advised()

    assert await provider.explain_plan(sql="SELECT 1", namespace="agentdb") is None


async def test_the_factory_builds_the_whole_ladder_beneath_a6() -> None:
    """A6 is A5 plus advice: the memory layer under it must carry the negatives."""
    live = store_connection()

    async def get_async_client(**_: Any) -> object:
        return SimpleNamespace(query=None)

    driver = SimpleNamespace(get_async_client=get_async_client)

    provider = await clickhouse_advised_provider(
        connector=lambda _dsn: cast(Connection, live),
        importer=lambda _name: cast(ModuleType, driver),
    )

    assert provider.name == "agentdb/A6_full"
    assert provider.base.name == "agentdb/A6_full/A5"
    assert provider.base.include_failures is True


def test_a_databricks_namespace_is_advised_by_the_databricks_advisor() -> None:
    """The engine decides which rules run — not the arm, and not the workload file."""
    from tests.fakes import databricks_tpch_fixture

    adapter = databricks_tpch_fixture()
    adapter.profiles = {
        "l_shipdate": ColumnProfile(
            name="l_shipdate",
            data_type="date",
            sample_method="sample",
            sampled_rows=1_000,
            approx_distinct=2_500,
        )
    }
    memory = build_memory_provider(
        base=build_provider(adapter=adapter),
        store=ExemplarStore(store_connection(), clock=lambda: NOW),
    )

    provider = build_advised_provider(base=memory, workload="tpch")

    assert provider.base.base.builder.adapter.engine == "databricks"
