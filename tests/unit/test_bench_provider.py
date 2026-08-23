"""agentdb's side of the benchmark seam (SPEC §4.1.6).

The provider is what makes agentdb scoreable by a harness that refuses to import
it. Two properties are load-bearing and tested here: the payload every task in a
suite sees is byte-identical, and the fingerprint changes whenever anything that
moves the number moves.
"""

from __future__ import annotations

import json
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

from agentdb.adapters import ExplainMode, RawPlan
from agentdb.adapters.base import EngineConnectionError
from agentdb.adapters.clickhouse import ClickHouseAdapter
from agentdb.adapters.clickhouse_client import (
    DEFAULT_PORT,
    DEFAULT_USER,
    ClickHouseTarget,
    build_client,
)
from agentdb.bench import GroundedContextProvider, build_provider, clickhouse_provider
from agentdb.config import Config
from agentdb.core import GroundingLevel
from tests.fakes import clickhouse_hits_fixture

QUESTION = "How many rows are in the hits table?"


# --------------------------------------------------------------------------
# the provider
# --------------------------------------------------------------------------


async def test_the_provider_serves_the_level_it_was_built_for() -> None:
    provider = build_provider(adapter=clickhouse_hits_fixture(), level="schema")

    payload = await provider.context(namespace="agentdb", question=QUESTION)

    assert "CREATE TABLE agentdb.hits" in payload
    assert "sort key" not in payload
    assert provider.name == "agentdb/schema"


async def test_the_layout_level_is_the_default_because_it_is_the_arm_under_test() -> None:
    provider = build_provider(adapter=clickhouse_hits_fixture())

    payload = await provider.context(namespace="agentdb", question=QUESTION)

    assert provider.level is GroundingLevel.LAYOUT
    assert "sort key (ORDER BY)" in payload


async def test_every_task_in_a_suite_sees_byte_identical_grounding() -> None:
    adapter = clickhouse_hits_fixture()
    provider = build_provider(adapter=adapter)

    first = await provider.context(namespace="agentdb", question="one question")
    second = await provider.context(namespace="agentdb", question="a different question")

    assert first == second
    assert len(adapter.calls_named("describe_relation")) == 1


async def test_a_second_namespace_is_built_rather_than_served_from_the_first() -> None:
    adapter = clickhouse_hits_fixture()
    provider = build_provider(adapter=adapter, level="schema")

    await provider.context(namespace="agentdb", question=QUESTION)
    await provider.context(namespace="other", question=QUESTION)

    assert adapter.calls_named("list_relations") == ["agentdb", "other"]


def test_an_unknown_level_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="not a valid GroundingLevel"):
        build_provider(adapter=clickhouse_hits_fixture(), level="everything")


def test_the_fingerprint_moves_when_the_level_moves() -> None:
    stats = build_provider(adapter=clickhouse_hits_fixture(), level="stats")
    layout = build_provider(adapter=clickhouse_hits_fixture(), level="layout")

    assert stats.fingerprint != layout.fingerprint


def test_the_fingerprint_moves_when_the_sampling_configuration_moves() -> None:
    baseline = build_provider(adapter=clickhouse_hits_fixture())
    sampled = build_provider(
        adapter=clickhouse_hits_fixture(), config=Config(default_sample_fraction=0.5)
    )

    assert baseline.fingerprint != sampled.fingerprint
    assert baseline.fingerprint == build_provider(adapter=clickhouse_hits_fixture()).fingerprint


def test_a_custom_name_survives_so_a_report_row_can_be_labelled() -> None:
    provider = build_provider(adapter=clickhouse_hits_fixture(), name="agentdb/A6_full")

    assert provider.name == "agentdb/A6_full"
    assert isinstance(provider, GroundedContextProvider)


# --------------------------------------------------------------------------
# connecting
# --------------------------------------------------------------------------


def test_the_target_reads_the_environment_and_defaults_to_the_read_only_account() -> None:
    target = ClickHouseTarget.from_env({})

    assert target.username == DEFAULT_USER
    assert target.port == DEFAULT_PORT

    configured = ClickHouseTarget.from_env(
        {
            "AGENTDB_CLICKHOUSE_HOST": "warehouse",
            "AGENTDB_CLICKHOUSE_PORT": "9000",
            "AGENTDB_CLICKHOUSE_USER": "reader",
            "AGENTDB_CLICKHOUSE_PASSWORD": "secret",
            "AGENTDB_CLICKHOUSE_DATABASE": "bench",
        }
    )
    assert configured == ClickHouseTarget(
        host="warehouse", port=9000, username="reader", password="secret", database="bench"
    )


def test_a_non_numeric_port_fails_before_any_connection_is_attempted() -> None:
    with pytest.raises(EngineConnectionError, match="must be a number"):
        ClickHouseTarget.from_env({"AGENTDB_CLICKHOUSE_PORT": "eight-one-two-three"})


async def test_a_missing_driver_says_which_extra_to_install() -> None:
    def importer(name: str) -> ModuleType:
        raise ImportError(name)

    with pytest.raises(EngineConnectionError) as caught:
        await build_client(ClickHouseTarget(), importer=importer)

    assert caught.value.suggestion is not None
    assert "--extra clickhouse" in caught.value.suggestion


async def test_an_unreachable_server_names_the_host_and_the_account() -> None:
    async def get_async_client(**_: Any) -> object:
        raise OSError("connection refused")

    module = SimpleNamespace(get_async_client=get_async_client)

    with pytest.raises(EngineConnectionError, match="cannot reach ClickHouse at localhost:58123"):
        await build_client(ClickHouseTarget(), importer=lambda _: cast(ModuleType, module))


async def test_the_provider_factory_connects_and_wraps_the_adapter() -> None:
    connected: dict[str, Any] = {}

    async def get_async_client(**kwargs: Any) -> object:
        connected.update(kwargs)
        return SimpleNamespace(query=None)

    module = SimpleNamespace(get_async_client=get_async_client)

    def importer(_: str) -> ModuleType:
        return cast(ModuleType, module)

    monkeypatched = ClickHouseTarget(host="warehouse", port=9000)
    client = await build_client(monkeypatched, importer=importer)
    provider = build_provider(adapter=ClickHouseAdapter(client=client), level="stats")

    assert connected["host"] == "warehouse"
    assert connected["port"] == 9000
    assert provider.level is GroundingLevel.STATS


async def test_the_dotted_path_factory_falls_back_to_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def get_async_client(**kwargs: Any) -> object:
        return SimpleNamespace(query=None, connected=kwargs)

    monkeypatch.setenv("AGENTDB_CLICKHOUSE_DATABASE", "from_env")

    provider = await clickhouse_provider(
        level="schema",
        name="agentdb/A0",
        importer=lambda _: cast(ModuleType, SimpleNamespace(get_async_client=get_async_client)),
    )

    assert provider.name == "agentdb/A0"
    assert provider.level is GroundingLevel.SCHEMA


async def test_explicit_connection_arguments_win_over_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    async def get_async_client(**kwargs: Any) -> object:
        seen.update(kwargs)
        return SimpleNamespace(query=None)

    monkeypatch.setenv("AGENTDB_CLICKHOUSE_HOST", "from_env")

    await clickhouse_provider(
        host="explicit",
        port=1234,
        username="reader",
        password="pw",
        database="bench",
        importer=lambda _: cast(ModuleType, SimpleNamespace(get_async_client=get_async_client)),
    )

    assert seen == {
        "host": "explicit",
        "port": 1234,
        "username": "reader",
        "password": "pw",
        "database": "bench",
    }


# --------------------------------------------------------------------------
# plan review (arm A3)
# --------------------------------------------------------------------------

PLAN_WITH_NO_PRUNING = {
    "Plan": {
        "Node Type": "ReadFromMergeTree",
        "Description": "agentdb.hits",
        "Indexes": [{"Type": "PrimaryKey", "Initial Granules": 1_000, "Selected Granules": 1_000}],
    }
}


WELL_PRUNED_PLAN = {
    "Plan": {
        "Node Type": "ReadFromMergeTree",
        "Description": "agentdb.hits",
        "Indexes": [{"Type": "PrimaryKey", "Initial Granules": 1_000, "Selected Granules": 10}],
    }
}


def _reviewing_provider(plan: object = PLAN_WITH_NO_PRUNING) -> GroundedContextProvider:
    adapter = clickhouse_hits_fixture()
    adapter.plan = RawPlan(
        engine="clickhouse",
        mode=ExplainMode.ESTIMATE,
        sql="",
        payload=json.dumps([plan]),
    )
    return build_provider(adapter=adapter, plan_review=True, name="agentdb/A3_plan")


async def test_a_reviewing_provider_reports_what_the_plan_would_do() -> None:
    provider = _reviewing_provider()

    review = await provider.explain_plan(
        sql="SELECT count() FROM hits WHERE SearchEngineID = 2", namespace="agentdb"
    )

    assert review is not None
    assert "SORT_KEY_UNUSED" in review
    assert "granules read after pruning" in review


async def test_a_plan_with_nothing_wrong_is_reported_as_nothing_at_all() -> None:
    provider = _reviewing_provider(WELL_PRUNED_PLAN)

    review = await provider.explain_plan(
        sql="SELECT count() FROM hits WHERE CounterID = 1 AND EventDate > '2013-07-01'",
        namespace="agentdb",
    )

    assert review is None


async def test_a_provider_without_plan_review_explains_nothing() -> None:
    provider = build_provider(adapter=clickhouse_hits_fixture())

    assert await provider.explain_plan(sql="SELECT 1", namespace="agentdb") is None


def test_plan_review_is_part_of_the_fingerprint() -> None:
    assert (
        _reviewing_provider().fingerprint
        != build_provider(adapter=clickhouse_hits_fixture(), name="agentdb/A3_plan").fingerprint
    )


async def test_the_dotted_path_factory_can_build_a_reviewing_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def get_async_client(**kwargs: Any) -> object:
        return SimpleNamespace(query=None)

    provider = await clickhouse_provider(
        plan_review=True,
        importer=lambda _: cast(ModuleType, SimpleNamespace(get_async_client=get_async_client)),
    )

    assert provider.explainer is not None


async def test_closing_a_provider_releases_the_connection_it_opened() -> None:
    """A matrix builds one provider per arm; nothing else can release its connection."""
    closed: list[bool] = []
    client = SimpleNamespace(query=None, close=lambda: _record(closed))
    provider = build_provider(adapter=ClickHouseAdapter(client=cast(Any, client)))

    await provider.aclose()

    assert closed == [True]


async def test_closing_a_provider_over_a_client_with_no_close_is_a_no_op() -> None:
    """The fake adapter in these tests has no connection to release, and must not care."""
    await build_provider(adapter=clickhouse_hits_fixture()).aclose()


async def _record(closed: list[bool]) -> None:
    closed.append(True)
