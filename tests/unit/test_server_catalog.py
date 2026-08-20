"""Dispatch: what the catalog does with a call, and with a call that fails.

The rule under test is the one that decides whether an agent can recover: an
expected failure — a bad argument, an engine that refuses — comes back as a
structured result it can read and act on, while a bug propagates instead of
being flattened into a plausible-looking error message.
"""

from __future__ import annotations

import json

import pytest

from agentdb.adapters import QueryPermissionError
from agentdb.server import ToolError, build_catalog
from agentdb.server.app import ToolResponse
from tests.fakes import FakeAdapter, clickhouse_hits_fixture
from tests.server_fakes import clickhouse_catalog


def test_the_catalog_advertises_the_spec_13_1_groups_in_order() -> None:
    catalog, _ = clickhouse_catalog()

    assert catalog.names == (
        "list_namespaces",
        "list_relations",
        "describe_relation",
        "physical_layout",
        "profile_columns",
        "grounded_context",
        "dialect_rules",
        "explain_plan",
        "explain_diff",
        "run_query",
        "mine_workload",
    )


def test_an_unknown_tool_names_the_tools_that_do_exist() -> None:
    catalog, _ = clickhouse_catalog()

    with pytest.raises(ToolError) as caught:
        catalog.get("advise_sort_key")

    assert "advise_sort_key" in caught.value.message
    assert "list_relations" in (caught.value.suggestion or "")


async def test_calling_an_unknown_tool_is_an_error_result_not_an_exception() -> None:
    catalog, _ = clickhouse_catalog()

    response = await catalog.call("nope", {})

    assert response.is_error
    assert response.structured["error"] == "no such tool: 'nope'"


async def test_a_bad_argument_comes_back_as_a_result_the_agent_can_act_on() -> None:
    catalog, _ = clickhouse_catalog()

    response = await catalog.call("describe_relation", {})

    assert response.is_error
    assert "'relation' is required" in str(response.structured["error"])


async def test_an_engine_refusal_comes_back_classified_rather_than_as_a_traceback() -> None:
    """The error class is what lets the benchmark bucket failures without regex."""
    adapter = clickhouse_hits_fixture()
    catalog = build_catalog(_refusing(adapter))

    response = await catalog.call("run_query", {"sql": "SELECT 1"})

    assert response.is_error
    assert response.structured["error_class"] == "permission"
    assert response.structured["suggestion"] == "ask for SELECT on agentdb"


async def test_an_unexpected_failure_is_left_to_propagate() -> None:
    """Swallowing a bug here would turn it into a quietly wrong answer."""
    adapter = clickhouse_hits_fixture()
    adapter.result = None
    catalog = build_catalog(adapter)

    with pytest.raises(AssertionError):
        await catalog.call("run_query", {"sql": "SELECT 1"})


def test_the_text_form_is_the_structured_form() -> None:
    """A client without schema support must not get a different answer."""
    response = ToolResponse(structured={"namespaces": ["agentdb"]})

    assert json.loads(response.text) == response.structured


def _refusing(adapter: FakeAdapter) -> FakeAdapter:
    """An adapter whose connection is not allowed to read."""

    async def execute(sql: str, limits: object) -> None:
        raise QueryPermissionError(
            "not authorized to read agentdb.hits",
            suggestion="ask for SELECT on agentdb",
        )

    adapter.execute = execute  # type: ignore[method-assign, assignment]
    return adapter
