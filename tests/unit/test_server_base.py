"""Argument handling and name resolution at the tool boundary.

The interesting assertions here are all refusals. A server that guesses at a
missing argument, or resolves a two-part Unity Catalog name against session
state it does not have, answers a question the agent did not ask — and the agent
has no way to notice it happened.
"""

from __future__ import annotations

import pytest

from agentdb.config import Config
from agentdb.server.base import (
    ServerContext,
    ToolError,
    optional_int,
    optional_str,
    require_str,
    require_str_list,
)
from tests.fakes import clickhouse_hits_fixture, databricks_tpch_fixture


def _clickhouse() -> ServerContext:
    return ServerContext(adapter=clickhouse_hits_fixture(), config=Config())


def _databricks() -> ServerContext:
    return ServerContext(adapter=databricks_tpch_fixture())


def test_a_tool_error_carries_its_suggestion_into_the_response() -> None:
    error = ToolError("no", suggestion="try yes")

    assert error.as_dict() == {"error": "no", "suggestion": "try yes"}


def test_a_two_part_name_resolves_on_clickhouse() -> None:
    ref = _clickhouse().parse_relation("agentdb.hits")

    assert (ref.catalog, ref.namespace, ref.name) == (None, "agentdb", "hits")


def test_a_three_part_name_resolves_on_databricks() -> None:
    ref = _databricks().parse_relation("samples.tpch.lineitem")

    assert (ref.catalog, ref.namespace, ref.name) == ("samples", "tpch", "lineitem")


def test_a_two_part_name_is_refused_on_databricks_rather_than_guessed() -> None:
    """Unity Catalog would resolve it against USE state the server does not have."""
    with pytest.raises(ToolError) as caught:
        _databricks().parse_relation("tpch.lineitem")

    assert caught.value.suggestion == "write it as catalog.schema.table"


def test_a_bare_name_is_refused_on_clickhouse() -> None:
    with pytest.raises(ToolError) as caught:
        _clickhouse().parse_relation("hits")

    assert caught.value.suggestion == "write it as database.table"


def test_empty_name_parts_do_not_pad_the_name_out_to_a_valid_length() -> None:
    with pytest.raises(ToolError):
        _clickhouse().parse_relation("agentdb..hits.")


def test_the_context_builds_a_builder_and_an_explainer_over_its_own_adapter() -> None:
    context = _clickhouse()

    assert context.builder.adapter is context.adapter
    assert context.explainer.config is context.config


@pytest.mark.parametrize("value", [None, "", 3])
def test_a_required_string_refuses_anything_else(value: object) -> None:
    with pytest.raises(ToolError, match="'relation' is required"):
        require_str({"relation": value}, "relation")  # type: ignore[dict-item]


def test_an_optional_string_passes_none_through() -> None:
    assert optional_str({}, "namespace") is None


def test_an_optional_string_still_refuses_a_wrong_type() -> None:
    with pytest.raises(ToolError, match="must be a string"):
        optional_str({"namespace": 7}, "namespace")


def test_an_optional_integer_passes_none_through() -> None:
    assert optional_int({}, "top_n") is None


@pytest.mark.parametrize("value", [0, -1, "5", True])
def test_an_optional_integer_refuses_a_non_positive_or_non_integer(value: object) -> None:
    """``True`` is an ``int`` in Python and is not an argument anybody meant to pass."""
    with pytest.raises(ToolError, match="must be an integer"):
        optional_int({"top_n": value}, "top_n")  # type: ignore[dict-item]


@pytest.mark.parametrize("value", [None, [], "a"])
def test_a_required_string_list_refuses_anything_else(value: object) -> None:
    with pytest.raises(ToolError, match="non-empty array"):
        require_str_list({"columns": value}, "columns")  # type: ignore[dict-item]


def test_a_required_string_list_refuses_a_list_of_other_things() -> None:
    with pytest.raises(ToolError, match="only strings"):
        require_str_list({"columns": ["a", 2]}, "columns")
