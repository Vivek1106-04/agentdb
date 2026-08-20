"""Every tool's declared schema, checked against what the tool actually returns.

SPEC §13.1 says every tool declares a full ``outputSchema`` and is covered by a
contract test. This is that test. It matters more than it looks: an
``outputSchema`` is a promise to an agent that it may read a field without
checking, and a promise nobody verifies is worse than no promise, because the
agent stops checking either way.

Both engines are exercised, because the two adapters populate different halves
of the same shapes — ClickHouse fills sort keys and granules, Databricks fills
clustering columns and files — and a schema that only ever saw one of them has
only ever been half tested.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest
from jsonschema import Draft202012Validator

from agentdb.server import ToolCatalog
from agentdb.server.schemas import SCHEMA_DIALECT, JsonValue
from tests.server_fakes import clickhouse_catalog, databricks_catalog

CLICKHOUSE_ARGS: dict[str, Mapping[str, JsonValue]] = {
    "list_namespaces": {},
    "list_relations": {"namespace": "agentdb"},
    "describe_relation": {"relation": "agentdb.hits"},
    "physical_layout": {"relation": "agentdb.hits"},
    "profile_columns": {"relation": "agentdb.hits", "columns": ["SearchEngineID", "UserID"]},
    "grounded_context": {"namespace": "agentdb", "question": "how many hits per counter?"},
    "dialect_rules": {},
    "explain_plan": {
        "sql": "SELECT count() FROM hits WHERE UserID = 42",
        "namespace": "agentdb",
    },
    "explain_diff": {
        "candidates": [
            "SELECT count() FROM hits WHERE UserID = 42",
            "SELECT count() FROM hits WHERE CounterID = 62",
        ],
        "namespace": "agentdb",
    },
    "run_query": {"sql": "SELECT CounterID, count() AS hits FROM hits GROUP BY CounterID"},
    "mine_workload": {"hours": 24, "top_n": 5},
}

DATABRICKS_ARGS: dict[str, Mapping[str, JsonValue]] = {
    "list_namespaces": {},
    "list_relations": {"namespace": "samples.tpch"},
    "describe_relation": {"relation": "samples.tpch.lineitem"},
    "physical_layout": {"relation": "samples.tpch.lineitem"},
    "profile_columns": {"relation": "samples.tpch.lineitem", "columns": ["l_shipdate"]},
    "grounded_context": {"namespace": "samples.tpch", "level": "layout"},
    "dialect_rules": {},
    "explain_plan": {
        "sql": "SELECT count(*) FROM samples.tpch.lineitem WHERE l_shipdate > '1998-01-01'",
        "namespace": "samples.tpch",
    },
    "explain_diff": {
        "candidates": [
            "SELECT count(*) FROM samples.tpch.lineitem WHERE l_shipdate > '1998-01-01'",
            "SELECT count(*) FROM samples.tpch.lineitem WHERE l_audit_note = 'x'",
        ],
        "namespace": "samples.tpch",
    },
    "run_query": {"sql": "SELECT count(*) FROM samples.tpch.lineitem"},
    "mine_workload": {},
}

CATALOGS: dict[str, tuple[ToolCatalog, dict[str, Mapping[str, JsonValue]]]] = {
    "clickhouse": (clickhouse_catalog()[0], CLICKHOUSE_ARGS),
    "databricks": (databricks_catalog()[0], DATABRICKS_ARGS),
}

TOOL_IDS = [(engine, name) for engine, (catalog, _) in CATALOGS.items() for name in catalog.names]


@pytest.mark.parametrize(("engine", "name"), TOOL_IDS)
def test_both_schemas_are_valid_json_schema_2020_12(engine: str, name: str) -> None:
    catalog, _ = CATALOGS[engine]
    tool = catalog.get(name)

    for schema in (tool.input_schema, tool.output_schema):
        assert schema["$schema"] == SCHEMA_DIALECT
        Draft202012Validator.check_schema(schema)


@pytest.mark.parametrize(("engine", "name"), TOOL_IDS)
def test_input_schema_is_an_object_root_with_documented_properties(engine: str, name: str) -> None:
    """MCP still requires an object root, and an undescribed argument is a guess."""
    catalog, _ = CATALOGS[engine]
    schema = catalog.get(name).input_schema

    assert schema["type"] == "object"
    properties = schema["properties"]
    assert isinstance(properties, dict)
    for argument, definition in properties.items():
        assert isinstance(definition, dict), argument
        assert definition.get("description"), f"{name}.{argument} has no description"


@pytest.mark.parametrize(("engine", "name"), TOOL_IDS)
async def test_response_conforms_to_the_declared_output_schema(engine: str, name: str) -> None:
    catalog, arguments = CATALOGS[engine]

    response = await catalog.call(name, arguments[name])

    assert not response.is_error, response.structured
    Draft202012Validator(catalog.get(name).output_schema).validate(response.structured)


@pytest.mark.parametrize("engine", list(CATALOGS))
def test_every_tool_is_covered_by_this_file(engine: str) -> None:
    """A tool added without arguments here would otherwise skip the contract silently."""
    catalog, arguments = CATALOGS[engine]

    assert set(catalog.names) == set(arguments)


def test_the_two_engines_serve_the_same_catalog() -> None:
    """Engine differences belong in the responses, never in which tools exist.

    A client that had to ask which engine it was talking to before knowing which
    tools it had would make the capability flags decorative.
    """
    clickhouse, _ = CATALOGS["clickhouse"]
    databricks, _ = CATALOGS["databricks"]

    assert clickhouse.names == databricks.names
