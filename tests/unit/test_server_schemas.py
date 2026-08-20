"""The schema builder: composition, and what it declines to ship.

Only the definitions a schema actually reaches are attached to it. That is not
tidiness — ``tools/list`` is read by a model on every session, and shipping the
whole registry with every tool would put kilobytes of irrelevant schema into the
context of the very agents this project measures for token discipline.
"""

from __future__ import annotations

from agentdb.server.schemas import (
    DEFS,
    SCHEMA_DIALECT,
    array_of,
    definition_schema,
    object_schema,
    ref,
    schema,
)


def test_a_schema_carries_the_2020_12_dialect() -> None:
    assert schema({"type": "object"})["$schema"] == SCHEMA_DIALECT


def test_only_the_reachable_definitions_are_attached() -> None:
    built = object_schema({"rules": ref("dialect_rules")}, required=["rules"])

    assert set(_defs(built)) == {"dialect_rules"}


def test_definitions_are_followed_transitively() -> None:
    """``relation`` refers to ``relation_ref``; a client resolving one needs both."""
    built = object_schema({"relations": array_of("relation", "")}, required=["relations"])

    assert set(_defs(built)) == {"relation", "relation_ref"}


def test_a_self_referential_definition_terminates() -> None:
    """``plan_node`` contains ``plan_node``: the tree is the point of the IR."""
    built = definition_schema("plan_summary")

    assert "plan_node" in _defs(built)


def test_a_schema_reaching_nothing_carries_no_defs_block() -> None:
    assert "$defs" not in schema({"type": "object"})


def test_refs_are_found_inside_arrays_as_well_as_objects() -> None:
    built = schema({"oneOf": [ref("projection"), {"type": "null"}]})

    assert set(_defs(built)) == {"projection"}


def test_an_object_schema_is_closed_and_keeps_its_description() -> None:
    """Open objects let a client believe in a field the server never promised."""
    built = object_schema({"a": {"type": "string"}}, required=["a"], description="thing")

    assert built["additionalProperties"] is False
    assert built["description"] == "thing"


def test_an_object_schema_without_a_description_omits_the_key() -> None:
    assert "description" not in object_schema({}, required=[])


def test_every_shared_definition_is_a_closed_object() -> None:
    for name, definition in DEFS.items():
        assert definition["additionalProperties"] is False, name


def _defs(built: object) -> dict[str, object]:
    assert isinstance(built, dict)
    block = built["$defs"]
    assert isinstance(block, dict)
    return block
