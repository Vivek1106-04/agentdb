"""JSON Schema 2020-12 definitions for the tool surface (SPEC §13).

The 2026-07-28 MCP revision made ``inputSchema`` and ``outputSchema`` full JSON
Schema 2020-12, so this module writes real composed schemas with ``$defs`` and
``$ref`` rather than the flat ``{"type": "object"}`` blob most servers ship. The
shared definitions here are the same value objects the adapters exchange, which
is what lets a client rely on ``structuredContent`` instead of re-parsing prose.

Schemas are hand-written rather than derived from the dataclasses on purpose.
Derivation would guarantee agreement and describe nothing: the ``description``
on a field like ``stats_columns`` — filters on other columns skip no files — is
the fact worth transmitting, and no type annotation carries it. The contract
tests hold the two halves together instead.
"""

from __future__ import annotations

from typing import Final

type JsonValue = str | int | float | bool | list[JsonValue] | dict[str, JsonValue] | None
"""Anything that survives a round trip through JSON. Tool results are these."""

SCHEMA_DIALECT: Final = "https://json-schema.org/draft/2020-12/schema"

_REF_PREFIX: Final = "#/$defs/"


def _nullable(kind: str, description: str) -> dict[str, JsonValue]:
    """A field the engine may genuinely not know.

    ``null`` here always means "this engine does not report it", never zero —
    the distinction the adapters preserve (SPEC §6) and the one an agent must be
    able to see.
    """
    return {"type": [kind, "null"], "description": description}


def _array(items: dict[str, JsonValue], description: str) -> dict[str, JsonValue]:
    return {"type": "array", "items": items, "description": description}


def _ref(name: str) -> dict[str, JsonValue]:
    return {"$ref": f"{_REF_PREFIX}{name}"}


DEFS: Final[dict[str, dict[str, JsonValue]]] = {
    "relation_ref": {
        "type": "object",
        "description": "A relation, always fully qualified so it round-trips.",
        "properties": {
            "catalog": _nullable("string", "Unity Catalog name; null on ClickHouse."),
            "namespace": {"type": "string", "description": "Database or schema."},
            "name": {"type": "string"},
            "fqn": {
                "type": "string",
                "description": "The fully-qualified name to write into SQL for this engine.",
            },
        },
        "required": ["catalog", "namespace", "name", "fqn"],
        "additionalProperties": False,
    },
    "relation": {
        "type": "object",
        "description": "A listed relation with the size facts the catalog already holds.",
        "properties": {
            "ref": _ref("relation_ref"),
            "kind": {"enum": ["table", "view", "materialized_view", "foreign_table"]},
            "engine_type": _nullable("string", "Table engine or file format."),
            "approx_rows": _nullable("integer", "Estimate from the engine's own catalog."),
            "on_disk_bytes": _nullable("integer", "Compressed footprint."),
            "comment": _nullable("string", ""),
        },
        "required": ["ref", "kind", "engine_type", "approx_rows", "on_disk_bytes", "comment"],
        "additionalProperties": False,
    },
    "column_def": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "data_type": {"type": "string"},
            "is_nullable": {"type": "boolean"},
            "ordinal": {
                "type": "integer",
                "description": (
                    "1-based position in schema order. Not cosmetic on Databricks: "
                    "Delta collects data-skipping statistics only for the leading "
                    "columns, so ordinal decides whether a filter can skip files."
                ),
            },
            "default_expression": _nullable("string", ""),
            "comment": _nullable("string", ""),
        },
        "required": ["name", "data_type", "is_nullable", "ordinal"],
        "additionalProperties": False,
    },
    "relation_detail": {
        "type": "object",
        "properties": {
            "ref": _ref("relation_ref"),
            "columns": _array(_ref("column_def"), "Columns in schema order."),
            "create_statement": {"type": "string", "description": "The engine's own DDL."},
            "comment": _nullable("string", ""),
        },
        "required": ["ref", "columns", "create_statement", "comment"],
        "additionalProperties": False,
    },
    "skip_index": {
        "type": "object",
        "description": "A ClickHouse data-skipping index.",
        "properties": {
            "name": {"type": "string"},
            "index_type": {"type": "string"},
            "expression": {"type": "string"},
            "granularity": {"type": "integer"},
            "compressed_bytes": _nullable("integer", ""),
        },
        "required": ["name", "index_type", "expression", "granularity"],
        "additionalProperties": False,
    },
    "projection": {
        "type": "object",
        "properties": {"name": {"type": "string"}, "query": {"type": "string"}},
        "required": ["name", "query"],
        "additionalProperties": False,
    },
    "physical_layout": {
        "type": "object",
        "description": (
            "How the relation is laid out on storage. The facts that decide "
            "whether a filter reads a percent of the data or all of it, and the "
            "ones no CREATE TABLE dump states."
        ),
        "properties": {
            "engine": {"enum": ["clickhouse", "databricks"]},
            "ref": _ref("relation_ref"),
            "create_statement": {"type": "string"},
            "table_engine": _nullable("string", "ClickHouse table engine."),
            "order_by": {
                "type": ["array", "null"],
                "items": {"type": "string"},
                "description": (
                    "ClickHouse sort key. Filters prune granules through this key "
                    "left to right; a filter that skips the leading column prunes nothing."
                ),
            },
            "partition_by": {"type": ["array", "null"], "items": {"type": "string"}},
            "primary_key": {"type": ["array", "null"], "items": {"type": "string"}},
            "sampling_key": _nullable("string", "Required before SAMPLE can be used."),
            "skip_indexes": _array(_ref("skip_index"), ""),
            "projections": _array(_ref("projection"), ""),
            "ttl": _nullable("string", ""),
            "table_format": _nullable("string", "Databricks: delta, parquet, …"),
            "clustering_columns": {
                "type": ["array", "null"],
                "items": {"type": "string"},
                "description": "Databricks liquid clustering key; prunes files.",
            },
            "zorder_columns": {"type": ["array", "null"], "items": {"type": "string"}},
            "is_managed": _nullable("boolean", ""),
            "deletion_vectors_enabled": _nullable("boolean", ""),
            "num_files": _nullable("integer", ""),
            "avg_file_bytes": _nullable("number", ""),
            "stats_indexed_columns": _nullable(
                "integer",
                "Delta indexes this many leading columns; filters on later columns "
                "skip no files at all.",
            ),
            "stats_columns": {
                "type": ["array", "null"],
                "items": {"type": "string"},
                "description": (
                    "Explicit data-skipping statistics set, overriding the ordinal "
                    "rule. Filters on columns outside it cannot prune."
                ),
            },
            "approx_rows": _nullable("integer", ""),
            "on_disk_bytes": _nullable("integer", ""),
            "compression_ratio": _nullable("number", ""),
        },
        "required": ["engine", "ref", "create_statement"],
        "additionalProperties": False,
    },
    "column_profile": {
        "type": "object",
        "description": "Sampled distribution facts. Every figure is labelled with how it was got.",
        "properties": {
            "name": {"type": "string"},
            "data_type": {"type": "string"},
            "sample_method": {
                "enum": ["full", "sample", "system_table", "unavailable"],
                "description": "Anything but 'full' is an estimate. Weigh it accordingly.",
            },
            "sampled_rows": {"type": "integer", "minimum": 0},
            "approx_distinct": _nullable("integer", ""),
            "null_ratio": {
                "type": ["number", "null"],
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "",
            },
            "min_value": _nullable("string", ""),
            "max_value": _nullable("string", ""),
            "top_values": {
                "type": "array",
                "maxItems": 10,
                "items": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                        "count": {"type": "integer"},
                    },
                    "required": ["value", "count"],
                    "additionalProperties": False,
                },
                "description": "Most frequent values in the sample, descending.",
            },
            "avg_bytes": _nullable("number", ""),
        },
        "required": ["name", "data_type", "sample_method", "sampled_rows", "top_values"],
        "additionalProperties": False,
    },
    "dialect_rules": {
        "type": "object",
        "properties": {
            "engine": {"enum": ["clickhouse", "databricks"]},
            "version": {"type": "string", "description": "The connected engine's version."},
            "identifier_quote": {"type": "string"},
            "string_quote": {"type": "string"},
            "supports_ilike": {"type": "boolean"},
            "reserved_words": _array({"type": "string"}, "Sorted; quote these as identifiers."),
            "quirks": _array({"type": "string"}, "Short, actionable engine-specific notes."),
        },
        "required": [
            "engine",
            "version",
            "identifier_quote",
            "string_quote",
            "supports_ilike",
            "reserved_words",
            "quirks",
        ],
        "additionalProperties": False,
    },
    "plan_warning": {
        "type": "object",
        "description": "One actionable finding about a plan. This is the product.",
        "properties": {
            "code": {"type": "string"},
            "severity": {"enum": ["info", "warning", "critical"]},
            "human_message": {"type": "string"},
            "relation": _nullable("string", ""),
            "columns": _array({"type": "string"}, ""),
            "suggested_rewrite": _nullable(
                "string", "Present only where the rewrite follows mechanically."
            ),
        },
        "required": ["code", "severity", "human_message", "relation", "columns"],
        "additionalProperties": False,
    },
    "plan_node": {
        "type": "object",
        "description": "One normalized plan node; children nest to the full tree.",
        "properties": {
            "op": {
                "enum": [
                    "scan",
                    "filter",
                    "aggregate",
                    "join",
                    "sort",
                    "limit",
                    "exchange",
                    "projection_read",
                    "other",
                ]
            },
            "node_type": {"type": "string", "description": "The engine's own node name."},
            "relation": _nullable("string", ""),
            "estimated_rows": _nullable("integer", ""),
            "actual_rows": _nullable(
                "integer", "Always null here: neither engine's EXPLAIN executes."
            ),
            "estimated_cost": _nullable("number", ""),
            "filters": _array({"type": "string"}, ""),
            "granules_total": _nullable("integer", ""),
            "granules_selected": _nullable("integer", ""),
            "parts_total": _nullable("integer", ""),
            "parts_selected": _nullable("integer", ""),
            "index_used": _array({"type": "string"}, "Indexes that actually fired."),
            "projection_used": _nullable("string", ""),
            "files_total": _nullable("integer", ""),
            "files_selected": _nullable("integer", ""),
            "partitions_total": _nullable("integer", ""),
            "partitions_selected": _nullable("integer", ""),
            "partition_filters": _array({"type": "string"}, ""),
            "pushed_filters": _array(
                {"type": "string"}, "Answerable from file statistics; these skip files."
            ),
            "data_filters": _array(
                {"type": "string"},
                "Evaluated row by row. A predicate here and not in pushed_filters "
                "cannot skip a single file.",
            ),
            "photon": _nullable("boolean", ""),
            "join_strategy": _nullable("string", ""),
            "pruning_ratio": _nullable("number", "Units kept over units considered. Low is good."),
            "children": _array(_ref("plan_node"), ""),
        },
        "required": ["op", "node_type", "relation", "children"],
        "additionalProperties": False,
    },
    "plan_summary": {
        "type": "object",
        "properties": {
            "engine": {"type": "string"},
            "sql": {"type": "string"},
            "root": _ref("plan_node"),
            "pruning_ratio": _nullable(
                "number", "Selected over considered across every scan. Low is good."
            ),
            "pruning_unit": {
                "enum": ["granule", "file", None],
                "description": "Granules on ClickHouse, files on Databricks. Never compare across.",
            },
            "pruning_source": {
                "enum": ["estimated", "measured", None],
                "description": (
                    "Databricks plans state no file counts, so a Databricks ratio "
                    "is only ever measured, and only after the query has run."
                ),
            },
            "full_scan_relations": _array({"type": "string"}, ""),
            "estimated_bytes_read": _nullable("integer", ""),
            "measured_bytes_read": _nullable("integer", "Null until the query has run."),
            "bytes_ratio": _nullable("number", ""),
            "photon_coverage": _nullable("number", "Databricks only."),
            "warnings": _array(_ref("plan_warning"), ""),
            "rendered": {"type": "string", "description": "The summary as short text."},
        },
        "required": ["engine", "sql", "root", "warnings", "rendered"],
        "additionalProperties": False,
    },
    "result_set": {
        "type": "object",
        "properties": {
            "columns": _array({"type": "string"}, ""),
            "rows": _array(
                {"type": "array", "items": True},
                "Row values in column order; JSON scalars, or strings where the "
                "engine returned a type JSON has no room for.",
            ),
            "row_count": {"type": "integer"},
            "truncated": {
                "type": "boolean",
                "description": "True when the result hit max_result_rows and was cut.",
            },
            "duration_ms": _nullable("integer", ""),
            "rows_read": _nullable("integer", ""),
            "bytes_read": _nullable("integer", ""),
            "query_id": _nullable("string", "The engine-side id this execution was tagged with."),
        },
        "required": ["columns", "rows", "row_count", "truncated"],
        "additionalProperties": False,
    },
    "workload_entry": {
        "type": "object",
        "properties": {
            "normalized_sql": {"type": "string", "description": "Literals replaced by holes."},
            "calls": {"type": "integer"},
            "relations": _array({"type": "string"}, ""),
            "total_duration_ms": _nullable("number", ""),
            "mean_duration_ms": _nullable("number", ""),
            "rows_read": _nullable("integer", ""),
            "bytes_read": _nullable("integer", ""),
            "sample_sql": _nullable("string", "One concrete instance of the shape."),
        },
        "required": ["normalized_sql", "calls", "relations"],
        "additionalProperties": False,
    },
    "relation_context": {
        "type": "object",
        "description": "What the assembled context knows about one relation.",
        "properties": {
            "detail": _ref("relation_detail"),
            "layout": {
                "oneOf": [_ref("physical_layout"), {"type": "null"}],
                "description": "Present from grounding level 'layout' upwards.",
            },
            "profiles": _array(_ref("column_profile"), ""),
            "profiled_columns_available": {
                "type": "integer",
                "description": (
                    "How many columns could have been profiled. Larger than the "
                    "number profiled means the profile is partial, not complete."
                ),
            },
        },
        "required": ["detail", "layout", "profiles", "profiled_columns_available"],
        "additionalProperties": False,
    },
    "exemplar": {
        "type": "object",
        "description": (
            "A remembered question/SQL/outcome triple, bound to a schema version "
            "and carried on two time axes (SPEC §10). Valid time says when it was "
            "true of the schema; transaction time says when agentdb learned it."
        ),
        "properties": {
            "id": {"type": "integer"},
            "engine": {"enum": ["clickhouse", "databricks"]},
            "namespace": {"type": "string"},
            "question": {"type": "string"},
            "sql": {"type": "string"},
            "normalized_sql": {
                "type": "string",
                "description": "Literals parameterized. Two queries differing only in "
                "their constants share this key.",
            },
            "relations": _array({"type": "string"}, "Relations the query names."),
            "columns": _array({"type": "string"}, "Columns re-validated on a schema change."),
            "schema_version_id": {
                "type": "integer",
                "description": "The schema version this was written against.",
            },
            "outcome": {"enum": ["success", "error", "rejected"]},
            "provenance": {"enum": ["agent", "workload_mined", "curated"]},
            "valid_from": {"type": "string", "format": "date-time"},
            "valid_to": _nullable(
                "string",
                "When the schema stopped supporting it; null while it still holds.",
            ),
            "tx_from": {"type": "string", "format": "date-time"},
            "tx_to": _nullable(
                "string", "When a correction superseded this record; null while current."
            ),
            "rows_returned": _nullable("integer", ""),
            "bytes_read": _nullable("integer", ""),
            "duration_ms": _nullable("integer", ""),
            "error_class": _nullable(
                "string", "syntax | semantic | plan_rejection | timeout | permission."
            ),
            "error_text": _nullable("string", ""),
        },
        "required": [
            "id",
            "engine",
            "namespace",
            "question",
            "sql",
            "normalized_sql",
            "relations",
            "columns",
            "schema_version_id",
            "outcome",
            "provenance",
            "valid_from",
            "valid_to",
            "tx_from",
            "tx_to",
            "rows_returned",
            "bytes_read",
            "duration_ms",
            "error_class",
            "error_text",
        ],
        "additionalProperties": False,
    },
    "scored_exemplar": {
        "type": "object",
        "description": "One retrieved exemplar with the ranking that selected it.",
        "properties": {
            "exemplar": _ref("exemplar"),
            "score": {"type": "number", "description": "The weighted total."},
            "components": {
                "type": "object",
                "description": (
                    "Each term before weighting. Reported because every weight in "
                    "the ranking is an ablation arm (SPEC §10.4)."
                ),
                "properties": {
                    "sem": {"type": "number", "description": "Cosine against the question."},
                    "rel": {"type": "number", "description": "Relation-set overlap."},
                    "success": {"type": "number", "description": "1 when the query worked."},
                    "recency": {"type": "number", "description": "Decay on when it was learned."},
                    "cost": {"type": "number", "description": "Bytes read, relative to the pool."},
                },
                "required": ["sem", "rel", "success", "recency", "cost"],
                "additionalProperties": False,
            },
        },
        "required": ["exemplar", "score", "components"],
        "additionalProperties": False,
    },
    "exemplar_revision": {
        "type": "object",
        "description": "One revision of a remembered query, for the bi-temporal history.",
        "properties": {
            "exemplar": _ref("exemplar"),
            "fingerprint": {
                "type": "string",
                "description": "Schema fingerprint this revision was written against; "
                "empty when that version is no longer on record.",
            },
            "reason": _nullable(
                "string",
                "What changed underneath it, recomputed against the schema that "
                "superseded it. Null while it is still valid, and null when no "
                "stored version covers the moment it was invalidated.",
            ),
        },
        "required": ["exemplar", "fingerprint", "reason"],
        "additionalProperties": False,
    },
}
"""Shared definitions. A tool's schema references these; it never restates them."""


def schema(body: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Complete ``body`` into a standalone schema carrying the ``$defs`` it uses.

    Only the reachable definitions are attached. Shipping the whole registry on
    every tool would be simpler and would put a few kilobytes of irrelevant
    schema into an agent's context on each ``tools/list`` — the same token
    carelessness the grounding payload is measured for avoiding.
    """
    reachable = _reachable(body)
    completed: dict[str, JsonValue] = {"$schema": SCHEMA_DIALECT, **body}
    if reachable:
        completed["$defs"] = {name: DEFS[name] for name in sorted(reachable)}
    return completed


def object_schema(
    properties: dict[str, JsonValue],
    *,
    required: list[str],
    description: str = "",
) -> dict[str, JsonValue]:
    """The common case: a closed object with named properties."""
    names: list[JsonValue] = list(required)
    body: dict[str, JsonValue] = {
        "type": "object",
        "properties": properties,
        "required": names,
        "additionalProperties": False,
    }
    if description:
        body["description"] = description
    return schema(body)


def ref(name: str) -> dict[str, JsonValue]:
    """A reference to a shared definition, for use inside a tool schema."""
    return _ref(name)


def definition_schema(name: str) -> dict[str, JsonValue]:
    """A shared definition as a standalone schema, for a tool that returns it flat.

    A tool whose whole result *is* one shared type returns it at the top level
    rather than wrapped in a single-key object: the wrapper would buy nothing and
    cost every caller an indirection.
    """
    return schema(DEFS[name])


def array_of(name: str, description: str) -> dict[str, JsonValue]:
    """An array of one shared definition."""
    return _array(_ref(name), description)


def _reachable(body: JsonValue) -> set[str]:
    """Definition names reachable from ``body``, following refs transitively."""
    found: set[str] = set()
    pending = _direct_refs(body)
    while pending:
        name = pending.pop()
        if name in found:
            continue
        found.add(name)
        pending |= _direct_refs(DEFS[name])
    return found


def _direct_refs(node: JsonValue) -> set[str]:
    if isinstance(node, dict):
        target = node.get("$ref")
        refs = {target[len(_REF_PREFIX) :]} if isinstance(target, str) else set()
        for value in node.values():
            refs |= _direct_refs(value)
        return refs
    if isinstance(node, list):
        return {name for item in node for name in _direct_refs(item)}
    return set()
