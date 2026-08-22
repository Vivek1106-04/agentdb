"""Memory tools: what this connection has already learned (SPEC §13.1).

Three tools over the bi-temporal store of SPEC §10, and they only exist when a
store is configured. A build with no Postgres behind it does not advertise them:
a tool that lists itself and then explains it has no backing store costs an agent
a turn and a schema read to learn nothing.

The store is synchronous — plain DB-API through psycopg — so every call crosses
into a worker thread. A blocking round trip on the event loop would stall an
adapter query running beside it, and the store is on the hot path of every
grounded answer.

Two design choices worth stating, because they shape what the memory arms
measure:

* **Positives and negatives come back separately.** SPEC §10.4 serves previously
  failing queries as explicit "do not do this" context, and an agent that had to
  infer which half was which from an ``outcome`` field would eventually paste a
  failure back in as an example.
* **The relations and columns are derived from the SQL, not asked for.**
  Re-validation is only as good as the names recorded beside the query, and an
  agent that had to list them would list them wrong exactly when it mattered.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence

from agentdb.core.memory import ExemplarDraft, Outcome, Provenance, normalize_sql
from agentdb.core.memory.store import ExemplarStore
from agentdb.core.query_shape import analyze
from agentdb.server import serialize
from agentdb.server.base import (
    ServerContext,
    ToolDef,
    ToolError,
    optional_int,
    optional_str,
    require_str,
)
from agentdb.server.schemas import JsonValue, array_of, definition_schema, object_schema

OUTCOMES = tuple(outcome.value for outcome in Outcome)
PROVENANCES = tuple(provenance.value for provenance in Provenance)


def memory_tools(context: ServerContext) -> tuple[ToolDef, ...]:
    """Build the memory group, or nothing when no store is configured."""
    store = context.store
    if store is None:
        return ()
    return (
        _retrieve_exemplars(context, store),
        _record_outcome(context, store),
        _explain_exemplar_history(context, store),
    )


def _retrieve_exemplars(context: ServerContext, store: ExemplarStore) -> ToolDef:
    async def handler(args: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        question = require_str(args, "question")
        namespace = require_str(args, "namespace")
        relations = _optional_strings(args, "relations")
        k = optional_int(args, "k")

        positive = await asyncio.to_thread(
            store.retrieve,
            question,
            engine=context.adapter.engine,
            namespace=namespace,
            relations=relations,
            k=k,
        )
        negative = await asyncio.to_thread(
            store.retrieve,
            question,
            engine=context.adapter.engine,
            namespace=namespace,
            relations=relations,
            k=k,
            failures_only=True,
        )
        return {
            "namespace": namespace,
            "question": question,
            "positive": [serialize.scored_exemplar(item) for item in positive],
            "negative": [serialize.scored_exemplar(item) for item in negative],
        }

    return ToolDef(
        name="retrieve_exemplars",
        title="Retrieve exemplars",
        description=(
            "Hybrid-ranked queries this connection has already run against this "
            "namespace: ones that worked, and — separately — ones that failed, "
            "with the error class that killed them. Only exemplars still true of "
            "the current schema are returned; a query invalidated by a column "
            "rename or a layout change is withheld rather than ranked down."
        ),
        input_schema=object_schema(
            {
                "question": {
                    "type": "string",
                    "description": "The question being answered now. Ranked against by meaning.",
                },
                "namespace": {
                    "type": "string",
                    "description": "Database or schema whose memory to search.",
                },
                "relations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Relations the answer is expected to touch. Overlap with an "
                        "exemplar's own relations is a ranking term; omitting it "
                        "leaves the semantic term to carry retrieval."
                    ),
                },
                "k": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Exemplars per half. Defaults to the configured budget.",
                },
            },
            required=["question", "namespace"],
        ),
        output_schema=object_schema(
            {
                "namespace": {"type": "string"},
                "question": {"type": "string"},
                "positive": array_of("scored_exemplar", "Queries that worked. Best first."),
                "negative": array_of(
                    "scored_exemplar",
                    "Queries that failed here before. Context for what not to write.",
                ),
            },
            required=["namespace", "question", "positive", "negative"],
        ),
        handler=handler,
    )


def _record_outcome(context: ServerContext, store: ExemplarStore) -> ToolDef:
    async def handler(args: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        sql = require_str(args, "sql")
        engine = context.adapter.engine
        shape = analyze(sql, engine)
        if not shape.tables:
            raise ToolError(
                "the SQL names no relation this server could parse",
                suggestion="write the query with its table named explicitly in a FROM clause",
            )
        outcome = _outcome(args)
        draft = ExemplarDraft(
            engine=engine,
            namespace=require_str(args, "namespace"),
            question=require_str(args, "question"),
            sql=sql,
            normalized_sql=normalize_sql(sql, engine),
            relations=shape.tables,
            columns=_columns(shape.filter_columns, shape.group_by_columns),
            outcome=outcome,
            provenance=_provenance(args),
            rows_returned=optional_int(args, "rows_returned", minimum=0),
            bytes_read=optional_int(args, "bytes_read", minimum=0),
            duration_ms=optional_int(args, "duration_ms", minimum=0),
            error_class=_error_class(args, outcome),
            error_text=optional_str(args, "error_text"),
        )
        recorded = await asyncio.to_thread(store.record, draft)
        return serialize.exemplar(recorded)

    return ToolDef(
        name="record_outcome",
        title="Record outcome",
        description=(
            "Remember a question, the SQL that answered it, and how that turned "
            "out. run_query records its own executions; this is the entry point "
            "for an agent outside that path. The relations and columns to "
            "re-validate on a schema change are parsed from the SQL, not taken "
            "on trust, and a failure is worth recording precisely because it "
            "becomes a negative exemplar."
        ),
        input_schema=object_schema(
            {
                "namespace": {"type": "string", "description": "Database or schema queried."},
                "question": {
                    "type": "string",
                    "description": "The natural-language question this SQL answered.",
                },
                "sql": {"type": "string", "description": "The query as it was issued."},
                "outcome": {
                    "enum": list(OUTCOMES),
                    "description": (
                        "'rejected' is agentdb declining to run a plan, which is not "
                        "the same as the engine refusing the query."
                    ),
                },
                "provenance": {
                    "enum": list(PROVENANCES),
                    "description": "Who wrote it. Defaults to 'agent'.",
                },
                "rows_returned": {"type": "integer", "minimum": 0, "description": "Rows produced."},
                "bytes_read": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Bytes scanned; the cost term of the ranking.",
                },
                "duration_ms": {"type": "integer", "minimum": 0, "description": "Wall clock."},
                "error_class": {
                    "type": "string",
                    "description": (
                        "syntax | semantic | plan_rejection | timeout | permission. "
                        "Required for any outcome other than 'success'."
                    ),
                },
                "error_text": {"type": "string", "description": "What the engine said."},
            },
            required=["namespace", "question", "sql", "outcome"],
        ),
        # The whole result is one shared type, so it is returned flat rather
        # than wrapped in a single-key object.
        output_schema=definition_schema("exemplar"),
        handler=handler,
    )


def _explain_exemplar_history(context: ServerContext, store: ExemplarStore) -> ToolDef:
    async def handler(args: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        sql = require_str(args, "sql")
        namespace = require_str(args, "namespace")
        normalized = normalize_sql(sql, context.adapter.engine)
        revisions = await asyncio.to_thread(
            store.history,
            engine=context.adapter.engine,
            namespace=namespace,
            normalized_sql=normalized,
        )
        return {
            "namespace": namespace,
            "normalized_sql": normalized,
            "revisions": [serialize.exemplar_revision(item) for item in revisions],
        }

    return ToolDef(
        name="explain_exemplar_history",
        title="Explain exemplar history",
        description=(
            "Every version of a remembered query, oldest first: when agentdb "
            "learned it, when it stopped being true of the schema, and what "
            "changed underneath it. The reason is recomputed against the schema "
            "version that superseded the query, so it names the column or the "
            "relation that broke it rather than reporting that something did."
        ),
        input_schema=object_schema(
            {
                "namespace": {"type": "string", "description": "Database or schema queried."},
                "sql": {
                    "type": "string",
                    "description": (
                        "Any version of the query. Literals are parameterized before "
                        "the lookup, so a query differing only in its constants finds "
                        "the same history."
                    ),
                },
            },
            required=["namespace", "sql"],
        ),
        output_schema=object_schema(
            {
                "namespace": {"type": "string"},
                "normalized_sql": {"type": "string"},
                "revisions": array_of("exemplar_revision", "Oldest first."),
            },
            required=["namespace", "normalized_sql", "revisions"],
        ),
        handler=handler,
    )


def _outcome(args: Mapping[str, JsonValue]) -> Outcome:
    value = require_str(args, "outcome")
    try:
        return Outcome(value)
    except ValueError as exc:
        raise ToolError(
            f"unknown outcome {value!r}",
            suggestion=f"use one of: {', '.join(OUTCOMES)}",
        ) from exc


def _provenance(args: Mapping[str, JsonValue]) -> Provenance:
    value = args.get("provenance")
    if value is None:
        return Provenance.AGENT
    try:
        return Provenance(str(value))
    except ValueError as exc:
        raise ToolError(
            f"unknown provenance {value!r}",
            suggestion=f"use one of: {', '.join(PROVENANCES)}",
        ) from exc


def _error_class(args: Mapping[str, JsonValue], outcome: Outcome) -> str | None:
    value = args.get("error_class")
    if outcome is Outcome.SUCCESS:
        if value is not None:
            raise ToolError("a successful outcome cannot carry an error_class")
        return None
    if not isinstance(value, str) or not value:
        raise ToolError(
            f"outcome {outcome.value!r} needs an error_class",
            suggestion="one of: syntax, semantic, plan_rejection, timeout, permission",
        )
    return value


def _columns(filters: frozenset[str], grouped: Sequence[str]) -> tuple[str, ...]:
    """The columns re-validation will check, deduplicated and ordered.

    Filter and grouping columns only: those are the ones whose disappearance or
    retype breaks the query outright, and recording every projected column would
    invalidate an exemplar over a ``SELECT *`` on a table that merely gained a
    column.
    """
    return tuple(sorted(frozenset(filters) | frozenset(grouped)))


def _optional_strings(args: Mapping[str, JsonValue], key: str) -> tuple[str, ...]:
    value = args.get(key)
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ToolError(f"argument {key!r} must be an array of strings")
    return tuple(str(item) for item in value)
