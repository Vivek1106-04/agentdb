"""Normalizing an engine's plan into the IR (SPEC §7, §8.1).

ClickHouse is asked for its plan as JSON rather than as the indented text tree.
The text tree is meant for humans and its shape is not a contract; the JSON
carries the same index evidence with keys that can be tested. Getting this wrong
is not a crash — it is a *quietly wrong pruning number*, which is worse, so the
parser refuses anything it cannot read rather than defaulting to zero.

What the parser will not do is invent. ClickHouse's ``EXPLAIN`` executes nothing,
so ``actual_rows`` stays ``None`` on every node. An agent has to be able to tell
an estimate from a measurement.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from agentdb.adapters import RawPlan
from agentdb.core.plan_ir import PlanNode, PlanOp, PlanSummary

PLAN_KEY = "Plan"
PLANS_KEY = "Plans"
NODE_TYPE_KEY = "Node Type"
INDEXES_KEY = "Indexes"

INITIAL_GRANULES = "Initial Granules"
SELECTED_GRANULES = "Selected Granules"
INITIAL_PARTS = "Initial Parts"
SELECTED_PARTS = "Selected Parts"

PRIMARY_KEY_INDEX = "PrimaryKey"
SKIP_INDEX = "Skip"


class PlanParseError(ValueError):
    """The engine's plan output could not be read.

    Raised rather than absorbed: a plan the analyzer half-understood would
    produce warnings nobody can trust.
    """


def parse_plan(raw: RawPlan) -> PlanNode:
    """Turn ``raw`` into a normalized tree."""
    document = _load(raw.payload)
    return _node(document)


def summarize(raw: RawPlan) -> PlanSummary:
    """Parse ``raw`` and aggregate the pruning evidence across its scans."""
    root = parse_plan(raw)
    scans = tuple(node for node in root.walk() if node.op is PlanOp.SCAN)

    considered = sum(node.granules_total or 0 for node in scans)
    selected = sum(node.granules_selected or 0 for node in scans)
    ratio = selected / considered if considered else None

    return PlanSummary(
        root=root,
        engine=raw.engine,
        sql=raw.sql,
        pruning_ratio=ratio,
        pruning_unit="granule" if considered else None,
        full_scan_relations=tuple(
            node.relation
            for node in scans
            if node.relation is not None and node.pruning_ratio == 1.0
        ),
    )


def _load(payload: str) -> Mapping[str, Any]:
    """Read the JSON document ClickHouse returns for ``EXPLAIN json = 1``."""
    try:
        document = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise PlanParseError(f"plan output is not JSON: {exc}") from exc

    if isinstance(document, Sequence) and not isinstance(document, str | bytes):
        if not document:
            raise PlanParseError("plan output is an empty document")
        document = document[0]
    if not isinstance(document, Mapping) or PLAN_KEY not in document:
        raise PlanParseError(f"plan output has no {PLAN_KEY!r} key")

    plan = document[PLAN_KEY]
    if not isinstance(plan, Mapping):
        raise PlanParseError(f"{PLAN_KEY!r} is not a plan node")
    return plan


def _node(payload: Mapping[str, Any]) -> PlanNode:
    node_type = str(payload.get(NODE_TYPE_KEY, "Unknown"))
    indexes = payload.get(INDEXES_KEY) or ()
    granules_total, granules_selected = _range(indexes, INITIAL_GRANULES, SELECTED_GRANULES)
    parts_total, parts_selected = _range(indexes, INITIAL_PARTS, SELECTED_PARTS)

    return PlanNode(
        op=classify(node_type),
        node_type=node_type,
        relation=_relation(payload),
        estimated_rows=_optional_int(payload.get("Estimated Rows")),
        estimated_cost=_optional_float(payload.get("Estimated Cost")),
        filters=_filters(payload),
        granules_total=granules_total,
        granules_selected=granules_selected,
        parts_total=parts_total,
        parts_selected=parts_selected,
        index_used=_indexes_that_fired(indexes),
        projection_used=_optional_str(payload.get("Projection")),
        children=tuple(_node(child) for child in payload.get(PLANS_KEY, ())),
    )


def classify(node_type: str) -> PlanOp:
    """Map an engine's node name onto the shared operation vocabulary."""
    if "Remote" in node_type or node_type in {"Resize", "Union", "Exchange"}:
        return PlanOp.EXCHANGE
    if node_type.startswith("ReadFromPreparedSource"):
        return PlanOp.OTHER
    if node_type.startswith("ReadFrom"):
        return PlanOp.SCAN
    if "Join" in node_type:
        return PlanOp.JOIN
    if "Aggregat" in node_type:
        return PlanOp.AGGREGATE
    if "Sorting" in node_type:
        return PlanOp.SORT
    if node_type.startswith(("Limit", "Offset")):
        return PlanOp.LIMIT
    if node_type.startswith("Filter"):
        return PlanOp.FILTER
    if "Projection" in node_type:
        return PlanOp.PROJECTION_READ
    return PlanOp.OTHER


def _relation(payload: Mapping[str, Any]) -> str | None:
    """The relation a scan reads, as ClickHouse names it in the plan."""
    if classify(str(payload.get(NODE_TYPE_KEY, ""))) is not PlanOp.SCAN:
        return None
    for key in ("Description", "Table", "Relation"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _filters(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Filter expressions the plan states, including index conditions."""
    stated = [
        str(index["Condition"])
        for index in payload.get(INDEXES_KEY) or ()
        if isinstance(index, Mapping) and index.get("Condition")
    ]
    column = payload.get("Filter Column")
    if isinstance(column, str) and column:
        stated.append(column)
    return tuple(stated)


def _range(indexes: object, initial_key: str, selected_key: str) -> tuple[int | None, int | None]:
    """What the scan started with and what survived every index in the chain.

    ClickHouse applies indexes in sequence, each narrowing the last, so the
    honest pair is the widest "initial" and the narrowest "selected" — not the
    first entry's numbers, which describe only the first filter.
    """
    if not isinstance(indexes, Sequence) or isinstance(indexes, str):
        return None, None

    initial = [
        _optional_int(entry.get(initial_key)) for entry in indexes if isinstance(entry, Mapping)
    ]
    selected = [
        _optional_int(entry.get(selected_key)) for entry in indexes if isinstance(entry, Mapping)
    ]
    known_initial = [value for value in initial if value is not None]
    known_selected = [value for value in selected if value is not None]
    return (
        max(known_initial) if known_initial else None,
        min(known_selected) if known_selected else None,
    )


def _indexes_that_fired(indexes: object) -> tuple[str, ...]:
    """Indexes that actually removed data, named as an agent would name them.

    An index that was consulted and pruned nothing is not listed: "the skip index
    fired" and "the skip index exists" are different facts, and only the first
    justifies leaving the schema alone.
    """
    if not isinstance(indexes, Sequence) or isinstance(indexes, str):
        return ()

    fired: list[str] = []
    for entry in indexes:
        if not isinstance(entry, Mapping):
            continue
        initial = _optional_int(entry.get(INITIAL_GRANULES))
        selected = _optional_int(entry.get(SELECTED_GRANULES))
        if initial is None or selected is None or selected >= initial:
            continue
        index_type = str(entry.get("Type", "Index"))
        name = entry.get("Name")
        fired.append(str(name) if index_type == SKIP_INDEX and name else index_type)
    return tuple(fired)


def _optional_int(value: object) -> int | None:
    return int(value) if isinstance(value, int | float) else None


def _optional_float(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
