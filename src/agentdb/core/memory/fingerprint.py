"""The schema fingerprint, and the cheap re-validation it triggers (SPEC §10.3).

A fingerprint is a sha256 over the sorted (relation, column, type, physical
layout) of one namespace. When it changes, every currently-valid exemplar is
re-checked against the new snapshot — not re-executed — and the ones whose
relations, columns or types no longer hold get ``valid_to`` stamped.

**The fingerprint covers design, not size.** Row counts, file counts, on-disk
bytes and compression ratios are deliberately excluded: they move on every
insert, and folding them in would invalidate every exemplar in the store hourly
while telling an agent nothing about whether its SQL still type-checks. What is
included is what a query's correctness and its pruning depend on — the sort key
on ClickHouse, the clustering columns and the statistics column set on
Databricks — so a layout change invalidates exactly as a column rename does.
That symmetry is the M5 requirement: a clustering-key change must invalidate an
exemplar exactly as a ClickHouse sort-key change does.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from agentdb.adapters import Engine, PhysicalLayout, RelationDetail

SNAPSHOT_VERSION = 1
"""Bumped when the fingerprinted field set changes.

Carried inside the hashed payload so that a change to *what* is fingerprinted
supersedes every stored version rather than silently comparing old hashes to new
ones — the one case where invalidating the whole store is the correct answer.
"""


@dataclass(frozen=True, slots=True)
class RelationSnapshot:
    """One relation's fingerprinted state: its columns and its physical design."""

    name: str
    columns: tuple[tuple[str, str], ...]
    """``(column, data_type)`` sorted by column name."""

    layout: tuple[tuple[str, str], ...]
    """``(property, value)`` sorted by property. Only design properties."""

    @property
    def column_types(self) -> Mapping[str, str]:
        return dict(self.columns)


@dataclass(frozen=True, slots=True)
class NamespaceSnapshot:
    """Every relation of one namespace, in a canonical order."""

    engine: Engine
    namespace: str
    relations: tuple[RelationSnapshot, ...]
    """Sorted by relation name."""

    @property
    def by_name(self) -> Mapping[str, RelationSnapshot]:
        return {relation.name: relation for relation in self.relations}


def snapshot(
    engine: Engine,
    namespace: str,
    details: Iterable[RelationDetail],
    layouts: Iterable[PhysicalLayout] = (),
) -> NamespaceSnapshot:
    """Build the canonical snapshot of ``namespace`` from what an adapter returned.

    ``layouts`` is optional because an engine may be queried for schema without
    layout; a relation with no layout contributes its columns alone, and the
    fingerprint changes the moment layout does arrive. That is correct: the
    grounding the exemplar was written against genuinely differs.
    """
    layout_by_name = {layout.ref.name: layout for layout in layouts}
    relations = tuple(
        sorted(
            (
                RelationSnapshot(
                    name=detail.ref.name,
                    columns=tuple(
                        sorted(
                            (column.name, _normalize_type(column.data_type))
                            for column in detail.columns
                        )
                    ),
                    layout=_layout_properties(layout_by_name.get(detail.ref.name)),
                )
                for detail in details
            ),
            key=lambda relation: relation.name,
        )
    )
    return NamespaceSnapshot(engine=engine, namespace=namespace, relations=relations)


def fingerprint(state: NamespaceSnapshot) -> str:
    """Return the sha256 hex digest of ``state``.

    Hashing the JSON encoding rather than a hand-built string keeps the digest
    stable against separator choices and unambiguous about nesting — two layouts
    that differ only in where a boundary falls must not collide.
    """
    payload = json.dumps(snapshot_to_json(state), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def snapshot_to_json(state: NamespaceSnapshot) -> Mapping[str, object]:
    """Encode ``state`` for the ``layout_json`` column, losslessly."""
    return {
        "version": SNAPSHOT_VERSION,
        "engine": state.engine,
        "namespace": state.namespace,
        "relations": [
            {
                "name": relation.name,
                "columns": [list(pair) for pair in relation.columns],
                "layout": [list(pair) for pair in relation.layout],
            }
            for relation in state.relations
        ],
    }


def snapshot_from_json(payload: Mapping[str, object]) -> NamespaceSnapshot:
    """Decode what :func:`snapshot_to_json` wrote.

    Raises :class:`ValueError` on a payload written by a different snapshot
    version: comparing across versions would report every relation as changed,
    and a loud failure beats a store that quietly invalidates itself.
    """
    version = payload.get("version")
    if version != SNAPSHOT_VERSION:
        raise ValueError(f"snapshot version {version!r} is not the supported {SNAPSHOT_VERSION}")

    engine = cast(Engine, payload["engine"])
    relations = cast(Sequence[Mapping[str, object]], payload["relations"])
    return NamespaceSnapshot(
        engine=engine,
        namespace=str(payload["namespace"]),
        relations=tuple(
            RelationSnapshot(
                name=str(relation["name"]),
                columns=_pairs(relation["columns"]),
                layout=_pairs(relation["layout"]),
            )
            for relation in relations
        ),
    )


def _pairs(value: object) -> tuple[tuple[str, str], ...]:
    """Decode a JSON list of two-element lists back into a tuple of pairs."""
    return tuple((str(pair[0]), str(pair[1])) for pair in cast(Sequence[Sequence[object]], value))


def invalidation_reason(
    relations: Sequence[str],
    columns: Sequence[str],
    current: NamespaceSnapshot,
    previous: NamespaceSnapshot | None = None,
) -> str | None:
    """Say why an exemplar naming ``relations``/``columns`` no longer holds, or ``None``.

    The returned string is the whole point of bi-temporality being mechanical
    rather than inferred: it is what ``explain_exemplar_history`` shows when
    asked *when did this query stop working, and what changed*. The first
    failure found is reported — an exemplar is invalid once, not by degree.

    ``columns`` may be qualified (``hits.EventTime``) or bare (``EventTime``); a
    bare name holds as long as some relation the exemplar names still offers it,
    which is exactly the resolution the engine itself would do.
    """
    present = current.by_name
    for relation in relations:
        if relation not in present:
            return f"relation {relation!r} no longer exists"

    for column in columns:
        reason = _column_reason(column, relations, present)
        if reason is not None:
            return reason

    if previous is None:
        return None
    return _type_change_reason(relations, columns, current, previous)


def _column_reason(
    column: str,
    relations: Sequence[str],
    present: Mapping[str, RelationSnapshot],
) -> str | None:
    relation_name, _, bare = column.rpartition(".")
    if relation_name:
        if relation_name not in present:
            return f"relation {relation_name!r} no longer exists"
        if bare not in present[relation_name].column_types:
            return f"column {column!r} no longer exists"
        return None

    if any(bare in present[name].column_types for name in relations if name in present):
        return None
    return f"column {bare!r} no longer exists on any relation this exemplar names"


def _type_change_reason(
    relations: Sequence[str],
    columns: Sequence[str],
    current: NamespaceSnapshot,
    previous: NamespaceSnapshot,
) -> str | None:
    """Report the first column whose declared type moved.

    Compatibility is exact-match on the normalized type name, which is stricter
    than the engines are: ``Int32`` widening to ``Int64`` breaks nothing at
    runtime. Strict is the right side to err on here, because the cases that do
    break — a ``String`` becoming a ``DateTime``, nullability appearing under a
    comparison — are invisible in the SQL text and expensive to discover as a
    wrong answer. A re-validated exemplar is cheap; a confidently wrong one is not.
    """
    was = previous.by_name
    now = current.by_name
    for column in columns:
        relation_name, _, bare = column.rpartition(".")
        candidates = [relation_name] if relation_name else list(relations)
        for name in candidates:
            if name not in was or name not in now:
                continue
            old_type = was[name].column_types.get(bare)
            new_type = now[name].column_types.get(bare)
            if old_type is not None and new_type is not None and old_type != new_type:
                return f"column {name}.{bare} changed type from {old_type} to {new_type}"
    return None


def _normalize_type(data_type: str) -> str:
    """Collapse whitespace so ``Decimal(10, 2)`` and ``Decimal(10,2)`` agree."""
    return " ".join(data_type.split()).replace(", ", ",")


def _layout_properties(layout: PhysicalLayout | None) -> tuple[tuple[str, str], ...]:
    """The design half of a physical layout, as sorted ``(property, value)`` pairs.

    Size-varying fields are absent on purpose (see the module docstring). What
    remains is every property that decides whether a predicate prunes.
    """
    if layout is None:
        return ()

    properties: dict[str, str | None] = {
        "table_engine": layout.table_engine,
        "table_format": layout.table_format,
        "order_by": _join(layout.order_by),
        "partition_by": _join(layout.partition_by),
        "primary_key": _join(layout.primary_key),
        "sampling_key": layout.sampling_key,
        "ttl": layout.ttl,
        "clustering_columns": _join(layout.clustering_columns),
        "zorder_columns": _join(layout.zorder_columns),
        "stats_columns": _join(layout.stats_columns),
        "deletion_vectors_enabled": _flag(layout.deletion_vectors_enabled),
        "skip_indexes": _join(
            tuple(
                f"{index.name}:{index.index_type}:{index.expression}:{index.granularity}"
                for index in layout.skip_indexes
            )
        ),
        "projections": _join(
            tuple(f"{projection.name}:{projection.query}" for projection in layout.projections)
        ),
    }
    return tuple(sorted((name, value) for name, value in properties.items() if value is not None))


def _join(values: tuple[str, ...] | None) -> str | None:
    """Join a tuple property, preserving order — a sort key is ordered, not a set."""
    return None if values is None else ",".join(values)


def _flag(value: bool | None) -> str | None:
    return None if value is None else str(value).lower()
