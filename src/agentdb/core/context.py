"""The grounded context payload and how it is rendered for a model (SPEC §13.1).

This is the thing under measurement. Each :class:`GroundingLevel` corresponds to
one arm of the Family A ablation ladder (SPEC §11.3), so "does layout knowledge
help, and by how much" is answered by rebuilding this payload at a different
level and re-running the same tasks against the same seeds.

Two rules the renderer obeys, because the benchmark depends on them:

* **Deterministic output.** The same facts render byte-identically every time.
  A payload that reordered itself would make ``context_bytes`` and the prompt
  hash meaningless, and two runs of one arm incomparable.
* **Estimates stay labelled.** Every profile line says how it was obtained and
  over how many rows. The agent is told what is measured and what is guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from agentdb.adapters import (
    ColumnProfile,
    Engine,
    PhysicalLayout,
    RelationDetail,
    RelationRef,
)


class GroundingLevel(StrEnum):
    """How much grounding a context payload carries.

    The ladder is cumulative: each level is the previous one plus one kind of
    fact, so a difference in accuracy between two arms is attributable to
    exactly one addition.
    """

    SCHEMA = "schema"
    """``CREATE TABLE`` DDL only — what the official ClickHouse MCP server gives
    an agent, and the ``A0_baseline`` arm."""

    STATS = "stats"
    """SCHEMA + sampled column profiles: cardinality, null ratio, top values,
    min/max. The ``A1_stats`` arm."""

    LAYOUT = "layout"
    """STATS + physical layout: table engine, sort key, partition key, skip
    indexes, projections. The ``A2_layout`` arm — the ClickHouse-differentiating
    payload, since nothing in a DDL dump says which filters prune granules."""

    @property
    def includes_stats(self) -> bool:
        return self is not GroundingLevel.SCHEMA

    @property
    def includes_layout(self) -> bool:
        return self is GroundingLevel.LAYOUT


@dataclass(frozen=True, slots=True)
class RelationContext:
    """Everything the payload knows about one relation."""

    detail: RelationDetail
    layout: PhysicalLayout | None = None
    profiles: tuple[ColumnProfile, ...] = ()
    profiled_columns_available: int = 0
    """How many columns *could* have been profiled. Rendered whenever it exceeds
    the number actually profiled, so a partial profile never reads as a complete
    one — a 105-column table with 30 profiled columns has to say so."""

    @property
    def ref(self) -> RelationRef:
        return self.detail.ref


@dataclass(frozen=True, slots=True)
class GroundedContext:
    """The assembled payload for one question, at one grounding level."""

    engine: Engine
    namespace: str
    level: GroundingLevel
    relations: tuple[RelationContext, ...]

    def render(self) -> str:
        """The payload as text, deterministically."""
        blocks = [_render_relation(relation, self.level) for relation in self.relations]
        return "\n\n".join(blocks)

    @property
    def size_bytes(self) -> int:
        """UTF-8 size of the rendered payload — grounding is not free (SPEC §11.1)."""
        return len(self.render().encode("utf-8"))


def _render_relation(relation: RelationContext, level: GroundingLevel) -> str:
    sections = [relation.detail.create_statement.strip()]
    if level.includes_layout and relation.layout is not None:
        sections.append(_render_layout(relation.layout))
    if level.includes_stats and relation.profiles:
        sections.append(_render_profiles(relation))
    return "\n\n".join(sections)


def _render_layout(layout: PhysicalLayout) -> str:
    """Physical design, as short lines an agent can act on.

    Only facts the engine reported are listed. An absent sort key is omitted
    rather than rendered as "none", because a missing line reads as "unknown"
    while an explicit "none" reads as "measured to be absent".
    """
    lines = [f"Physical layout of {layout.ref}:"]
    if layout.table_engine is not None:
        lines.append(f"- engine: {layout.table_engine}")
    if layout.order_by:
        lines.append(
            f"- sort key (ORDER BY): {', '.join(layout.order_by)}"
            " — filters prune granules through this key, left to right"
        )
    if layout.partition_by:
        lines.append(f"- partition key: {', '.join(layout.partition_by)}")
    if layout.primary_key:
        lines.append(f"- primary key: {', '.join(layout.primary_key)}")
    if layout.sampling_key is not None:
        lines.append(f"- sampling key: {layout.sampling_key}")
    for index in layout.skip_indexes:
        lines.append(
            f"- skip index {index.name}: {index.index_type} on {index.expression}"
            f" (granularity {index.granularity})"
        )
    for projection in layout.projections:
        lines.append(f"- projection {projection.name}: {projection.query}")
    for definition in layout.indexes:
        columns = ", ".join(definition.columns)
        lines.append(f"- index {definition.name}: {definition.method} on {columns}")
    if layout.approx_rows is not None:
        lines.append(f"- approx rows: {layout.approx_rows:,}")
    if layout.compression_ratio is not None:
        lines.append(f"- compression ratio: {layout.compression_ratio:.1f}x")
    return "\n".join(lines)


def _render_profiles(relation: RelationContext) -> str:
    profiled = len(relation.profiles)
    header = f"Column profiles for {relation.ref}"
    if relation.profiled_columns_available > profiled:
        header += f" ({profiled} of {relation.profiled_columns_available} columns)"
    lines = [f"{header}:"]
    lines.extend(_render_profile(profile) for profile in relation.profiles)
    return "\n".join(lines)


def _render_profile(profile: ColumnProfile) -> str:
    facts: list[str] = [profile.data_type]
    if profile.approx_distinct is not None:
        facts.append(f"~{profile.approx_distinct:,} distinct")
    if profile.null_ratio is not None:
        facts.append(f"{profile.null_ratio:.0%} null")
    if profile.min_value is not None and profile.max_value is not None:
        facts.append(f"range {profile.min_value}..{profile.max_value}")
    if profile.top_values:
        top = ", ".join(f"{value} ({count:,})" for value, count in profile.top_values)
        facts.append(f"top: {top}")
    facts.append(f"[{profile.sample_method}, {profile.sampled_rows:,} rows]")
    return f"- {profile.name}: {'; '.join(facts)}"
