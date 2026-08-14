"""Assembling a grounded context from an adapter (SPEC §4, §13.1).

The builder talks to engines only through the :class:`~agentdb.adapters.Adapter`
protocol, so it is provable against a fake adapter with no database running —
anything that could only be tested against a live ClickHouse would be an engine
leak into core.

What it does *not* do is as deliberate as what it does: it never invents a fact
an adapter could not supply, and it never quietly downgrades. Asking for a level
the engine cannot support raises rather than returning a thinner payload under
the requested level's name, because a silently downgraded arm would show up in
the report as a measurement of something that never ran.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from agentdb.adapters import (
    Adapter,
    Capability,
    ColumnProfile,
    RelationDetail,
    RelationRef,
    SamplePolicy,
    UnsupportedCapabilityError,
)
from agentdb.config import Config
from agentdb.core.context import GroundedContext, GroundingLevel, RelationContext


@dataclass(frozen=True, slots=True)
class ContextBuilder:
    """Builds the payload one arm of the ablation ladder sends to a model."""

    adapter: Adapter
    config: Config = field(default_factory=Config)

    async def build(
        self,
        namespace: str,
        level: GroundingLevel = GroundingLevel.SCHEMA,
        relations: Sequence[str] | None = None,
    ) -> GroundedContext:
        """Assemble the context for ``namespace`` at ``level``.

        ``relations`` narrows the payload to named tables; without it every
        relation in the namespace is included, in the order the engine lists
        them, so two runs of the same arm produce the same bytes.
        """
        self._require_capabilities(level)
        refs = await self._resolve(namespace, relations)
        return GroundedContext(
            engine=self.adapter.engine,
            namespace=namespace,
            level=level,
            relations=tuple([await self._relation_context(ref, level) for ref in refs]),
        )

    def _require_capabilities(self, level: GroundingLevel) -> None:
        """Refuse a level this engine cannot honestly serve."""
        if level.includes_stats and not self.adapter.supports(Capability.COLUMN_STATS):
            raise UnsupportedCapabilityError(self.adapter.engine, Capability.COLUMN_STATS)

    async def _resolve(self, namespace: str, relations: Sequence[str] | None) -> list[RelationRef]:
        if relations is not None:
            return [RelationRef(namespace=namespace, name=name) for name in relations]
        listed = await self.adapter.list_relations(namespace)
        return [relation.ref for relation in listed]

    async def _relation_context(self, ref: RelationRef, level: GroundingLevel) -> RelationContext:
        detail = await self.adapter.describe_relation(ref)
        layout = await self.adapter.physical_layout(ref) if level.includes_layout else None
        profiles = await self._profiles(detail) if level.includes_stats else ()
        return RelationContext(
            detail=detail,
            layout=layout,
            profiles=profiles,
            profiled_columns_available=len(detail.columns),
        )

    async def _profiles(self, detail: RelationDetail) -> tuple[ColumnProfile, ...]:
        """Profile up to ``max_profiled_columns`` columns, in declaration order.

        Declaration order is the naive rule on purpose: it is deterministic, it
        needs no knowledge the level has not already granted, and it is itself
        under measurement. A question-aware selection is a change to the arm, and
        the benchmark has to be able to attribute the difference to it.
        """
        names = [column.name for column in detail.columns][: self.config.max_profiled_columns]
        if not names:
            return ()
        return tuple(await self.adapter.column_profile(detail.ref, names, self._sample_policy()))

    def _sample_policy(self) -> SamplePolicy:
        """The bounds every profiling probe runs under. Profiling is never a full scan."""
        return SamplePolicy(
            fraction=self.config.default_sample_fraction,
            max_rows=self.config.profile_max_rows,
            timeout_s=self.config.query_timeout_s,
        )
