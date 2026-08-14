"""The plan-introspection service an agent calls before committing to a query.

This is the whole of arm ``A3`` in one call: take a draft query, ask the engine
what it would do with it, normalize the answer, and say what is wrong in terms
the agent can act on — *before* the query runs against a hundred million rows.

The facts it gathers are the minimum the rules need. Profiling is restricted to
the columns the query groups by, because those are the only columns a
distribution figure can change a warning about, and a probe per column is not
free.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentdb.adapters import (
    Adapter,
    Capability,
    ColumnProfile,
    ExplainMode,
    RelationRef,
    SamplePolicy,
)
from agentdb.config import Config
from agentdb.core.plan_analyzer import summarize
from agentdb.core.plan_ir import PlanSummary
from agentdb.core.plan_rules import RelationFacts, evaluate
from agentdb.core.query_shape import QueryShape, analyze


@dataclass(frozen=True, slots=True)
class PlanExplainer:
    """Explains a draft query against one engine."""

    adapter: Adapter
    config: Config = field(default_factory=Config)

    async def explain(self, sql: str, namespace: str) -> PlanSummary:
        """Plan ``sql`` and evaluate every rule its evidence supports.

        Nothing here executes the query. On ClickHouse that is not a choice —
        ``EXPLAIN`` cannot execute — and it is the property that makes this call
        safe to offer an agent mid-loop.
        """
        raw = await self.adapter.explain(sql, ExplainMode.ESTIMATE)
        summary = summarize(raw)
        shape = analyze(sql, self.adapter.engine)
        facts = await self._facts(shape, namespace)
        return evaluate(summary, shape, facts, self.config)

    async def _facts(self, shape: QueryShape, namespace: str) -> dict[str, RelationFacts]:
        """Layout, width and the profiles the rules can actually use."""
        facts: dict[str, RelationFacts] = {}
        for table in shape.tables:
            ref = RelationRef(namespace=namespace, name=table.rpartition(".")[2])
            detail = await self.adapter.describe_relation(ref)
            layout = await self.adapter.physical_layout(ref)
            facts[table] = RelationFacts(
                layout=layout,
                column_count=len(detail.columns),
                profiles=await self._profiles(ref, shape, detail.column_names),
            )
        return facts

    async def _profiles(
        self, ref: RelationRef, shape: QueryShape, columns: tuple[str, ...]
    ) -> dict[str, ColumnProfile]:
        wanted = [column for column in shape.group_by_columns if column in columns]
        if not wanted or not self.adapter.supports(Capability.COLUMN_STATS):
            return {}
        profiles = await self.adapter.column_profile(ref, wanted, self._sample_policy())
        return {profile.name: profile for profile in profiles}

    def _sample_policy(self) -> SamplePolicy:
        return SamplePolicy(
            fraction=self.config.default_sample_fraction,
            max_rows=self.config.profile_max_rows,
            timeout_s=self.config.query_timeout_s,
        )
