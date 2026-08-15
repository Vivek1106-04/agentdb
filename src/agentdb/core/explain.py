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

from collections.abc import Mapping
from dataclasses import dataclass, field

from agentdb.adapters import (
    Adapter,
    Capability,
    ColumnProfile,
    ExplainMode,
    RawPlan,
    RelationRef,
    SamplePolicy,
)
from agentdb.config import Config
from agentdb.core.plan_analyzer import summarize
from agentdb.core.plan_analyzer_databricks import summarize as summarize_databricks
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

        Nothing here executes the query. Neither engine's ``EXPLAIN`` can — which
        is what makes this call safe to offer an agent mid-loop, before the query
        touches a hundred million rows.

        Facts are gathered before the plan is summarized on Databricks, because
        the plan reports how many files were *read* and the denominator lives in
        ``DESCRIBE DETAIL``. A ratio without both numbers is not reported.
        """
        raw = await self.adapter.explain(sql, ExplainMode.ESTIMATE)
        shape = analyze(sql, self.adapter.engine)
        facts = await self._facts(shape, namespace)
        summary = self._summarize(raw, facts)
        return evaluate(summary, shape, facts, self.config)

    def _summarize(self, raw: RawPlan, facts: Mapping[str, RelationFacts]) -> PlanSummary:
        """Normalize the engine's plan output with the engine's own parser."""
        if self.adapter.engine == "databricks":
            return summarize_databricks(
                raw,
                files_total={
                    relation: known.layout.num_files
                    for relation, known in facts.items()
                    if known.layout.num_files is not None
                },
            )
        return summarize(raw)

    async def _facts(self, shape: QueryShape, namespace: str) -> dict[str, RelationFacts]:
        """Layout, width, ordinals and the profiles the rules can actually use."""
        facts: dict[str, RelationFacts] = {}
        for table in shape.qualified_tables:
            ref = self._ref(table, namespace)
            detail = await self.adapter.describe_relation(ref)
            layout = await self.adapter.physical_layout(ref)
            facts[table] = RelationFacts(
                layout=layout,
                column_count=len(detail.columns),
                profiles=await self._profiles(ref, shape, detail.column_names),
                column_ordinals={
                    name: position for position, name in enumerate(detail.column_names, start=1)
                },
            )
        return facts

    def _ref(self, table: str, namespace: str) -> RelationRef:
        """Build a reference the adapter can resolve, however the query wrote it.

        On Databricks a catalog written into the query is kept: the agent may
        have named a catalog other than the adapter's default, and silently
        replacing it would explain a plan for a different table.
        """
        parts = table.split(".")
        name = parts[-1]
        if self.adapter.engine != "databricks":
            return RelationRef(namespace=namespace, name=name)
        catalog = parts[-3] if len(parts) >= 3 else None
        schema = parts[-2] if len(parts) >= 2 else namespace
        return RelationRef(catalog=catalog, namespace=schema, name=name)

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
