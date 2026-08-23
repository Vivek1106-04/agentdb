"""``A6_full`` — the arm that carries the advisor's findings (SPEC §11.3).

A5 gives the model the schema, the layout, the plan and what this connection has
already learned. A6 adds what the advisor found: which filters cannot prune on
this table, which columns Delta keeps no statistics for, which query shapes the
key cannot serve.

**Where the demand signal comes from, and why not from the query log.** The
advisor needs to know what the table is normally asked. In a deployment that is
``mine_workload``, reading the engine's own log. On a benchmark instance that log
holds *this project's own gold executions*, so mining it would feed the advisor
the answers and call the result an ablation. A6 therefore reads a committed
reference workload of third-party query shapes — ClickBench's published queries
and TPC-H's — authored by neither this project nor for this benchmark. The file
is committed, the arm's fingerprint covers it, and the honest limitation is that
A6 measures the advisor against a *representative* workload rather than the
operator's real one.

Arm A6 is the advisor under measurement. If it does not beat A5, the finding is
that the advisor is not worth its tokens, and SPEC §9 already says to publish
that rather than bury it.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from agentdb.adapters.clickhouse_client import Importer
from agentdb.bench.memory_provider import (
    Connector,
    MemoryContextProvider,
    clickhouse_memory_provider,
)
from agentdb.bench.provider import fingerprint_config
from agentdb.core.advisor import (
    ClickHouseAdvisor,
    DatabricksAdvisor,
    Recommendation,
    demand_from_queries,
)
from agentdb.core.advisor.render import render_recommendations
from agentdb.core.context import GroundedContext, RelationContext
from agentdb.core.memory.postgres import connect
from agentdb.core.query_shape import QueryShape, analyze

VERSION = "1.0"

WORKLOADS = {"clickbench": "clickbench.sql", "tpch": "tpch.sql"}
"""Committed reference workloads, by name. Chosen per namespace by the operator,
because only they know which table this namespace holds."""


def load_workload(name: str) -> tuple[str, ...]:
    """Read one committed reference workload into statements.

    A named workload rather than a path by default: the file ships inside the
    package, so a reader running ``make bench`` gets the same demand signal the
    published numbers were produced with.
    """
    if name in WORKLOADS:
        text = (
            resources.files("agentdb.bench.workloads")
            .joinpath(WORKLOADS[name])
            .read_text(encoding="utf-8")
        )
    else:
        text = Path(name).read_text(encoding="utf-8")
    return tuple(
        statement.strip() for statement in _strip_comments(text).split(";") if statement.strip()
    )


@dataclass(frozen=True, slots=True)
class AdvisedContextProvider:
    """``A6_full``: the A5 payload plus the advisor's findings."""

    base: MemoryContextProvider
    workload: tuple[str, ...]
    name: str
    version: str = VERSION

    _cache: dict[str, str] = field(default_factory=dict, repr=False, compare=False)

    @property
    def fingerprint(self) -> str:
        """Covers the base arm and the workload the advice was derived from."""
        return fingerprint_config(
            {
                "provider": self.name,
                "version": self.version,
                "base": self.base.fingerprint,
                "workload_sha256": _digest(self.workload),
                "workload_statements": len(self.workload),
            }
        )

    async def context(self, *, namespace: str, question: str) -> str:
        """The A5 payload, plus one advice block per namespace."""
        payload = await self.base.context(namespace=namespace, question=question)
        advice = self._cache.get(namespace)
        if advice is None:
            built = await self.base.base.build(namespace)
            advice = await asyncio.to_thread(self._advise, built)
            self._cache[namespace] = advice
        return f"{payload}\n\n{advice}" if advice else payload

    async def explain_plan(self, *, sql: str, namespace: str) -> str | None:
        return await self.base.explain_plan(sql=sql, namespace=namespace)

    def _advise(self, built: GroundedContext) -> str:
        """Run the engine's advisor over every relation the payload describes.

        Advice is per namespace and cached, like the grounding it sits beside:
        the physical design does not change between two questions in one run,
        and rebuilding it per task would let a background merge move the payload
        halfway through an arm.
        """
        shapes = tuple(analyze(statement, built.engine) for statement in self.workload)
        found: list[Recommendation] = []
        for relation in built.relations:
            found.extend(self._for_relation(built.engine, relation, shapes))
        return render_recommendations(found)

    def _for_relation(
        self, engine: str, relation: RelationContext, shapes: Sequence[QueryShape]
    ) -> tuple[Recommendation, ...]:
        if relation.layout is None:
            return ()
        demand = demand_from_queries(relation.ref.name, shapes)
        if not demand.queries:
            return ()
        if engine == "clickhouse":
            return ClickHouseAdvisor(config=self.base.base.builder.config).advise(
                ref=relation.ref,
                layout=relation.layout,
                profiles=relation.profiles,
                demand=demand,
            )
        return DatabricksAdvisor(config=self.base.base.builder.config).advise(
            ref=relation.ref,
            layout=relation.layout,
            detail=relation.detail,
            profiles=relation.profiles,
            demand=demand,
        )


async def clickhouse_advised_provider(
    *,
    workload: str = "clickbench",
    name: str | None = None,
    k: int | None = None,
    dsn: str | None = None,
    importer: Importer = importlib.import_module,
    connector: Connector = connect,
) -> AdvisedContextProvider:
    """Build ``A6_full`` against a live ClickHouse, store and reference workload.

    Named by dotted path from ``eval/providers.yaml`` like every other Family A
    arm. A6 is A5 plus advice, so the memory layer underneath it carries the
    negative exemplars too: the ladder is cumulative or the deltas mean nothing.
    """
    arm = name or "agentdb/A6_full"
    base = await clickhouse_memory_provider(
        include_failures=True,
        name=f"{arm}/A5",
        k=k,
        dsn=dsn,
        importer=importer,
        connector=connector,
    )
    return build_advised_provider(base=base, workload=workload, name=arm)


def build_advised_provider(
    *,
    base: MemoryContextProvider,
    workload: str,
    name: str | None = None,
) -> AdvisedContextProvider:
    """Wrap an A5 provider so its payload carries the advisor's findings."""
    return AdvisedContextProvider(
        base=base,
        workload=load_workload(workload),
        name=name or "agentdb/A6_full",
    )


def _strip_comments(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("--"))


def _digest(statements: Sequence[str]) -> str:
    """Hash of the workload text, so a changed reference workload changes the arm."""
    joined = "\n".join(statements).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()
