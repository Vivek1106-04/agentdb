"""``A4_memory`` and ``A5_negmemory`` — the arms that remember (SPEC §11.3).

A3 gives the model the schema, the layout and the plan. These two add what the
connection has already learned: A4 the queries that worked, A5 those plus the
queries that failed, with the error class attached. One provider serves both,
differing in a single flag, so the A4→A5 delta is attributable to the negative
block and to nothing else.

Three properties this arm has to hold to be measurable:

* **Invalidated exemplars never reach the model.** Every namespace is
  fingerprinted before retrieval, so an exemplar the schema broke is withheld
  rather than ranked down (SPEC §10.3). Without this the arm would sometimes
  measure how well a model recovers from stale context.
* **Retrieval is question-aware; grounding is not.** The A0 to A3 payload is per
  namespace and cached, and the exemplar block is per question. That is the
  whole difference between these arms and the ones below them.
* **The store is agentdb's, not the harness's.** ``agenteval`` sees a
  ``ContextProvider`` — a name, a version, a fingerprint and a coroutine — and
  never learns that a Postgres exists (SPEC §4.1.6).
"""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import Callable
from dataclasses import dataclass, field

from agentdb.adapters.clickhouse_client import Importer
from agentdb.bench.provider import (
    GroundedContextProvider,
    clickhouse_provider,
    databricks_provider,
    fingerprint_config,
)
from agentdb.core import GroundingLevel
from agentdb.core.context import GroundedContext
from agentdb.core.memory.fingerprint import snapshot
from agentdb.core.memory.models import ScoredExemplar
from agentdb.core.memory.postgres import connect
from agentdb.core.memory.render import render_exemplars
from agentdb.core.memory.store import Connection, ExemplarStore

VERSION = "1.0"
"""Bumped whenever the exemplar block changes shape, because that changes the number."""


@dataclass(frozen=True, slots=True)
class MemoryContextProvider:
    """A grounded provider that appends retrieved exemplars to its payload."""

    base: GroundedContextProvider
    store: ExemplarStore
    name: str
    include_failures: bool = False
    """``A5_negmemory`` when true. The single flag the two arms differ by."""

    k: int | None = None
    """Exemplars per block. ``None`` takes the configured context budget."""

    version: str = VERSION

    _synced: set[str] = field(default_factory=set, repr=False, compare=False)

    @property
    def fingerprint(self) -> str:
        """Hash of everything that decides what this provider returns.

        It carries the base provider's fingerprint plus the retrieval weights,
        because "A4 scored 64%" means nothing without saying which ranking
        produced the exemplars — and every weight in that ranking is itself an
        ablation arm (SPEC §10.4).
        """
        config = self.base.builder.config
        return fingerprint_config(
            {
                "provider": self.name,
                "version": self.version,
                "base": self.base.fingerprint,
                "include_failures": self.include_failures,
                "k": self.k if self.k is not None else config.exemplar_top_k,
                "weights": dict(config.retrieval_weights.as_mapping()),
                "recency_tau_days": config.exemplar_recency_tau_days,
                "candidate_pool": config.exemplar_candidate_pool,
            }
        )

    async def context(self, *, namespace: str, question: str) -> str:
        """The A3 payload, plus the exemplars that survived the current schema."""
        built = await self.base.build(namespace)
        await self._sync(namespace, built)

        relations = _relations(built)
        positive = await asyncio.to_thread(
            self.store.retrieve,
            question,
            engine=self.base.builder.adapter.engine,
            namespace=namespace,
            relations=relations,
            k=self.k,
        )
        negative: tuple[ScoredExemplar, ...] = ()
        if self.include_failures:
            negative = await asyncio.to_thread(
                self.store.retrieve,
                question,
                engine=self.base.builder.adapter.engine,
                namespace=namespace,
                relations=relations,
                k=self.k,
                failures_only=True,
            )

        exemplars = render_exemplars(positive, negative)
        grounding = built.render()
        return f"{grounding}\n\n{exemplars}" if exemplars else grounding

    async def explain_plan(self, *, sql: str, namespace: str) -> str | None:
        """Delegated: these arms are A3 plus memory, and A3 is where the plan lives."""
        return await self.base.explain_plan(sql=sql, namespace=namespace)

    async def aclose(self) -> None:
        """Release the engine connection underneath. The store's outlives the run."""
        await self.base.aclose()

    async def _sync(self, namespace: str, built: GroundedContext) -> None:
        """Fingerprint the namespace once per run, invalidating what it broke.

        Once, not per question: the schema does not move mid-run, and a sync per
        task would charge every arm a round trip to prove the same thing 100
        times. A live server that *did* change underneath a run would be caught
        on the next one, which is the honest granularity for a benchmark.
        """
        if namespace in self._synced:
            return
        state = snapshot(
            built.engine,
            namespace,
            [relation.detail for relation in built.relations],
            [relation.layout for relation in built.relations if relation.layout is not None],
        )
        await asyncio.to_thread(self.store.sync, state)
        self._synced.add(namespace)


def _relations(built: GroundedContext) -> tuple[str, ...]:
    """The namespace's relation names, as the relation term's candidate set.

    The question's own relations are unknown before the SQL exists, so what is
    offered is what the payload describes. Where a namespace holds one table the
    term is constant and the semantic term decides the ranking; where it holds
    many, an exemplar over the same tables outranks one merely sharing words.
    """
    return tuple(relation.ref.name for relation in built.relations)


def build_memory_provider(
    *,
    base: GroundedContextProvider,
    store: ExemplarStore,
    include_failures: bool = False,
    name: str | None = None,
    k: int | None = None,
) -> MemoryContextProvider:
    """Wrap ``base`` so its payload carries retrieved exemplars."""
    arm = "negmemory" if include_failures else "memory"
    return MemoryContextProvider(
        base=base,
        store=store,
        name=name or f"agentdb/{arm}",
        include_failures=include_failures,
        k=k,
    )


Connector = Callable[[str | None], Connection]
"""How the factory opens its store. Injected so the arm is testable without a database."""


async def clickhouse_memory_provider(
    *,
    include_failures: bool = False,
    name: str | None = None,
    k: int | None = None,
    dsn: str | None = None,
    level: str = GroundingLevel.LAYOUT.value,
    plan_review: bool = True,
    connector: Connector = connect,
    importer: Importer = importlib.import_module,
) -> MemoryContextProvider:
    """Build ``A4_memory`` or ``A5_negmemory`` against a live ClickHouse and store.

    Named by dotted path from ``eval/providers.yaml``, like every other Family A
    arm, so the harness resolves it at run time and still imports nothing from
    agentdb (SPEC §4.1.6).

    ``plan_review`` defaults to true because these arms are A3 plus memory: an
    A4 that quietly dropped the plan turn would report the cost of exemplars
    against the wrong baseline.
    """
    arm = name or ("agentdb/A5_negmemory" if include_failures else "agentdb/A4_memory")
    base = await clickhouse_provider(
        level=level,
        plan_review=plan_review,
        name=f"{arm}/base",
        importer=importer,
    )
    store = ExemplarStore(connector(dsn))
    store.ensure_schema()
    return build_memory_provider(
        base=base,
        store=store,
        include_failures=include_failures,
        name=arm,
        k=k,
    )


async def databricks_memory_provider(
    *,
    include_failures: bool = False,
    name: str | None = None,
    k: int | None = None,
    dsn: str | None = None,
    level: str = GroundingLevel.LAYOUT.value,
    plan_review: bool = True,
    connector: Connector = connect,
    importer: Importer = importlib.import_module,
) -> MemoryContextProvider:
    """``A4_memory`` / ``A5_negmemory`` against a Databricks warehouse.

    The store is the same Postgres either way — it is agentdb's own state, not
    the engine's — but the exemplars in it are keyed by engine, so a warehouse
    run never retrieves a ClickHouse query and vice versa. That separation is
    what makes the cross-engine memory arms comparable rather than contaminated.
    """
    arm = name or ("agentdb/A5_negmemory" if include_failures else "agentdb/A4_memory")
    base = await databricks_provider(
        level=level, plan_review=plan_review, name=f"{arm}/base", importer=importer
    )
    store = ExemplarStore(connector(dsn))
    store.ensure_schema()
    return build_memory_provider(
        base=base,
        store=store,
        include_failures=include_failures,
        name=arm,
        k=k,
    )
