"""agentdb's grounding, exposed to a benchmark harness that does not import it.

``agenteval`` scores context providers structurally: anything with ``name``,
``version``, ``fingerprint`` and an async ``context(namespace, question)`` can be
an arm. This module is agentdb's implementation of that shape, and it imports
nothing from the harness — the independence runs in both directions, which is
what lets the same harness score agentdb and its competitors on equal terms
(SPEC §4.1.6, §11.3).

The provider is deliberately thin. All it does is pick a
:class:`~agentdb.core.GroundingLevel` and render what the builder assembled; if
an arm scores well, the credit belongs to the facts the adapter read, and a
reader can see there is no prompt-side cleverness hiding in between.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import dataclass, field

from agentdb.adapters.base import Adapter
from agentdb.adapters.clickhouse import ClickHouseAdapter
from agentdb.adapters.clickhouse_client import ClickHouseTarget, Importer, build_client
from agentdb.config import Config
from agentdb.core import ContextBuilder, GroundedContext, GroundingLevel, PlanExplainer

VERSION = "1.0"
"""Bumped whenever the assembled payload changes shape, because that changes the number."""


@dataclass(frozen=True, slots=True)
class GroundedContextProvider:
    """Serves one grounding level over one adapter.

    ``question`` is accepted and currently unused: the payload is per-namespace,
    not per-question. Question-aware selection is a later arm (SPEC §11.3, A4),
    and taking the argument now keeps that a change of behaviour rather than a
    change of interface.
    """

    builder: ContextBuilder
    level: GroundingLevel
    name: str
    version: str = VERSION
    explainer: PlanExplainer | None = None
    """Present when the arm may show the model its plan (``A3`` and above)."""

    _cache: dict[str, GroundedContext] = field(default_factory=dict, repr=False, compare=False)

    @property
    def fingerprint(self) -> str:
        """Hash of everything that decides what this provider returns."""
        config = self.builder.config
        return fingerprint_config(
            {
                "provider": self.name,
                "version": self.version,
                "level": self.level.value,
                "engine": self.builder.adapter.engine,
                "sample_fraction": config.default_sample_fraction,
                "profile_max_rows": config.profile_max_rows,
                "max_profiled_columns": config.max_profiled_columns,
                "plan_review": self.explainer is not None,
            }
        )

    async def context(self, *, namespace: str, question: str) -> str:  # noqa: ARG002
        """The rendered payload for ``namespace``, built once per namespace per run."""
        return (await self.build(namespace)).render()

    async def build(self, namespace: str) -> GroundedContext:
        """The assembled context object, built once per namespace per run.

        Caching is not an optimization here so much as a fairness property: every
        task in a suite must see byte-identical grounding, and rebuilding from a
        live server per task would let a merge or a background insert change the
        payload halfway through an arm.

        The object rather than its rendering, because the arms above this one
        need the facts as well as the text — the memory arm fingerprints the
        schema it was built against (SPEC §10.3).
        """
        cached = self._cache.get(namespace)
        if cached is None:
            cached = await self.builder.build(namespace, self.level)
            self._cache[namespace] = cached
        return cached

    async def aclose(self) -> None:
        """Close the engine connection this provider opened.

        A benchmark builds one provider per arm and runs hundreds of tasks
        through it, so the connection outlives every task and nothing else is in
        a position to release it.
        """
        client = getattr(self.builder.adapter, "client", None)
        closer = getattr(client, "close", None)
        if closer is not None:
            await closer()

    async def explain_plan(self, *, sql: str, namespace: str) -> str | None:
        """What the engine would do with ``sql``, or ``None`` when nothing is wrong.

        Silence is the useful default. A review that always says something
        teaches a model to skim it, and the arm then measures politeness rather
        than plan-awareness — so a plan with no warnings produces no turn at all.
        """
        if self.explainer is None:
            return None
        summary = await self.explainer.explain(sql, namespace)
        return summary.render() if summary.warnings else None


async def clickhouse_provider(
    *,
    level: str = GroundingLevel.LAYOUT.value,
    name: str | None = None,
    plan_review: bool = False,
    host: str | None = None,
    port: int | None = None,
    username: str | None = None,
    password: str | None = None,
    database: str | None = None,
    importer: Importer = importlib.import_module,
) -> GroundedContextProvider:
    """Build a provider against a live ClickHouse, for a harness to call by dotted path.

    Every argument is optional and falls back to ``AGENTDB_CLICKHOUSE_*``, so a
    benchmark config can name the provider and nothing else.
    """
    env_target = ClickHouseTarget.from_env()
    target = ClickHouseTarget(
        host=host if host is not None else env_target.host,
        port=port if port is not None else env_target.port,
        username=username if username is not None else env_target.username,
        password=password if password is not None else env_target.password,
        database=database if database is not None else env_target.database,
    )
    client = await build_client(target, importer=importer)
    return build_provider(
        adapter=ClickHouseAdapter(client=client),
        level=level,
        name=name,
        plan_review=plan_review,
    )


def build_provider(
    *,
    adapter: Adapter,
    level: str = GroundingLevel.LAYOUT.value,
    name: str | None = None,
    plan_review: bool = False,
    config: Config | None = None,
) -> GroundedContextProvider:
    """Wrap ``adapter`` in a provider at ``level``, rejecting an unknown level by name."""
    resolved = GroundingLevel(level)
    effective = config or Config()
    return GroundedContextProvider(
        builder=ContextBuilder(adapter=adapter, config=effective),
        level=resolved,
        name=name or f"agentdb/{resolved.value}",
        explainer=PlanExplainer(adapter=adapter, config=effective) if plan_review else None,
    )


def fingerprint_config(config: dict[str, object]) -> str:
    """SHA-256 over the canonical JSON form, so the hash is stable across runs."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
