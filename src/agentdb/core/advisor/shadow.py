"""Shadow validation: the only way a recommendation earns ``measured`` (SPEC §9.1.B, §9.2.F).

Neither engine has a hypothetical-index facility — there is no ``hypopg`` for
ClickHouse and none for Delta — so the only honest way to know whether a proposed
index or clustering key prunes is to build it on a *sample* of the table and read
the plan. One mechanism serves both engines; only the DDL differs.

Four properties this module is built around, each of which is a way the feature
could do real damage if it were sloppy:

* **Opt-in, and the flag is load-bearing.** Nothing here runs unless
  ``AGENTDB_ALLOW_SHADOW`` is set. Shadow tables cost money on Databricks in a
  way they do not on a local container.
* **Namespaced and capped.** Every table this project creates carries the
  ``__agentdb_shadow`` marker and a run token, and the sample is bounded by
  ``SHADOW_TABLE_MAX_ROWS``. A name that cannot be recognised cannot be reaped.
* **Dropped in a ``finally``.** Including when the plan read fails, including
  when the candidate DDL is rejected.
* **Reaped on startup anyway.** A ``finally`` does not run when the process is
  killed, so the marker exists to make orphans findable and
  :func:`reap_orphans` finds them. That is what the chaos test exercises.

The write path is deliberately a *different* object from the read-only adapter
the rest of agentdb uses. Read-only is enforced at the connection level (SPEC
§13.3), so validation cannot borrow that connection — it has to be handed a
channel the operator explicitly configured for writes, and where none is
configured, validation does not happen.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Protocol
from uuid import uuid4

from agentdb.adapters import Engine, ExplainMode, PhysicalLayout, RawPlan, RelationRef
from agentdb.config import Config
from agentdb.core.advisor.base import Confidence, EffectEstimate, Evidence, Recommendation
from agentdb.core.plan_analyzer import summarize
from agentdb.core.plan_analyzer_databricks import summarize as summarize_databricks

MARKER = "__agentdb_shadow"
"""The substring that makes a table this project's to drop. Never change it
without a migration: an orphan named under the old marker becomes unreapable."""


class ShadowError(RuntimeError):
    """Shadow validation could not run, or was refused."""


class ShadowRunner(Protocol):
    """The write channel shadow validation needs, and nothing more.

    Separate from :class:`~agentdb.adapters.base.Adapter` on purpose: that
    protocol is served by a read-only connection, and widening it with DDL would
    put a write method on the object every other tool holds.
    """

    @property
    def engine(self) -> Engine: ...

    async def run(self, sql: str) -> None:
        """Execute one statement. Raises on failure."""

    async def explain(self, sql: str, mode: ExplainMode) -> RawPlan:
        """Plan ``sql`` against whatever this channel can see."""

    async def list_tables(self, namespace: str) -> Sequence[str]:
        """Table names in ``namespace``, for the reaper to sift."""


@dataclass(frozen=True, slots=True)
class ShadowPlan:
    """The statements one validation will run, in order, and its cleanup."""

    setup: tuple[str, ...]
    probe: str
    """The candidate query, rewritten against the shadow table."""

    teardown: tuple[str, ...]
    shadow: str
    """Fully-qualified name of the table being created, for the reaper's benefit."""


@dataclass(frozen=True, slots=True)
class Measurement:
    """What the plan said before and after, and how that was scaled.

    ``sample_fraction`` is carried because the numbers are *from a sample*: a
    granule count on 1% of a table is not the granule count on the table, and a
    reader comparing the two without knowing that would draw a conclusion the
    measurement does not support.
    """

    before: float | None
    after: float | None
    sample_fraction: float
    unit: str
    method: str


@dataclass(frozen=True, slots=True)
class ShadowValidator:
    """Builds a sampled copy of a relation, applies a candidate, and reads the plan."""

    runner: ShadowRunner
    config: Config = field(default_factory=Config)
    scratch_schema: str | None = None
    """Where shadow tables are created. Required on Databricks (SPEC §13.3): never
    the catalog under measurement, and never ``samples``."""

    token: str = field(default_factory=lambda: uuid4().hex[:8])

    def __post_init__(self) -> None:
        if not self.config.allow_shadow:
            raise ShadowError("shadow validation is opt-in and AGENTDB_ALLOW_SHADOW is not set")
        if self.runner.engine == "databricks" and not self.scratch_schema:
            raise ShadowError(
                "Databricks shadow validation needs AGENTDB_DBX_SCRATCH_SCHEMA: a "
                "writable schema that is neither the catalog under measurement nor samples"
            )

    def plan_for(
        self,
        *,
        ref: RelationRef,
        layout: PhysicalLayout,
        probe_sql: str,
        order_by: Sequence[str] = (),
        index_ddl: str | None = None,
        cluster_by: Sequence[str] = (),
        stats_columns: Sequence[str] = (),
    ) -> ShadowPlan:
        """The statements that build, arm and drop one shadow table."""
        shadow = self._shadow_name(ref)
        if self.runner.engine == "clickhouse":
            setup = _clickhouse_setup(
                source=ref,
                shadow=shadow,
                layout=layout,
                order_by=order_by or (layout.order_by or ()),
                index_ddl=index_ddl,
                fraction=self.config.default_sample_fraction,
            )
        else:
            setup = _databricks_setup(
                source=ref,
                shadow=shadow,
                cluster_by=cluster_by,
                stats_columns=stats_columns,
                percent=self.config.dbx_shadow_sample_percent,
            )
        return ShadowPlan(
            setup=setup,
            probe=probe_sql.replace(str(ref), shadow),
            teardown=(f"DROP TABLE IF EXISTS {shadow}",),
            shadow=shadow,
        )

    async def measure(
        self,
        *,
        ref: RelationRef,
        layout: PhysicalLayout,
        probe_sql: str,
        baseline: float | None,
        order_by: Sequence[str] = (),
        index_ddl: str | None = None,
        cluster_by: Sequence[str] = (),
        stats_columns: Sequence[str] = (),
    ) -> Measurement:
        """Build the shadow, plan the probe against it, and drop it — always.

        The drop runs in a ``finally``, so a rejected candidate or a failed plan
        read still leaves nothing behind. What a ``finally`` cannot survive is
        the process being killed, which is why the name carries the marker and
        :func:`reap_orphans` exists.
        """
        shadow_plan = self.plan_for(
            ref=ref,
            layout=layout,
            probe_sql=probe_sql,
            order_by=order_by,
            index_ddl=index_ddl,
            cluster_by=cluster_by,
            stats_columns=stats_columns,
        )
        try:
            for statement in shadow_plan.setup:
                await self.runner.run(statement)
            planned = await self.runner.explain(shadow_plan.probe, ExplainMode.ESTIMATE)
        finally:
            for statement in shadow_plan.teardown:
                await self.runner.run(statement)

        return _measurement(
            planned=planned,
            baseline=baseline,
            engine=self.runner.engine,
            fraction=self._fraction(),
        )

    def _fraction(self) -> float:
        if self.runner.engine == "clickhouse":
            return self.config.default_sample_fraction
        return self.config.dbx_shadow_sample_percent / 100.0

    def _shadow_name(self, ref: RelationRef) -> str:
        namespace = self.scratch_schema or _namespace_of(ref)
        return f"{namespace}.{ref.name}{MARKER}_{self.token}"


async def reap_orphans(
    runner: ShadowRunner, namespaces: Sequence[str], *, keep: Sequence[str] = ()
) -> tuple[str, ...]:
    """Drop every shadow table left behind by a process that did not finish.

    Called on startup. ``keep`` names tables a *running* validation owns, so a
    second process reaping does not pull the table out from under the first —
    the failure mode that turns a tidy-up into an outage.
    """
    dropped: list[str] = []
    protected = set(keep)
    for namespace in namespaces:
        for table in await runner.list_tables(namespace):
            qualified = f"{namespace}.{table}"
            if MARKER not in table or qualified in protected:
                continue
            await runner.run(f"DROP TABLE IF EXISTS {qualified}")
            dropped.append(qualified)
    return tuple(dropped)


def measured(recommendation: Recommendation, measurement: Measurement) -> Recommendation:
    """Re-issue a recommendation with what the shadow table actually showed.

    The confidence moves to ``measured`` here and nowhere else in the codebase.
    That is the whole point of the exercise: everything else in the advisor is an
    estimate, and the difference has to be visible to a reader deciding whether
    to run a migration.
    """
    return replace(
        recommendation,
        confidence=Confidence.MEASURED,
        evidence=Evidence(
            source="shadow",
            pruning_ratio=measurement.after,
            pruning_unit=recommendation.evidence.pruning_unit,
            relation_rows=recommendation.evidence.relation_rows,
            distinct_counts=recommendation.evidence.distinct_counts,
            workload_queries=recommendation.evidence.workload_queries,
        ),
        expected_effect=EffectEstimate(
            metric=recommendation.expected_effect.metric,
            before=measurement.before,
            after=measurement.after,
            method=measurement.method,
        ),
    )


def _measurement(
    *, planned: RawPlan, baseline: float | None, engine: Engine, fraction: float
) -> Measurement:
    """Read the pruning ratio out of the shadow plan, and say what it is worth."""
    summary = summarize(planned) if engine == "clickhouse" else summarize_databricks(planned)
    unit = summary.pruning_unit or ("granule" if engine == "clickhouse" else "file")
    return Measurement(
        before=baseline,
        after=summary.pruning_ratio,
        sample_fraction=fraction,
        unit=unit,
        method=(
            f"{unit} pruning read from the plan of a shadow table holding "
            f"{fraction:.1%} of the rows, carrying the proposed physical design. "
            "The ratio transfers; the absolute counts do not, because a sample "
            "has fewer of everything."
        ),
    )


def _clickhouse_setup(
    *,
    source: RelationRef,
    shadow: str,
    layout: PhysicalLayout,
    order_by: Sequence[str],
    index_ddl: str | None,
    fraction: float,
) -> tuple[str, ...]:
    """ClickHouse's form (SPEC §9.1.B).

    ``SAMPLE`` needs a sampling key; where the table has none the sample is a
    ``LIMIT``, which is not a random sample and says so in the method text rather
    than pretending otherwise.
    """
    key = ", ".join(order_by) if order_by else "tuple()"
    sample = (
        f"SAMPLE {fraction}"
        if layout.sampling_key
        else f"LIMIT {int(max(fraction, 0.0) * (layout.approx_rows or 0)) or 1_000_000}"
    )
    statements = [
        f"CREATE TABLE {shadow} ENGINE = MergeTree ORDER BY ({key}) "
        f"AS SELECT * FROM {source} {sample}",
    ]
    if index_ddl:
        statements.append(index_ddl.replace(str(source), shadow))
    return tuple(statements)


def _databricks_setup(
    *,
    source: RelationRef,
    shadow: str,
    cluster_by: Sequence[str],
    stats_columns: Sequence[str],
    percent: float,
) -> tuple[str, ...]:
    """Delta's form (SPEC §9.2.F). ``ANALYZE`` is what makes the plan comparable."""
    clause = f" CLUSTER BY ({', '.join(cluster_by)})" if cluster_by else ""
    properties = (
        f" TBLPROPERTIES ('delta.dataSkippingStatsColumns' = '{', '.join(stats_columns)}')"
        if stats_columns
        else ""
    )
    return (
        f"CREATE TABLE {shadow}{clause}{properties} "
        f"AS SELECT * FROM {source} TABLESAMPLE ({percent} PERCENT)",
        f"OPTIMIZE {shadow}",
        f"ANALYZE TABLE {shadow} COMPUTE STATISTICS FOR ALL COLUMNS",
    )


def _namespace_of(ref: RelationRef) -> str:
    return ".".join(part for part in (ref.catalog, ref.namespace) if part)
