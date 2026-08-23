"""Shadow validation and the reaper that survives it (SPEC §9.1.B, §9.2.F).

This is the feature that can do real damage if it is sloppy: it writes, on a
warehouse that bills. So the tests are mostly about restraint — that nothing runs
unless the flag is set, that the table is dropped even when the plan read fails,
that a name is recognisable enough to reap, and that a second process reaping
does not pull a table out from under a validation that is still using it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest

from agentdb.adapters import (
    Engine,
    ExplainMode,
    PhysicalLayout,
    RawPlan,
    RelationRef,
)
from agentdb.config import Config
from agentdb.core.advisor import (
    MARKER,
    Confidence,
    EffectEstimate,
    Evidence,
    Kind,
    Measurement,
    Recommendation,
    ShadowError,
    ShadowValidator,
    measured,
    reap_orphans,
)

HITS = RelationRef(namespace="agentdb", name="hits")
LINEITEM = RelationRef(catalog="samples", namespace="tpch", name="lineitem")

PRUNED_PLAN = json.dumps(
    {
        "Plan": {
            "Node Type": "ReadFromMergeTree",
            "Description": "agentdb.hits",
            "Indexes": [{"Type": "PrimaryKey", "Initial Granules": 1_000, "Selected Granules": 40}],
        }
    }
)

ALLOW = Config(allow_shadow=True)


@dataclass
class FakeRunner:
    """A write channel that records what it was asked to run."""

    engine: Engine = "clickhouse"
    tables: dict[str, list[str]] = field(default_factory=dict)
    statements: list[str] = field(default_factory=list)
    plan: str = PRUNED_PLAN
    explain_error: Exception | None = None

    async def run(self, sql: str) -> None:
        self.statements.append(sql)

    async def explain(self, sql: str, mode: ExplainMode) -> RawPlan:
        if self.explain_error is not None:
            raise self.explain_error
        return RawPlan(engine=self.engine, mode=mode, sql=sql, payload=self.plan)

    async def list_tables(self, namespace: str) -> Sequence[str]:
        return self.tables.get(namespace, [])


def layout(**overrides: object) -> PhysicalLayout:
    fields: dict[str, object] = {
        "engine": "clickhouse",
        "ref": HITS,
        "create_statement": "",
        "table_engine": "MergeTree",
        "order_by": ("CounterID",),
        "sampling_key": "intHash32(UserID)",
        "approx_rows": 100_000_000,
    }
    fields.update(overrides)
    return PhysicalLayout(**fields)  # type: ignore[arg-type]  # test factory, keyed by field name


def validator(runner: FakeRunner, **kwargs: object) -> ShadowValidator:
    return ShadowValidator(runner=runner, config=ALLOW, token="tok123", **kwargs)  # type: ignore[arg-type]  # kwargs


# --------------------------------------------------------------------------
# refusing to run
# --------------------------------------------------------------------------


def test_shadow_validation_does_nothing_unless_it_was_switched_on() -> None:
    """Shadow tables cost money on a warehouse; the flag is the whole safeguard."""
    with pytest.raises(ShadowError, match="AGENTDB_ALLOW_SHADOW"):
        ShadowValidator(runner=FakeRunner(), config=Config())


def test_databricks_validation_refuses_without_a_scratch_schema() -> None:
    """Never the catalog under measurement, and never samples (SPEC §13.3)."""
    with pytest.raises(ShadowError, match="AGENTDB_DBX_SCRATCH_SCHEMA"):
        ShadowValidator(runner=FakeRunner(engine="databricks"), config=ALLOW)


def test_a_configured_scratch_schema_is_where_the_shadow_lands() -> None:
    runner = FakeRunner(engine="databricks")

    plan = validator(runner, scratch_schema="scratch.agentdb").plan_for(
        ref=LINEITEM,
        layout=layout(engine="databricks", ref=LINEITEM),
        probe_sql="SELECT count(*) FROM samples.tpch.lineitem",
        cluster_by=("l_shipdate",),
    )

    assert plan.shadow.startswith("scratch.agentdb.lineitem")
    assert "samples.tpch" not in plan.shadow


# --------------------------------------------------------------------------
# what it builds
# --------------------------------------------------------------------------


def test_the_clickhouse_shadow_samples_rather_than_copying_the_table() -> None:
    plan = validator(FakeRunner()).plan_for(
        ref=HITS,
        layout=layout(),
        probe_sql="SELECT count() FROM agentdb.hits WHERE UserID = 7",
        order_by=("UserID",),
        index_ddl="ALTER TABLE agentdb.hits ADD INDEX idx TYPE minmax GRANULARITY 4",
    )

    assert "CREATE TABLE agentdb.hits__agentdb_shadow_tok123" in plan.setup[0]
    assert "ORDER BY (UserID)" in plan.setup[0]
    assert "SAMPLE 0.01" in plan.setup[0] or "SAMPLE 0." in plan.setup[0]
    assert plan.setup[1].startswith("ALTER TABLE agentdb.hits__agentdb_shadow_tok123")


def test_a_table_with_no_sampling_key_is_bounded_by_a_limit_instead() -> None:
    """SAMPLE needs a sampling key; a LIMIT is not a random sample and is not called one."""
    plan = validator(FakeRunner()).plan_for(
        ref=HITS,
        layout=layout(sampling_key=None),
        probe_sql="SELECT count() FROM agentdb.hits",
    )

    assert "SAMPLE" not in plan.setup[0]
    assert "LIMIT" in plan.setup[0]


def test_the_databricks_shadow_analyzes_so_the_plans_are_comparable() -> None:
    plan = validator(FakeRunner(engine="databricks"), scratch_schema="scratch.agentdb").plan_for(
        ref=LINEITEM,
        layout=layout(engine="databricks", ref=LINEITEM),
        probe_sql="SELECT count(*) FROM samples.tpch.lineitem",
        cluster_by=("l_shipdate",),
        stats_columns=("l_shipdate", "l_orderkey"),
    )

    assert "TABLESAMPLE" in plan.setup[0]
    assert "CLUSTER BY (l_shipdate)" in plan.setup[0]
    assert "delta.dataSkippingStatsColumns" in plan.setup[0]
    assert plan.setup[1].startswith("OPTIMIZE")
    assert plan.setup[2].startswith("ANALYZE TABLE")


def test_the_probe_is_the_candidate_query_pointed_at_the_shadow() -> None:
    plan = validator(FakeRunner()).plan_for(
        ref=HITS, layout=layout(), probe_sql="SELECT count() FROM agentdb.hits WHERE UserID = 7"
    )

    assert plan.probe == (
        "SELECT count() FROM agentdb.hits__agentdb_shadow_tok123 WHERE UserID = 7"
    )


# --------------------------------------------------------------------------
# measuring, and cleaning up
# --------------------------------------------------------------------------


async def test_a_measurement_reads_the_pruning_the_shadow_plan_shows() -> None:
    runner = FakeRunner()

    measurement = await validator(runner).measure(
        ref=HITS,
        layout=layout(),
        probe_sql="SELECT count() FROM agentdb.hits WHERE UserID = 7",
        baseline=1.0,
    )

    assert measurement.before == 1.0
    assert measurement.after == pytest.approx(0.04)
    assert measurement.unit == "granule"
    assert "shadow table holding" in measurement.method
    assert "absolute counts do not" in measurement.method


async def test_the_shadow_table_is_dropped_when_everything_works() -> None:
    runner = FakeRunner()

    await validator(runner).measure(
        ref=HITS, layout=layout(), probe_sql="SELECT count() FROM agentdb.hits", baseline=None
    )

    assert runner.statements[-1].startswith("DROP TABLE IF EXISTS agentdb.hits__agentdb_shadow")


async def test_the_shadow_table_is_dropped_when_the_plan_read_fails() -> None:
    """A rejected candidate must not leave a copy of a hundred-million-row table behind."""
    runner = FakeRunner(explain_error=RuntimeError("EXPLAIN refused"))

    with pytest.raises(RuntimeError, match="EXPLAIN refused"):
        await validator(runner).measure(
            ref=HITS, layout=layout(), probe_sql="SELECT count() FROM agentdb.hits", baseline=None
        )

    assert any(statement.startswith("DROP TABLE") for statement in runner.statements)


async def test_a_databricks_measurement_reports_the_percentage_it_sampled() -> None:
    """Delta samples by percent and ClickHouse by fraction; the report says which."""
    runner = FakeRunner(
        engine="databricks",
        plan=(
            "== Physical Plan ==\n"
            "PhotonScan parquet scratch.agentdb.lineitem__agentdb_shadow_tok123 (1)\n"
            "\n"
            "(1) PhotonScan parquet scratch.agentdb.lineitem__agentdb_shadow_tok123\n"
            "Output [1]: [l_shipdate#2]\n"
            "PushedFilters: [IsNotNull(l_shipdate)]\n"
            "number of files read: 4\n"
            "number of files pruned: 16\n"
        ),
    )

    measurement = await validator(runner, scratch_schema="scratch.agentdb").measure(
        ref=LINEITEM,
        layout=layout(engine="databricks", ref=LINEITEM),
        probe_sql="SELECT count(*) FROM samples.tpch.lineitem",
        baseline=1.0,
        cluster_by=("l_shipdate",),
    )

    assert measurement.sample_fraction == pytest.approx(0.01)
    assert measurement.unit in {"file", "granule"}


# --------------------------------------------------------------------------
# the reaper
# --------------------------------------------------------------------------


async def test_the_reaper_drops_what_a_killed_process_left_behind() -> None:
    """A finally does not run when the process is killed. The marker is why this works."""
    runner = FakeRunner(
        tables={"agentdb": ["hits", f"hits{MARKER}_dead01", f"visits{MARKER}_dead02"]}
    )

    dropped = await reap_orphans(runner, ["agentdb"])

    assert dropped == (f"agentdb.hits{MARKER}_dead01", f"agentdb.visits{MARKER}_dead02")
    assert all(statement.startswith("DROP TABLE IF EXISTS") for statement in runner.statements)


async def test_the_reaper_never_touches_a_table_it_did_not_create() -> None:
    runner = FakeRunner(tables={"agentdb": ["hits", "hits_shadow", "shadow_hits"]})

    assert await reap_orphans(runner, ["agentdb"]) == ()
    assert runner.statements == []


async def test_a_running_validation_is_not_reaped_out_from_under_itself() -> None:
    """Two processes, one tidy-up: the second must not break the first."""
    live = f"agentdb.hits{MARKER}_live01"
    runner = FakeRunner(tables={"agentdb": [f"hits{MARKER}_live01", f"hits{MARKER}_dead01"]})

    dropped = await reap_orphans(runner, ["agentdb"], keep=[live])

    assert dropped == (f"agentdb.hits{MARKER}_dead01",)


async def test_the_reaper_sweeps_every_namespace_it_was_given() -> None:
    runner = FakeRunner(tables={"agentdb": [f"hits{MARKER}_a"], "tpch": [f"lineitem{MARKER}_b"]})

    dropped = await reap_orphans(runner, ["agentdb", "tpch"])

    assert dropped == (f"agentdb.hits{MARKER}_a", f"tpch.lineitem{MARKER}_b")


# --------------------------------------------------------------------------
# what a measurement does to a recommendation
# --------------------------------------------------------------------------


def test_measurement_is_the_only_thing_that_promotes_a_recommendation() -> None:
    estimated = Recommendation(
        kind=Kind.SKIP_INDEX,
        relation=HITS,
        rationale="UserID is filtered and is not in the sort key",
        evidence=Evidence(source="profile", distinct_counts=(("UserID", 17_000_000),)),
        expected_effect=EffectEstimate(
            metric="granules_read", before=1.0, after=0.0001, method="1 / approx_distinct"
        ),
        confidence=Confidence.ESTIMATED,
        ddl="ALTER TABLE agentdb.hits ADD INDEX ...",
    )

    promoted = measured(estimated, _measurement_of(before=1.0, after=0.04))

    assert promoted.confidence is Confidence.MEASURED
    assert promoted.evidence.source == "shadow"
    assert promoted.expected_effect.after == 0.04
    assert promoted.ddl == estimated.ddl, "the migration is unchanged by measuring it"
    assert estimated.confidence is Confidence.ESTIMATED, "recommendations are never mutated"


def _measurement_of(*, before: float, after: float) -> Measurement:
    return Measurement(
        before=before,
        after=after,
        sample_fraction=0.01,
        unit="granule",
        method="granule pruning read from the plan of a shadow table holding 1.0% of the rows",
    )
