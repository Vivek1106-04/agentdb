"""In-memory doubles used across the unit suite.

A fake adapter is not a convenience here, it is the point: core must be provable
against the :class:`~agentdb.adapters.Adapter` protocol alone, with no engine
running. Anything core can only be tested against a live ClickHouse is a leak.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from agentdb.adapters import (
    Adapter,
    BaseAdapter,
    Capability,
    ColumnDef,
    ColumnProfile,
    DialectRules,
    Engine,
    ExplainMode,
    Limits,
    PhysicalLayout,
    RawPlan,
    Relation,
    RelationDetail,
    RelationRef,
    ResultSet,
    SamplePolicy,
    TimeWindow,
    WorkloadEntry,
)

CLICKHOUSE_CAPABILITIES = frozenset(
    {
        Capability.ESTIMATE_ONLY_PLAN,
        Capability.SKIP_INDEX,
        Capability.PROJECTION,
        Capability.SORT_KEY,
        Capability.WORKLOAD_LOG,
        Capability.COLUMN_STATS,
        Capability.SAMPLING,
    }
)

POSTGRES_CAPABILITIES = frozenset(
    {
        Capability.HYPOTHETICAL_INDEX,
        Capability.ANALYZE_PLAN,
        Capability.WORKLOAD_LOG,
        Capability.COLUMN_STATS,
    }
)


@dataclass
class FakeAdapter(BaseAdapter):
    """A fully scripted adapter. Every response is supplied by the test.

    Calls are recorded so a test can assert *what core asked the engine*, which
    is usually the interesting assertion — for example that column profiling
    went through a bounded sample policy rather than a bare scan.
    """

    engine: Engine = "clickhouse"
    capabilities: frozenset[Capability] = CLICKHOUSE_CAPABILITIES
    relations: tuple[Relation, ...] = ()
    details: dict[str, RelationDetail] = field(default_factory=dict)
    layouts: dict[str, PhysicalLayout] = field(default_factory=dict)
    profiles: dict[str, ColumnProfile] = field(default_factory=dict)
    plan: RawPlan | None = None
    result: ResultSet | None = None
    workload_entries: tuple[WorkloadEntry, ...] = ()
    rules: DialectRules | None = None
    calls: list[tuple[str, object]] = field(default_factory=list)

    async def list_relations(self, namespace: str | None = None) -> list[Relation]:
        self.calls.append(("list_relations", namespace))
        if namespace is None:
            return list(self.relations)
        return [r for r in self.relations if r.ref.namespace == namespace]

    async def describe_relation(self, ref: RelationRef) -> RelationDetail:
        self.calls.append(("describe_relation", ref))
        return self.details[str(ref)]

    async def physical_layout(self, ref: RelationRef) -> PhysicalLayout:
        self.calls.append(("physical_layout", ref))
        return self.layouts[str(ref)]

    async def column_profile(
        self, ref: RelationRef, columns: list[str], sample: SamplePolicy
    ) -> list[ColumnProfile]:
        self.calls.append(("column_profile", (ref, tuple(columns), sample)))
        return [self.profiles[name] for name in columns if name in self.profiles]

    async def explain(self, sql: str, mode: ExplainMode) -> RawPlan:
        self.calls.append(("explain", (sql, mode)))
        if self.plan is None:
            raise AssertionError("FakeAdapter.plan was not scripted")
        return replace(self.plan, sql=sql, mode=mode)

    async def execute(self, sql: str, limits: Limits) -> ResultSet:
        self.calls.append(("execute", (sql, limits)))
        if self.result is None:
            raise AssertionError("FakeAdapter.result was not scripted")
        return self.result

    async def workload(self, window: TimeWindow, top_n: int) -> list[WorkloadEntry]:
        self.calls.append(("workload", (window, top_n)))
        return list(self.workload_entries[:top_n])

    async def dialect_rules(self) -> DialectRules:
        self.calls.append(("dialect_rules", None))
        if self.rules is None:
            return DialectRules(engine=self.engine, version="0.0", identifier_quote="`")
        return self.rules

    def calls_named(self, name: str) -> list[object]:
        """Arguments of every recorded call to ``name``, in order."""
        return [args for called, args in self.calls if called == name]


def clickhouse_hits_fixture() -> FakeAdapter:
    """A ClickBench-shaped adapter: one wide MergeTree table with a real sort key."""
    ref = RelationRef(namespace="agentdb", name="hits")
    create = (
        "CREATE TABLE agentdb.hits (CounterID UInt32, EventDate Date, UserID UInt64, "
        "SearchEngineID UInt16, URL String) ENGINE = MergeTree "
        "PARTITION BY toYYYYMM(EventDate) ORDER BY (CounterID, EventDate, UserID)"
    )
    detail = RelationDetail(
        ref=ref,
        columns=(
            ColumnDef(name="CounterID", data_type="UInt32", is_nullable=False),
            ColumnDef(name="EventDate", data_type="Date", is_nullable=False),
            ColumnDef(name="UserID", data_type="UInt64", is_nullable=False),
            ColumnDef(name="SearchEngineID", data_type="UInt16", is_nullable=False),
            ColumnDef(name="URL", data_type="String", is_nullable=False),
        ),
        create_statement=create,
    )
    layout = PhysicalLayout(
        engine="clickhouse",
        ref=ref,
        create_statement=create,
        table_engine="MergeTree",
        order_by=("CounterID", "EventDate", "UserID"),
        partition_by=("toYYYYMM(EventDate)",),
        primary_key=("CounterID", "EventDate", "UserID"),
        approx_rows=99_997_497,
        on_disk_bytes=14_779_976_446,
    )
    return FakeAdapter(
        relations=(
            Relation(
                ref=ref,
                kind="table",
                engine_type="MergeTree",
                approx_rows=99_997_497,
                on_disk_bytes=14_779_976_446,
            ),
        ),
        details={str(ref): detail},
        layouts={str(ref): layout},
        profiles={
            "SearchEngineID": ColumnProfile(
                name="SearchEngineID",
                data_type="UInt16",
                sample_method="sample",
                sampled_rows=999_974,
                approx_distinct=42,
                null_ratio=0.0,
                top_values=(("2", 500_000), ("3", 200_000)),
            ),
            "UserID": ColumnProfile(
                name="UserID",
                data_type="UInt64",
                sample_method="sample",
                sampled_rows=999_974,
                approx_distinct=17_630_976,
                null_ratio=0.0,
            ),
        },
        rules=DialectRules(
            engine="clickhouse",
            version="25.9",
            identifier_quote="`",
            reserved_words=frozenset({"SELECT", "ORDER", "SAMPLE"}),
            quirks=("EXPLAIN is estimate-only; actual rows come from system.query_log",),
        ),
    )


_ADAPTER_PROTOCOL_CHECK: Adapter = FakeAdapter()
"""Import-time proof that the fake satisfies the protocol mypy checks against."""
