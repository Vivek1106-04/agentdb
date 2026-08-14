"""The ClickHouse adapter, exercised against a scripted client rather than a server.

The interesting assertions are about *what the adapter asked the engine* and
*how honestly it labelled the answer*: that a profile built from a sample says so,
that a missing ``ANALYZE`` is refused instead of approximated, and that every
statement carries the ``log_comment`` a reader needs to audit it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pytest

from agentdb.adapters import (
    Adapter,
    Capability,
    ExplainMode,
    Limits,
    RelationRef,
    SamplePolicy,
    TimeWindow,
    UnsupportedCapabilityError,
)
from agentdb.adapters.base import QuerySemanticError, QuerySyntaxError
from agentdb.adapters.clickhouse import LOG_COMMENT_PREFIX, ClickHouseAdapter

REF = RelationRef(namespace="agentdb", name="hits")

CREATE_STATEMENT = "CREATE TABLE agentdb.hits (`CounterID` UInt32) ENGINE = MergeTree"

TABLE_ROW = [
    "MergeTree",
    "CounterID, toDate(EventTime)",
    "toYYYYMM(EventDate)",
    "CounterID",
    "intHash32(UserID)",
    CREATE_STATEMENT,
    1_000_000,
    5_000_000,
]


@dataclass
class FakeResult:
    column_names: Sequence[str] = ()
    result_rows: Sequence[Sequence[Any]] = ()
    summary: Mapping[str, str] = field(default_factory=dict)


@dataclass
class FakeClient:
    """Answers by matching a fragment of the statement, and records every call.

    Routing on the statement rather than on call order keeps a test that only
    cares about ``physical_layout``'s output from having to script the four
    system-table reads in the right sequence.
    """

    routes: dict[str, FakeResult | Exception] = field(default_factory=dict)
    queries: list[str] = field(default_factory=list)
    parameters: list[Mapping[str, Any]] = field(default_factory=list)
    settings: list[Mapping[str, Any]] = field(default_factory=list)

    async def query(
        self,
        query: str,
        *,
        parameters: Mapping[str, Any],
        settings: Mapping[str, Any],
    ) -> FakeResult:
        self.queries.append(query)
        self.parameters.append(parameters)
        self.settings.append(settings)
        for fragment, outcome in self.routes.items():
            if fragment in query:
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome
        raise AssertionError(f"no scripted result for: {query}")


def _adapter(**routes: FakeResult | Exception) -> tuple[ClickHouseAdapter, FakeClient]:
    """An adapter over a scripted client, with the table lookup already answered."""
    scripted: dict[str, FakeResult | Exception] = {
        "create_table_query": FakeResult(result_rows=[TABLE_ROW])
    }
    scripted.update({_FRAGMENTS[key]: value for key, value in routes.items()})
    client = FakeClient(routes=scripted)
    return ClickHouseAdapter(client=client, turn_id=lambda: "turn0001"), client


_FRAGMENTS = {
    "table": "create_table_query",
    "listing": "ORDER BY name",
    "listing_all": "database NOT IN",
    "columns": "ORDER BY position",
    "footprint": "sum(data_compressed_bytes)",
    "skip_indexes": "system.data_skipping_indices",
    "projections": "system.projections",
    "workload": "system.query_log",
    "version": "version()",
    "profile": "uniqCombined64",
    "query": "SELECT",
}


# --------------------------------------------------------------------------
# contract
# --------------------------------------------------------------------------


def test_the_adapter_satisfies_the_adapter_protocol() -> None:
    adapter, _ = _adapter()

    assert isinstance(adapter, Adapter)
    assert adapter.engine == "clickhouse"


def test_analyze_is_absent_because_clickhouse_cannot_measure_a_plan() -> None:
    adapter, _ = _adapter()

    assert adapter.supports(Capability.ESTIMATE_ONLY_PLAN)
    assert not adapter.supports(Capability.COST_ANNOTATED_PLAN)
    assert not adapter.supports(Capability.CLUSTERING_KEY)


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------


async def test_list_relations_reads_one_database_when_a_namespace_is_given() -> None:
    adapter, client = _adapter(
        listing=FakeResult(
            result_rows=[
                ["agentdb", "hits", "MergeTree", 1_000_000, 5_000, "clickbench"],
                ["agentdb", "hits_view", "View", None, None, ""],
            ]
        )
    )

    relations = await adapter.list_relations("agentdb")

    assert client.parameters[0] == {"database": "agentdb"}
    assert [(str(r.ref), r.kind) for r in relations] == [
        ("agentdb.hits", "table"),
        ("agentdb.hits_view", "view"),
    ]
    assert relations[0].comment == "clickbench"
    assert relations[1].approx_rows is None
    assert relations[1].comment is None


async def test_list_relations_without_a_namespace_excludes_the_system_databases() -> None:
    adapter, client = _adapter(
        listing_all=FakeResult(result_rows=[["agentdb", "hits", "MergeTree", 1, 2, ""]])
    )

    relations = await adapter.list_relations()

    assert "'system'" in client.queries[0]
    assert relations[0].ref.namespace == "agentdb"


async def test_describe_relation_returns_columns_in_declaration_order() -> None:
    adapter, _ = _adapter(
        columns=FakeResult(
            result_rows=[
                ["CounterID", "UInt32", "", "", 100, 400],
                ["Title", "LowCardinality(Nullable(String))", "''", "page title", None, None],
            ]
        )
    )

    detail = await adapter.describe_relation(REF)

    assert detail.column_names == ("CounterID", "Title")
    assert detail.create_statement == CREATE_STATEMENT
    assert detail.columns[0].is_nullable is False
    assert detail.columns[0].compression_ratio == 4.0
    assert detail.columns[1].is_nullable is True
    assert detail.columns[1].default_expression == "''"
    assert detail.columns[1].comment == "page title"
    assert detail.columns[0].comment is None


async def test_a_missing_relation_is_a_semantic_error_naming_what_to_call_next() -> None:
    adapter, _ = _adapter(table=FakeResult(result_rows=[]))

    with pytest.raises(QuerySemanticError) as caught:
        await adapter.describe_relation(REF)

    assert "agentdb.hits" in str(caught.value)
    assert caught.value.suggestion is not None


async def test_dialect_rules_carry_the_connected_version_and_the_engine_quirks() -> None:
    adapter, _ = _adapter(version=FakeResult(result_rows=[["25.9.1.1"]]))

    rules = await adapter.dialect_rules()

    assert rules.version == "25.9.1.1"
    assert rules.quote_identifier("Order") == "`Order`"
    assert rules.needs_quoting("Order") is True
    assert any("estimate-only" in quirk for quirk in rules.quirks)


# --------------------------------------------------------------------------
# physical layout
# --------------------------------------------------------------------------


async def test_physical_layout_parses_the_keys_that_decide_granule_pruning() -> None:
    adapter, _ = _adapter(
        skip_indexes=FakeResult(
            result_rows=[["idx_url", "bloom_filter", "bloom_filter(0.01)", "URL", 4, 2_048]]
        ),
        projections=FakeResult(result_rows=[["by_user", "SELECT UserID, count()"]]),
        footprint=FakeResult(result_rows=[[1_000, 4_000]]),
    )

    layout = await adapter.physical_layout(REF)

    assert layout.order_by == ("CounterID", "toDate(EventTime)")
    assert layout.leading_sort_column == "CounterID"
    assert layout.partition_by == ("toYYYYMM(EventDate)",)
    assert layout.primary_key == ("CounterID",)
    assert layout.is_sampleable is True
    assert layout.compression_ratio == 4.0
    assert layout.skip_indexes[0].index_type == "bloom_filter"
    assert layout.skip_indexes[0].granularity == 4
    assert layout.projections[0].name == "by_user"
    assert layout.approx_rows == 1_000_000


async def test_layout_reports_an_unmeasurable_footprint_as_unknown_not_as_a_ratio_of_one() -> None:
    adapter, _ = _adapter(
        skip_indexes=FakeResult(result_rows=[]),
        projections=FakeResult(result_rows=[]),
        footprint=FakeResult(result_rows=[]),
    )

    layout = await adapter.physical_layout(REF)

    assert layout.compression_ratio is None
    assert layout.skip_indexes == ()


async def test_layout_tolerates_a_table_whose_columns_report_zero_compressed_bytes() -> None:
    adapter, _ = _adapter(
        skip_indexes=FakeResult(result_rows=[]),
        projections=FakeResult(result_rows=[]),
        footprint=FakeResult(result_rows=[[0, 0]]),
    )

    layout = await adapter.physical_layout(REF)

    assert layout.compression_ratio is None


# --------------------------------------------------------------------------
# column profiling
# --------------------------------------------------------------------------

SAMPLE = SamplePolicy(fraction=0.01, max_rows=100_000, timeout_s=10)

_COLUMNS = FakeResult(result_rows=[["CounterID", "UInt32", "", "", 100, 400]])
_PROFILE_ROW = FakeResult(
    result_rows=[[42, 0.25, "1", "9999", [("1", 10, 0), ("2", 5, 0)], 10_000]]
)


async def test_a_sampled_profile_says_it_was_sampled_and_how_many_rows_it_read() -> None:
    adapter, client = _adapter(
        columns=_COLUMNS,
        skip_indexes=FakeResult(result_rows=[]),
        projections=FakeResult(result_rows=[]),
        footprint=FakeResult(result_rows=[[1, 2]]),
        profile=_PROFILE_ROW,
    )

    profiles = await adapter.column_profile(REF, ["CounterID"], SAMPLE)

    assert profiles[0].sample_method == "sample"
    assert profiles[0].is_estimate is True
    assert profiles[0].sampled_rows == 10_000
    assert profiles[0].approx_distinct == 42
    assert profiles[0].null_ratio == 0.25
    assert profiles[0].top_values == (("1", 10), ("2", 5))
    assert profiles[0].is_low_cardinality(threshold=100) is True

    probe = client.queries[-1]
    assert "SAMPLE 0.01" in probe
    assert client.settings[-1]["max_execution_time"] == 10
    assert client.settings[-1]["max_rows_to_read"] == 100_000


async def test_a_table_without_a_sampling_key_is_profiled_from_a_bounded_prefix() -> None:
    adapter, client = _adapter(
        columns=_COLUMNS,
        skip_indexes=FakeResult(result_rows=[]),
        projections=FakeResult(result_rows=[]),
        footprint=FakeResult(result_rows=[[1, 2]]),
        profile=_PROFILE_ROW,
    )
    unsampleable = [*TABLE_ROW]
    unsampleable[4] = ""
    client.routes["create_table_query"] = FakeResult(result_rows=[unsampleable])

    profiles = await adapter.column_profile(REF, ["CounterID"], SAMPLE)

    assert "SAMPLE" not in client.queries[-1]
    assert "LIMIT 100000" in client.queries[-1]
    assert profiles[0].sample_method == "sample"


async def test_reading_the_whole_relation_is_the_only_thing_labelled_full() -> None:
    adapter, _ = _adapter(
        columns=_COLUMNS,
        skip_indexes=FakeResult(result_rows=[]),
        projections=FakeResult(result_rows=[]),
        footprint=FakeResult(result_rows=[[1, 2]]),
        profile=_PROFILE_ROW,
    )

    profiles = await adapter.column_profile(
        REF, ["CounterID"], SamplePolicy(fraction=1.0, max_rows=100, timeout_s=5)
    )

    assert profiles[0].sample_method == "full"
    assert profiles[0].is_estimate is False


async def test_profiling_a_column_that_does_not_exist_fails_before_touching_the_data() -> None:
    adapter, client = _adapter(
        columns=_COLUMNS,
        skip_indexes=FakeResult(result_rows=[]),
        projections=FakeResult(result_rows=[]),
        footprint=FakeResult(result_rows=[[1, 2]]),
    )

    with pytest.raises(QuerySemanticError):
        await adapter.column_profile(REF, ["NoSuchColumn"], SAMPLE)

    assert not any("uniqCombined64" in query for query in client.queries)


async def test_profiling_requires_the_column_stats_capability() -> None:
    adapter, _ = _adapter()
    stripped = ClickHouseAdapter(
        client=adapter.client, capabilities=adapter.capabilities - {Capability.COLUMN_STATS}
    )

    with pytest.raises(UnsupportedCapabilityError) as caught:
        await stripped.column_profile(REF, ["CounterID"], SAMPLE)

    assert caught.value.capability is Capability.COLUMN_STATS


# --------------------------------------------------------------------------
# plans
# --------------------------------------------------------------------------


async def test_explain_keeps_the_engine_output_verbatim_and_the_statement_beside_it() -> None:
    adapter, _ = _adapter(query=FakeResult(result_rows=[["Expression"], ["  ReadFromMergeTree"]]))

    plan = await adapter.explain("SELECT 1", ExplainMode.ESTIMATE)

    assert plan.payload == "Expression\n  ReadFromMergeTree"
    assert plan.mode is ExplainMode.ESTIMATE
    assert "use_query_condition_cache = 0" in plan.statements[0]


async def test_asking_clickhouse_for_an_analyzed_plan_is_refused_not_approximated() -> None:
    adapter, client = _adapter()

    with pytest.raises(UnsupportedCapabilityError) as caught:
        await adapter.explain("SELECT 1", ExplainMode.COST)

    assert caught.value.capability is Capability.COST_ANNOTATED_PLAN
    assert client.queries == []


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------

LIMITS = Limits(timeout_s=30, max_result_rows=2, max_rows_to_read=500, max_bytes_to_read=1_000)


async def test_execute_passes_every_ceiling_and_attributes_the_query() -> None:
    adapter, client = _adapter(
        query=FakeResult(
            column_names=["n"],
            result_rows=[[1]],
            summary={"read_rows": "10", "read_bytes": "80"},
        )
    )

    result = await adapter.execute("SELECT 1 AS n", LIMITS)

    assert result.columns == ("n",)
    assert result.rows == ((1,),)
    assert result.truncated is False
    assert result.rows_read == 10
    assert result.bytes_read == 80
    assert client.settings[-1] == {
        "log_comment": f"{LOG_COMMENT_PREFIX}:agentdb:turn0001",
        "max_execution_time": 30,
        "max_result_rows": 2,
        "max_rows_to_read": 500,
        "max_bytes_to_read": 1_000,
    }


async def test_a_result_past_the_row_ceiling_is_cut_and_says_it_was_cut() -> None:
    adapter, _ = _adapter(
        query=FakeResult(column_names=["n"], result_rows=[[1], [2], [3]]),
    )

    result = await adapter.execute("SELECT n", LIMITS)

    assert result.truncated is True
    assert result.row_count == 2


async def test_execution_without_scan_ceilings_sends_only_what_was_asked_for() -> None:
    adapter, client = _adapter(query=FakeResult(column_names=["n"], result_rows=[]))

    result = await adapter.execute("SELECT n", Limits(timeout_s=5, max_result_rows=10))

    assert result.rows_read is None
    assert result.bytes_read is None
    assert "max_rows_to_read" not in client.settings[-1]
    assert "max_bytes_to_read" not in client.settings[-1]


async def test_a_rejected_query_surfaces_as_a_typed_adapter_error() -> None:
    adapter, _ = _adapter(query=RuntimeError("Code: 62. DB::Exception: Syntax error"))

    with pytest.raises(QuerySyntaxError) as caught:
        await adapter.execute("SELECT", Limits(timeout_s=5, max_result_rows=10))

    assert caught.value.suggestion is not None


# --------------------------------------------------------------------------
# workload
# --------------------------------------------------------------------------

WINDOW = TimeWindow(
    start=datetime(2026, 8, 14, 0, 0, tzinfo=UTC), end=datetime(2026, 8, 14, 1, 0, tzinfo=UTC)
)


async def test_workload_returns_normalized_shapes_ranked_by_bytes_read() -> None:
    adapter, client = _adapter(
        workload=FakeResult(
            result_rows=[
                [
                    "SELECT count() FROM hits WHERE CounterID = ?",
                    12,
                    240.0,
                    20.0,
                    9_000,
                    80_000,
                    "SELECT count() FROM hits WHERE CounterID = 42",
                    ["agentdb.hits"],
                ]
            ]
        )
    )

    entries = await adapter.workload(WINDOW, top_n=5)

    assert entries[0].calls == 12
    assert entries[0].mean_duration_ms == 20.0
    assert entries[0].bytes_read == 80_000
    assert entries[0].relations == ("agentdb.hits",)
    assert client.parameters[-1] == {"start": WINDOW.start, "end": WINDOW.end, "top_n": 5}


async def test_workload_requires_the_workload_log_capability() -> None:
    adapter, _ = _adapter()
    stripped = ClickHouseAdapter(
        client=adapter.client, capabilities=adapter.capabilities - {Capability.WORKLOAD_LOG}
    )

    with pytest.raises(UnsupportedCapabilityError):
        await stripped.workload(WINDOW, top_n=5)


async def test_the_default_turn_id_is_unique_per_statement() -> None:
    client = FakeClient(routes={"version()": FakeResult(result_rows=[["25.9"]])})
    adapter = ClickHouseAdapter(client=client)

    await adapter.dialect_rules()
    await adapter.dialect_rules()

    comments = [str(setting["log_comment"]) for setting in client.settings]
    assert comments[0] != comments[1]
    assert all(comment.startswith(f"{LOG_COMMENT_PREFIX}:agentdb:") for comment in comments)
