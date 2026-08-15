"""The Databricks adapter against a scripted warehouse.

What these tests hold the adapter to, beyond "it parses":

* the three-level name is always complete, and the adapter's catalog fills a
  two-part reference rather than the session's ``USE`` state;
* the statistics facts survive — ``delta.dataSkippingNumIndexedCols`` and
  ``delta.dataSkippingStatsColumns`` are what make ``STATS_NOT_COLLECTED``
  computable, and losing them is a silent failure, not a crash;
* every statement carries its attribution comment, so a reported figure can be
  joined to ``system.query.history`` by someone who does not trust us;
* nothing measured is invented: an absent count stays ``None``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pytest

from agentdb.adapters import Capability, ExplainMode, QuerySemanticError, RelationRef
from agentdb.adapters.base import UnsupportedCapabilityError
from agentdb.adapters.databricks import DatabricksAdapter
from agentdb.adapters.models import Limits, SamplePolicy, TimeWindow

LINEITEM = RelationRef(catalog="samples", namespace="tpch", name="lineitem")


@dataclass(frozen=True, slots=True)
class FakeResult:
    """One scripted statement response."""

    columns: tuple[str, ...] = ()
    rows: tuple[tuple[Any, ...], ...] = ()
    statement_id: str | None = "01ef-statement"
    truncated: bool = False
    rows_read: int | None = None
    bytes_read: int | None = None
    duration_ms: int | None = None


@dataclass
class FakeClient:
    """A warehouse that answers by matching a fragment of the statement text."""

    responses: dict[str, FakeResult] = field(default_factory=dict)
    failure: Exception | None = None
    calls: list[tuple[str, Mapping[str, Any], dict[str, Any]]] = field(default_factory=list)

    async def statement(
        self,
        sql: str,
        *,
        parameters: Mapping[str, Any],
        row_limit: int | None = None,
        byte_limit: int | None = None,
        timeout_s: int | None = None,
    ) -> FakeResult:
        self.calls.append(
            (
                sql,
                parameters,
                {"row_limit": row_limit, "byte_limit": byte_limit, "timeout_s": timeout_s},
            )
        )
        if self.failure is not None:
            raise self.failure
        for fragment, response in self.responses.items():
            if fragment in sql:
                return response
        return FakeResult()

    def statements(self) -> list[str]:
        return [sql for sql, _, _ in self.calls]


DETAIL_COLUMNS = (
    "format",
    "partitionColumns",
    "clusteringColumns",
    "numFiles",
    "sizeInBytes",
    "numRows",
    "tableType",
)


def _client(**overrides: FakeResult) -> FakeClient:
    responses: dict[str, FakeResult] = {
        # the existence probe is matched before the listing: both read
        # information_schema.tables and only the predicate tells them apart
        "SELECT table_type": FakeResult(
            columns=("table_type", "data_source_format", "comment"),
            rows=(("MANAGED", "DELTA", "TPC-H lineitem"),),
        ),
        "information_schema.tables": FakeResult(
            columns=(
                "table_catalog",
                "table_schema",
                "table_name",
                "table_type",
                "data_source_format",
                "comment",
            ),
            rows=(("samples", "tpch", "lineitem", "MANAGED", "DELTA", None),),
        ),
        "information_schema.columns": FakeResult(
            columns=("column_name", "ordinal_position", "full_data_type", "is_nullable", "comment"),
            rows=(
                ("l_orderkey", 1, "bigint", "NO", None),
                ("l_shipdate", 11, "date", "YES", "ship date"),
            ),
        ),
        "SHOW CREATE TABLE": FakeResult(
            columns=("createtab_stmt",),
            rows=(("CREATE TABLE samples.tpch.lineitem (...) USING delta",),),
        ),
        "DESCRIBE DETAIL": FakeResult(
            columns=DETAIL_COLUMNS,
            rows=(
                ("delta", ["l_shipdate"], ["l_orderkey"], 400, 4_194_304_000, 6_001_215, "MANAGED"),
            ),
        ),
        "SHOW TBLPROPERTIES": FakeResult(
            columns=("key", "value"),
            rows=(
                ("delta.dataSkippingNumIndexedCols", "8"),
                ("delta.enableDeletionVectors", "true"),
            ),
        ),
        "DESCRIBE HISTORY": FakeResult(
            columns=("version", "operation", "operationParameters"),
            rows=((3, "OPTIMIZE", {"zOrderBy": '["l_partkey"]'}),),
        ),
    }
    responses.update(overrides)
    return FakeClient(responses=responses)


def _adapter(client: FakeClient | None = None, **overrides: Any) -> DatabricksAdapter:
    return DatabricksAdapter(
        client=client or _client(),
        catalog="samples",
        turn_id=lambda: "turn01",
        **overrides,
    )


# -- capabilities -----------------------------------------------------------


def test_the_adapter_declares_what_delta_can_do_and_not_what_it_cannot() -> None:
    adapter = _adapter()

    assert adapter.supports(Capability.FILE_PRUNING)
    assert adapter.supports(Capability.CLUSTERING_KEY)
    assert adapter.supports(Capability.DATA_SKIPPING_STATS)
    assert adapter.supports(Capability.THREE_LEVEL_NAMESPACE)
    # granules and skip indexes are ClickHouse mechanisms and are absent, not faked
    assert not adapter.supports(Capability.GRANULE_PRUNING)
    assert not adapter.supports(Capability.SKIP_INDEX)
    assert not adapter.supports(Capability.PROJECTION)


# -- discovery --------------------------------------------------------------


async def test_listing_a_schema_returns_fully_qualified_references() -> None:
    client = _client(
        **{
            "information_schema.tables": FakeResult(
                columns=(
                    "table_catalog",
                    "table_schema",
                    "table_name",
                    "table_type",
                    "data_source_format",
                    "comment",
                ),
                rows=(("samples", "tpch", "lineitem", "MANAGED", "DELTA", None),),
            )
        }
    )

    relations = await _adapter(client).list_relations("tpch")

    assert relations[0].ref == LINEITEM
    assert str(relations[0].ref) == "samples.tpch.lineitem"
    assert relations[0].kind == "table"
    assert relations[0].engine_type == "DELTA"
    # information_schema carries no size facts, and the adapter does not invent them
    assert relations[0].approx_rows is None


async def test_a_catalog_qualified_namespace_overrides_the_adapters_default() -> None:
    client = _client()

    await _adapter(client).list_relations("main.sales")

    assert client.calls[0][1] == {"catalog": "main", "schema": "sales"}


async def test_listing_without_a_namespace_stays_inside_the_configured_catalog() -> None:
    client = _client()

    await _adapter(client).list_relations()

    assert client.calls[0][1] == {"catalog": "samples"}
    assert "table_schema NOT IN" in client.statements()[0]


async def test_describe_reads_columns_in_ordinal_order() -> None:
    detail = await _adapter().describe_relation(LINEITEM)

    assert detail.column_names == ("l_orderkey", "l_shipdate")
    assert detail.columns[0].is_nullable is False
    assert detail.columns[1].is_nullable is True
    assert detail.create_statement.startswith("CREATE TABLE samples.tpch.lineitem")


async def test_a_two_part_reference_is_completed_from_the_adapters_catalog() -> None:
    client = _client()

    detail = await _adapter(client).describe_relation(
        RelationRef(namespace="tpch", name="lineitem")
    )

    assert detail.ref == LINEITEM
    assert client.calls[0][1]["catalog"] == "samples"


async def test_a_missing_relation_is_a_semantic_error_not_an_empty_description() -> None:
    client = _client(**{"SELECT table_type": FakeResult(columns=("table_type",), rows=())})

    with pytest.raises(QuerySemanticError, match="does not exist"):
        await _adapter(client).describe_relation(LINEITEM)


# -- physical layout --------------------------------------------------------


async def test_layout_carries_the_delta_facts_no_schema_dump_reveals() -> None:
    layout = await _adapter().physical_layout(LINEITEM)

    assert layout.table_format == "delta"
    assert layout.clustering_columns == ("l_orderkey",)
    assert layout.partition_by == ("l_shipdate",)
    assert layout.zorder_columns == ("l_partkey",)
    assert layout.stats_indexed_columns == 8
    assert layout.deletion_vectors_enabled is True
    assert layout.is_managed is True
    assert layout.num_files == 400
    assert layout.avg_file_bytes == 4_194_304_000 / 400
    assert layout.approx_rows == 6_001_215


async def test_the_statistics_column_list_overrides_the_indexed_column_count() -> None:
    client = _client(
        **{
            "SHOW TBLPROPERTIES": FakeResult(
                columns=("key", "value"),
                rows=(
                    ("delta.dataSkippingNumIndexedCols", "8"),
                    ("delta.dataSkippingStatsColumns", "l_shipdate, l_orderkey"),
                ),
            )
        }
    )

    layout = await _adapter(client).physical_layout(LINEITEM)

    assert layout.stats_columns == ("l_shipdate", "l_orderkey")
    assert layout.has_file_statistics("l_shipdate", 11, default_indexed=32) is True
    assert layout.has_file_statistics("l_comment", 16, default_indexed=32) is False


async def test_a_table_that_sets_no_delta_properties_reports_unknown_not_zero() -> None:
    client = _client(**{"SHOW TBLPROPERTIES": FakeResult(columns=("key", "value"), rows=())})

    layout = await _adapter(client).physical_layout(LINEITEM)

    assert layout.stats_indexed_columns is None
    assert layout.stats_columns is None
    assert layout.deletion_vectors_enabled is None
    # and the Delta default then decides, which is config, not a hardcoded guess
    assert layout.has_file_statistics("l_shipdate", 11, default_indexed=32) is True


async def test_a_detail_row_missing_its_size_gives_no_average_file_size() -> None:
    client = _client(
        **{"DESCRIBE DETAIL": FakeResult(columns=("format", "numFiles"), rows=(("delta", 0),))}
    )

    layout = await _adapter(client).physical_layout(LINEITEM)

    assert layout.avg_file_bytes is None
    assert layout.is_managed is None
    assert layout.clustering_columns is None


async def test_a_relation_with_no_detail_row_is_refused() -> None:
    client = _client(**{"DESCRIBE DETAIL": FakeResult(columns=DETAIL_COLUMNS, rows=())})

    with pytest.raises(QuerySemanticError, match="no detail row"):
        await _adapter(client).physical_layout(LINEITEM)


# -- profiling --------------------------------------------------------------


def _profile_client() -> FakeClient:
    return _client(
        **{
            "approx_count_distinct": FakeResult(
                columns=("approx_distinct", "null_ratio", "min_value", "max_value", "sampled_rows"),
                rows=((2_526, 0.0, "1992-01-02", "1998-12-01", 60_012),),
            ),
            "GROUP BY 1 ORDER BY 2 DESC": FakeResult(
                columns=("value", "occurrences"),
                rows=(("1995-06-15", 41), ("1996-03-13", 39)),
            ),
        }
    )


async def test_profiling_samples_and_says_so() -> None:
    policy = SamplePolicy(fraction=0.01, max_rows=1_000_000, timeout_s=30)

    profiles = await _adapter(_profile_client()).column_profile(LINEITEM, ["l_shipdate"], policy)

    assert profiles[0].sample_method == "sample"
    assert profiles[0].is_estimate is True
    assert profiles[0].approx_distinct == 2_526
    assert profiles[0].sampled_rows == 60_012
    assert profiles[0].top_values == (("1995-06-15", 41), ("1996-03-13", 39))


async def test_profiling_a_column_costs_two_statements_on_databricks() -> None:
    client = _profile_client()
    policy = SamplePolicy(fraction=0.01, max_rows=1_000_000, timeout_s=30)

    await _adapter(client).column_profile(LINEITEM, ["l_shipdate"], policy)

    probes = [sql for sql in client.statements() if "TABLESAMPLE" in sql]
    assert len(probes) == 2
    assert "TABLESAMPLE (1.0 PERCENT)" in probes[0]


async def test_a_full_fraction_falls_back_to_the_configured_percent() -> None:
    client = _profile_client()
    policy = SamplePolicy(fraction=1.0, max_rows=1_000, timeout_s=5)

    await _adapter(client, sample_percent=2.5).column_profile(LINEITEM, ["l_shipdate"], policy)

    assert any("TABLESAMPLE (2.5 PERCENT)" in sql for sql in client.statements())


async def test_profiling_an_unknown_column_is_refused_before_a_probe_runs() -> None:
    client = _profile_client()
    policy = SamplePolicy(fraction=0.01, max_rows=1_000, timeout_s=5)

    with pytest.raises(QuerySemanticError, match="has no column"):
        await _adapter(client).column_profile(LINEITEM, ["nope"], policy)

    assert not [sql for sql in client.statements() if "TABLESAMPLE" in sql]


async def test_profiling_requires_the_capability() -> None:
    adapter = _adapter(capabilities=frozenset())
    policy = SamplePolicy(fraction=0.01, max_rows=1_000, timeout_s=5)

    with pytest.raises(UnsupportedCapabilityError):
        await adapter.column_profile(LINEITEM, ["l_shipdate"], policy)


# -- plans and execution ----------------------------------------------------


async def test_the_estimate_plan_is_explain_formatted_and_is_kept_verbatim() -> None:
    client = _client(
        **{
            "EXPLAIN FORMATTED": FakeResult(
                columns=("plan",), rows=(("== Physical Plan ==",), ("PhotonScan parquet",))
            )
        }
    )

    plan = await _adapter(client).explain("SELECT 1", ExplainMode.ESTIMATE)

    assert plan.engine == "databricks"
    assert plan.payload == "== Physical Plan ==\nPhotonScan parquet"
    assert plan.statements == ("EXPLAIN FORMATTED SELECT 1",)


async def test_a_cost_plan_is_refused_when_the_capability_is_absent() -> None:
    adapter = _adapter(capabilities=frozenset({Capability.ESTIMATE_ONLY_PLAN}))

    with pytest.raises(UnsupportedCapabilityError) as caught:
        await adapter.explain("SELECT 1", ExplainMode.COST)

    assert caught.value.capability is Capability.COST_ANNOTATED_PLAN


async def test_execution_reports_the_statement_id_so_the_figure_can_be_audited() -> None:
    client = _client(
        **{
            "SELECT l_orderkey": FakeResult(
                columns=("l_orderkey",),
                rows=((1,), (2,)),
                statement_id="01ef-abc",
                rows_read=6_001_215,
                bytes_read=48_000_000,
                duration_ms=812,
            )
        }
    )

    result = await _adapter(client).execute(
        "SELECT l_orderkey FROM samples.tpch.lineitem",
        Limits(timeout_s=30, max_result_rows=10),
    )

    assert result.query_id == "01ef-abc"
    assert result.rows_read == 6_001_215
    assert result.bytes_read == 48_000_000
    assert result.duration_ms == 812
    assert result.truncated is False


async def test_a_result_over_the_row_cap_is_truncated_and_says_so() -> None:
    client = _client(
        **{"SELECT l_orderkey": FakeResult(columns=("l_orderkey",), rows=((1,), (2,), (3,)))}
    )

    result = await _adapter(client).execute(
        "SELECT l_orderkey FROM samples.tpch.lineitem",
        Limits(timeout_s=30, max_result_rows=2),
    )

    assert result.row_count == 2
    assert result.truncated is True


async def test_a_warehouse_reported_truncation_is_carried_through() -> None:
    client = _client(
        **{"SELECT l_orderkey": FakeResult(columns=("l_orderkey",), rows=((1,),), truncated=True)}
    )

    result = await _adapter(client).execute(
        "SELECT l_orderkey FROM samples.tpch.lineitem",
        Limits(timeout_s=30, max_result_rows=10),
    )

    assert result.truncated is True


async def test_limits_reach_the_client_rather_than_being_advisory() -> None:
    client = _client()

    await _adapter(client).execute(
        "SELECT 1", Limits(timeout_s=17, max_result_rows=25, max_bytes_to_read=1_000)
    )

    _, _, options = client.calls[-1]
    assert options == {"row_limit": 25, "byte_limit": 1_000, "timeout_s": 17}


# -- workload and dialect ---------------------------------------------------


async def test_workload_reads_statements_one_row_per_execution() -> None:
    client = _client(
        **{
            "system.query.history": FakeResult(
                columns=(
                    "statement_text",
                    "statement_type",
                    "execution_status",
                    "total_duration_ms",
                    "read_bytes",
                    "read_rows",
                    "produced_rows",
                    "statement_id",
                ),
                rows=(
                    (
                        "SELECT count(*) FROM lineitem",
                        "SELECT",
                        "FINISHED",
                        812.0,
                        48_000,
                        6_001,
                        1,
                        "01ef-abc",
                    ),
                ),
            )
        }
    )
    window = TimeWindow(
        start=datetime(2026, 8, 14, tzinfo=UTC), end=datetime(2026, 8, 15, tzinfo=UTC)
    )

    entries = await _adapter(client).workload(window, top_n=5)

    assert entries[0].calls == 1
    assert entries[0].bytes_read == 48_000
    assert entries[0].rows_read == 6_001
    assert entries[0].query_id == "01ef-abc"


async def test_workload_requires_the_capability() -> None:
    adapter = _adapter(capabilities=frozenset())
    window = TimeWindow(
        start=datetime(2026, 8, 14, tzinfo=UTC), end=datetime(2026, 8, 15, tzinfo=UTC)
    )

    with pytest.raises(UnsupportedCapabilityError):
        await adapter.workload(window, top_n=5)


async def test_dialect_rules_state_the_facts_that_cause_databricks_failures() -> None:
    client = _client(**{"current_version": FakeResult(columns=("version",), rows=(("2026.30",),))})

    rules = await _adapter(client).dialect_rules()

    assert rules.version == "2026.30"
    assert rules.identifier_quote == "`"
    assert rules.quote_identifier("order") == "`order`"
    assert any("catalog.schema.table" in quirk for quirk in rules.quirks)
    assert any("first 32 columns" in quirk for quirk in rules.quirks)


async def test_a_warehouse_that_reports_no_version_is_unknown_rather_than_blank() -> None:
    client = _client(**{"current_version": FakeResult(columns=("version",), rows=())})

    assert (await _adapter(client).dialect_rules()).version == "unknown"


# -- attribution and failure ------------------------------------------------


async def test_every_statement_carries_its_attribution_comment() -> None:
    client = _client()

    await _adapter(client, context_id="bench").physical_layout(LINEITEM)

    assert client.statements()
    assert all(sql.startswith("/* agentdb:bench:turn01 */") for sql in client.statements())


async def test_a_driver_failure_becomes_a_classified_adapter_error() -> None:
    client = _client()
    client.failure = RuntimeError("[TABLE_OR_VIEW_NOT_FOUND] cannot find `samples`.`tpch`.`nope`")

    with pytest.raises(QuerySemanticError) as caught:
        await _adapter(client).list_relations("tpch")

    assert "catalog.schema.table" in (caught.value.suggestion or "")


def _unused(value: Sequence[object]) -> None:  # pragma: no cover - typing aid only
    return None


async def test_a_probe_that_reported_no_null_ratio_leaves_it_unknown() -> None:
    client = _client(
        **{
            "approx_count_distinct": FakeResult(
                columns=("approx_distinct", "null_ratio", "min_value", "max_value", "sampled_rows"),
                rows=((None, None, None, None, None),),
            ),
            "GROUP BY 1 ORDER BY 2 DESC": FakeResult(columns=("value", "occurrences"), rows=()),
        }
    )
    policy = SamplePolicy(fraction=0.01, max_rows=1_000, timeout_s=5)

    profile = (await _adapter(client).column_profile(LINEITEM, ["l_shipdate"], policy))[0]

    assert profile.null_ratio is None
    assert profile.approx_distinct is None
    assert profile.min_value is None
    assert profile.sampled_rows == 0


async def test_numeric_facts_arriving_as_strings_are_still_read() -> None:
    client = _client(
        **{
            "approx_count_distinct": FakeResult(
                columns=("approx_distinct", "null_ratio", "min_value", "max_value", "sampled_rows"),
                rows=(("2526", "0.25", "1992-01-02", "1998-12-01", "60012"),),
            ),
            "GROUP BY 1 ORDER BY 2 DESC": FakeResult(columns=("value", "occurrences"), rows=()),
        }
    )
    policy = SamplePolicy(fraction=0.01, max_rows=1_000, timeout_s=5)

    profile = (await _adapter(client).column_profile(LINEITEM, ["l_shipdate"], policy))[0]

    assert profile.approx_distinct == 2_526
    assert profile.null_ratio == 0.25


async def test_an_unreadable_duration_in_the_history_is_unknown_rather_than_zero() -> None:
    client = _client(
        **{
            "system.query.history": FakeResult(
                columns=("statement_text",),
                rows=(("SELECT 1", "SELECT", "FINISHED", "not a number", None, None, None, None),),
            )
        }
    )
    window = TimeWindow(
        start=datetime(2026, 8, 14, tzinfo=UTC), end=datetime(2026, 8, 15, tzinfo=UTC)
    )

    entry = (await _adapter(client).workload(window, top_n=1))[0]

    assert entry.total_duration_ms is None
    assert entry.bytes_read is None
    assert entry.query_id is None


async def test_an_explain_that_returned_an_analysis_error_is_raised_not_returned() -> None:
    client = _client(
        **{
            "EXPLAIN FORMATTED": FakeResult(
                columns=("plan",),
                rows=(
                    ("Error occurred during query planning: ",),
                    (
                        "[UNRESOLVED_COLUMN.WITH_SUGGESTION] A column with name `nope` "
                        "cannot be resolved. SQLSTATE: 42703",
                    ),
                ),
            )
        }
    )

    # the warehouse answers EXPLAIN over invalid SQL with success and puts the
    # error where the plan should be; handing that to the parser would turn a
    # repairable semantic error into a crash
    with pytest.raises(QuerySemanticError, match="UNRESOLVED_COLUMN"):
        await _adapter(client).explain(
            "SELECT nope FROM samples.tpch.lineitem", ExplainMode.ESTIMATE
        )
