"""Live checks against a Databricks SQL warehouse (SPEC §12, M3.5).

Everything the Databricks adapter knows was read from documentation. These tests
are where that becomes an observation instead of a claim: they run the adapter's
own methods against ``samples.tpch``, which every workspace ships pre-loaded,
including Free Edition.

They **skip loudly** rather than silently: a fork PR cannot reach workspace
secrets, and a suite that quietly reports green while never touching a warehouse
is how a Databricks adapter rots between releases. The skip message names the
variable that was missing.

Run with::

    uv sync --extra databricks
    uv run pytest -m e2e tests/e2e/test_databricks_live.py

Nothing here writes. Every statement is a read, ``ANALYZE`` is never issued, and
the principal needs ``SELECT`` and nothing more.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from uuid import uuid4

import pytest

from agentdb.adapters.base import AdapterError, EngineConnectionError
from agentdb.adapters.databricks import DatabricksAdapter
from agentdb.adapters.databricks_client import DatabricksTarget, build_client
from agentdb.adapters.models import ExplainMode, Limits, RelationRef, SamplePolicy
from agentdb.config import Config
from agentdb.core.explain import PlanExplainer
from agentdb.core.plan_analyzer_databricks import parse_plan

pytestmark = pytest.mark.e2e

REQUIRED = ("AGENTDB_DBX_HOST", "AGENTDB_DBX_WAREHOUSE_ID")
CATALOG = os.environ.get("AGENTDB_DBX_CATALOG", "samples")
SCHEMA = os.environ.get("AGENTDB_DBX_SCHEMA", "tpch")

PROBE_SQL = f"""
SELECT l_returnflag, count(*) AS orders
FROM {CATALOG}.{SCHEMA}.lineitem
WHERE l_shipdate >= DATE '1995-01-01'
GROUP BY l_returnflag
""".strip()

LIMITS = Limits(timeout_s=60, max_result_rows=1_000)
"""Room for a serverless warehouse that may be cold (SPEC §8.2 footgun 6)."""


def _missing() -> list[str]:
    return [name for name in REQUIRED if not os.environ.get(name)]


@pytest.fixture(scope="module")
def lineitem() -> RelationRef:
    return RelationRef(catalog=CATALOG, namespace=SCHEMA, name="lineitem")


@pytest.fixture(scope="module")
def _warehouse() -> Iterator[None]:
    missing = _missing()
    if missing:
        pytest.skip(
            f"Databricks e2e skipped: {', '.join(missing)} unset. "
            "Copy .env.example to .env, or export the variables, then re-run. "
            "This suite is the only thing that observes the shapes SPEC §8.2 marks VERIFY:."
        )
    yield


@pytest.fixture
async def adapter(_warehouse: None) -> DatabricksAdapter:
    target = DatabricksTarget.from_env()
    return DatabricksAdapter(client=await build_client(target), catalog=CATALOG, context_id="e2e")


async def test_the_sample_catalog_lists_the_tpch_tables(adapter: DatabricksAdapter) -> None:
    relations = await adapter.list_relations(SCHEMA)

    names = {relation.ref.name for relation in relations}
    assert {"lineitem", "orders"} <= names
    assert all(relation.ref.catalog is not None for relation in relations)


async def test_columns_arrive_in_the_ordinal_order_delta_statistics_stop_at(
    adapter: DatabricksAdapter, lineitem: RelationRef
) -> None:
    detail = await adapter.describe_relation(lineitem)

    assert detail.column_names[0] == "l_orderkey"
    assert "l_shipdate" in detail.column_names
    assert detail.create_statement.startswith("CREATE ")


async def test_the_layout_reports_the_delta_facts_the_rules_depend_on(
    adapter: DatabricksAdapter, lineitem: RelationRef
) -> None:
    layout = await adapter.physical_layout(lineitem)

    # format and file counts are what DESCRIBE DETAIL is read for; if a column
    # was renamed upstream these come back None and the pruning ratio silently
    # stops being computable
    assert layout.table_format is not None
    assert layout.num_files is not None
    assert layout.on_disk_bytes is not None


async def test_the_dialect_version_is_readable(adapter: DatabricksAdapter) -> None:
    rules = await adapter.dialect_rules()

    assert rules.version != "unknown"
    assert rules.identifier_quote == "`"


async def test_a_sampled_profile_returns_real_distribution_facts(
    adapter: DatabricksAdapter, lineitem: RelationRef
) -> None:
    profiles = await adapter.column_profile(
        lineitem, ["l_returnflag"], SamplePolicy(fraction=0.01, max_rows=1_000_000, timeout_s=60)
    )

    profile = profiles[0]
    assert profile.sample_method == "sample"
    assert profile.sampled_rows > 0
    assert profile.approx_distinct is not None
    assert profile.top_values


async def test_the_plan_parser_reads_a_real_explain_formatted(
    adapter: DatabricksAdapter,
) -> None:
    plan = await adapter.explain(PROBE_SQL, ExplainMode.ESTIMATE)

    root = parse_plan(plan)
    scans = [node for node in root.walk() if node.relation is not None]
    assert scans, f"no scan node parsed from:\n{plan.payload[:2000]}"
    # a filter on a clustered/indexed column should reach the scan as a pushed
    # or partition filter; if neither is populated the field names have moved
    assert scans[0].pushed_filters or scans[0].partition_filters


async def test_the_whole_a3_path_produces_a_summary_an_agent_could_act_on(
    adapter: DatabricksAdapter,
) -> None:
    summary = await PlanExplainer(adapter=adapter, config=Config()).explain(PROBE_SQL, SCHEMA)

    assert summary.engine == "databricks"
    assert summary.photon_coverage is not None
    rendered = summary.render()
    assert "Plan summary (databricks)" in rendered


async def test_a_rejected_query_is_classified_not_raised_as_a_connection_failure(
    adapter: DatabricksAdapter,
) -> None:
    with pytest.raises(AdapterError) as caught:
        await adapter.explain(
            f"SELECT no_such_column FROM {CATALOG}.{SCHEMA}.lineitem", ExplainMode.ESTIMATE
        )

    # the warehouse answered, so this must not look like a dead socket
    assert not isinstance(caught.value, EngineConnectionError)


# -- measured metrics --------------------------------------------------------


async def test_measured_metrics_supply_the_pruning_evidence_explain_does_not(
    adapter: DatabricksAdapter,
) -> None:
    """The whole reason this path exists.

    A Photon plan carries no file counts at all, so on Databricks the estimate
    plan cannot say what fraction of a table a query read. The warehouse can, but
    only afterwards.
    """
    # the nonce goes in a predicate, not a comment: a comment does not defeat the
    # result cache, and a cached answer reports every counter as zero
    nonce = int(uuid4().hex[:4], 16) % 97
    sql = PROBE_SQL.replace(
        "WHERE l_shipdate >= DATE '1995-01-01'",
        f"WHERE l_shipdate >= DATE '1995-01-01' AND l_orderkey >= {nonce}",
    )
    result = await adapter.execute(sql, LIMITS)
    assert result.query_id is not None

    measured = await adapter.query_metrics(result.query_id)

    assert measured is not None
    assert measured.measured is True
    assert measured.files_read is not None and measured.files_read > 0
    assert measured.pruning_ratio is not None
    assert measured.source == "query_history_api"


async def test_a_statement_id_the_warehouse_never_saw_measures_nothing(
    adapter: DatabricksAdapter,
) -> None:
    # well-formed but never issued. A *malformed* id is rejected by the API with
    # "has an invalid length of 32" rather than reported as missing, which is the
    # right behaviour — that is a bug in the caller, not an unmeasured statement.
    assert await adapter.query_metrics("01f19800-0000-0000-0000-000000000000") is None


async def test_the_system_table_is_too_far_behind_to_attribute_a_live_statement(
    adapter: DatabricksAdapter,
) -> None:
    """Why the Query History API is used instead of ``system.query.history``.

    SPEC §8.2 names the system table as the workload source, and it is the right
    one for mining last week. It is not a source of attribution for the statement
    that just ran: measured 1,514 and 23,290 seconds behind the warehouse clock
    on two runs. This asserts the *reason*, so that if the lag ever closes the
    failure is a prompt to simplify rather than a silent missed opportunity.
    """
    lag = await adapter.execute(
        "SELECT timestampdiff(SECOND, max(start_time), current_timestamp()) AS lag_seconds "
        "FROM system.query.history",
        LIMITS,
    )
    seconds = int(str(lag.rows[0][0]))

    assert seconds > 60, (
        f"system.query.history is only {seconds}s behind; it may now be usable for "
        "inline attribution, which would remove the need for the history API"
    )
