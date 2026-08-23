"""The ClickHouse write channel, against a scripted driver (SPEC §9.1.B, §13.3).

The live tier proves the DDL runs. What is checked here is the part a running
server would not tell you about: that the plan read disables the caches whose
whole purpose is to answer the second identical query from the first — a
validation that measured its own cache would report every candidate as a triumph.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

from agentdb.adapters.base import QuerySemanticError
from agentdb.adapters.clickhouse_shadow import ClickHouseShadowRunner
from agentdb.adapters.models import ExplainMode


@dataclass
class FakeResult:
    result_rows: Sequence[Sequence[Any]] = ()


@dataclass
class FakeClient:
    rows: list[Sequence[Any]] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    parameters: list[Mapping[str, Any]] = field(default_factory=list)
    failure: Exception | None = None

    async def query(
        self, query: str, *, parameters: Mapping[str, Any], settings: Mapping[str, Any]
    ) -> FakeResult:
        self.queries.append(query)
        self.parameters.append(parameters)
        if self.failure is not None:
            raise self.failure
        return FakeResult(result_rows=self.rows)


async def test_a_statement_is_run_as_written() -> None:
    client = FakeClient()

    await ClickHouseShadowRunner(client=client).run("DROP TABLE IF EXISTS tpch.x__agentdb_shadow_1")

    assert client.queries == ["DROP TABLE IF EXISTS tpch.x__agentdb_shadow_1"]


async def test_the_plan_read_disables_the_caches_that_would_answer_it_from_itself() -> None:
    client = FakeClient(rows=[['{"Plan": {}}']])

    plan = await ClickHouseShadowRunner(client=client).explain(
        "SELECT count() FROM tpch.orders", ExplainMode.ESTIMATE
    )

    assert "use_query_condition_cache = 0" in client.queries[0]
    assert "use_skip_indexes_on_data_read = 0" in client.queries[0]
    assert plan.payload == '{"Plan": {}}'
    assert plan.engine == "clickhouse"


async def test_a_multi_row_plan_is_reassembled_in_order() -> None:
    """ClickHouse returns JSON EXPLAIN a line at a time; a reordered plan is nonsense."""
    client = FakeClient(rows=[["[{"], ['"Plan":'], ["{}}]"]])

    plan = await ClickHouseShadowRunner(client=client).explain("SELECT 1", ExplainMode.ESTIMATE)

    assert plan.payload == '[{"Plan":{}}]'


async def test_the_reaper_asks_for_table_names_by_parameter_not_by_interpolation() -> None:
    client = FakeClient(rows=[["orders__agentdb_shadow_a"], ["orders"]])

    tables = await ClickHouseShadowRunner(client=client).list_tables("tpch")

    assert tables == ("orders__agentdb_shadow_a", "orders")
    assert client.parameters[0] == {"namespace": "tpch"}
    assert "tpch" not in client.queries[0], "the namespace travels as a bound parameter"


async def test_a_driver_failure_arrives_as_the_adapters_own_error_type() -> None:
    """Core never sees a driver exception, on the write path either."""
    client = FakeClient(failure=RuntimeError("Code: 47. DB::Exception: Unknown identifier"))

    with pytest.raises(QuerySemanticError):
        await ClickHouseShadowRunner(client=client).run("CREATE TABLE nope AS SELECT bad")
