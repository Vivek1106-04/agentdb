"""The README's before/after panel: measured, and honest when the measurement disappoints."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from agentdb.adapters import ExplainMode, Limits, PhysicalLayout, RawPlan, ResultSet
from agentdb.core import PlanNode, PlanOp, PlanSummary
from agentdb.demo import CLICKHOUSE_CASE, DATABRICKS_CASE, Panel, render, run_demo
from tests.fakes import FakeAdapter, clickhouse_hits_fixture

PLAN_WITH_NO_PRUNING: dict[str, Any] = {
    "Plan": {
        "Node Type": "ReadFromMergeTree",
        "Description": "agentdb.hits",
        "Indexes": [
            {"Type": "PrimaryKey", "Initial Granules": 1_000, "Selected Granules": 1_000},
        ],
    }
}


@dataclass
class ScriptedAdapter(FakeAdapter):
    """A fake that answers each of the demo's two queries differently.

    The demo's whole point is the gap between the two executions, which a single
    scripted result cannot express.
    """

    by_sql: dict[str, ResultSet] = field(default_factory=dict)

    async def execute(self, sql: str, limits: Limits) -> ResultSet:
        self.calls.append(("execute", (sql, limits)))
        return self.by_sql[sql]


def scripted(naive: ResultSet, grounded: ResultSet) -> ScriptedAdapter:
    fixture = clickhouse_hits_fixture()
    return ScriptedAdapter(
        relations=fixture.relations,
        details=fixture.details,
        layouts=fixture.layouts,
        profiles=fixture.profiles,
        rules=fixture.rules,
        plan=RawPlan(
            engine="clickhouse",
            mode=ExplainMode.ESTIMATE,
            sql="",
            payload=json.dumps([PLAN_WITH_NO_PRUNING]),
        ),
        by_sql={CLICKHOUSE_CASE.naive_sql: naive, CLICKHOUSE_CASE.grounded_sql: grounded},
    )


def count_result(value: int, *, bytes_read: int | None, rows_read: int | None = None) -> ResultSet:
    return ResultSet(
        columns=("count()",),
        rows=((value,),),
        row_count=1,
        truncated=False,
        duration_ms=42,
        rows_read=rows_read,
        bytes_read=bytes_read,
    )


async def test_the_panel_reports_both_queries_and_the_ratio_between_them() -> None:
    adapter = scripted(
        count_result(1_234, bytes_read=4 * 1024**3, rows_read=99_997_497),
        count_result(1_234, bytes_read=256 * 1024**2, rows_read=2_400_000),
    )

    panel = await run_demo(adapter, CLICKHOUSE_CASE)

    assert CLICKHOUSE_CASE.naive_sql in panel
    assert CLICKHOUSE_CASE.grounded_sql in panel
    assert "ORDER BY (CounterID, EventDate, UserID)" in panel
    assert "99,997,497 rows" in panel
    assert "Same answer (1234), 4.0 GB read against 256 MB — 16.0x." in panel


async def test_a_disagreement_is_printed_rather_than_averaged_away() -> None:
    adapter = scripted(
        count_result(1_234, bytes_read=4 * 1024**3),
        count_result(999, bytes_read=1024),
    )

    panel = await run_demo(adapter, CLICKHOUSE_CASE)

    assert "did not agree" in panel
    assert "1234 vs 999" in panel


async def test_an_engine_that_reports_no_bytes_makes_no_claim() -> None:
    adapter = scripted(count_result(7, bytes_read=None), count_result(7, bytes_read=None))

    panel = await run_demo(adapter, CLICKHOUSE_CASE)

    assert "no byte counts to compare" in panel
    assert "bytes unreported" in panel


async def test_the_plan_warnings_of_both_halves_are_shown() -> None:
    adapter = scripted(
        count_result(1, bytes_read=2048, rows_read=10),
        count_result(1, bytes_read=1024, rows_read=5),
    )

    panel = await run_demo(adapter, CLICKHOUSE_CASE)

    assert panel.count("  plan     ") >= 2
    assert "MISSING_PARTITION_PREDICATE" in panel


async def test_a_multi_column_answer_is_described_rather_than_flattened() -> None:
    wide = ResultSet(
        columns=("a", "b"),
        rows=((1, 2), (3, 4)),
        row_count=2,
        truncated=False,
        bytes_read=1024,
    )
    adapter = scripted(wide, wide)

    panel = await run_demo(adapter, CLICKHOUSE_CASE)

    assert "2 rows" in panel


async def test_the_databricks_case_asks_the_same_question_both_ways() -> None:
    assert DATABRICKS_CASE.namespace == "samples.tpch"
    assert "year(o.o_orderdate) = 1995" in DATABRICKS_CASE.naive_sql
    assert "o.o_orderdate >= DATE'1995-01-01'" in DATABRICKS_CASE.grounded_sql


def bare_panel(heading: str, *, bytes_read: int | None) -> Panel:
    """A panel with nothing to warn about, for the quiet paths."""
    return Panel(
        heading=heading,
        sql="SELECT 1",
        note="note",
        answer="1",
        bytes_read=bytes_read,
        rows_read=None,
        duration_ms=None,
        summary=PlanSummary(
            root=PlanNode(op=PlanOp.OTHER, node_type="Nothing"),
            engine="clickhouse",
            sql="SELECT 1",
        ),
    )


def test_a_table_with_no_declared_layout_says_only_what_it_knows() -> None:
    layout = PhysicalLayout(
        engine="clickhouse",
        ref=CLICKHOUSE_CASE.ref,
        create_statement="CREATE TABLE agentdb.hits (x UInt8) ENGINE = Log",
    )

    panel = render(
        CLICKHOUSE_CASE,
        layout,
        bare_panel("before", bytes_read=2 * 1024**4),
        bare_panel("after", bytes_read=1024**4),
    )

    assert "ORDER BY" not in panel
    assert "PARTITION BY" not in panel
    assert "rows" not in panel.split("Question")[0]
    assert "plan     no warnings" in panel
    assert "2.0 TB read against 1.0 TB — 2.0x." in panel
