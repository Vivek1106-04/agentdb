"""The whole benchmark pipeline against live engines, minus the model (SPEC §18).

Everything between a task file and a committed report runs here for real: gold
resolution against the 100M-row table, arm construction including a Family A
provider resolved by dotted path, execution under the read-only role, grading by
result hash, trace writing, and the report renderer reading those traces back.

The one thing stubbed is the model itself. That is deliberate — the vendor SDK
call is the only step that costs money, and a pipeline failure has never once
been in the SDK. What this test protects is the part that has broken before: an
executor pointed at the wrong database, an arm whose provider will not load, a
grader comparing an unordered result as if it were ordered.

Run with::

    make up && make load-clickbench CLICKBENCH_PARTS=100
    uv run pytest -m e2e tests/e2e/test_bench_pipeline_live.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from agenteval.cli import BenchOptions, ReportOptions, run_bench, run_report
from agenteval.engines.clickhouse import ClickHouseExecutor
from agenteval.engines.connect import ClickHouseTarget, build_client
from agenteval.models.base import ModelResponse, Turn
from agenteval.suites import load_builtin
from agenteval.systems.base import ModelSpec, TokenUsage

pytestmark = pytest.mark.e2e

MODEL = "anthropic/claude-opus-5"
"""Named, never called: the stub answers instead. The trace still records which
model a real run would have used, which is what the report keys its rows on."""


@dataclass
class GoldReplayClient:
    """A model that always writes the gold query for the task it was asked about.

    Not a cheat — nothing about accuracy is being measured here. It is the one
    reply that makes every downstream step observable: the executor must run it,
    the grader must hash it, and the report must show a full column of correct
    answers. A wrong answer would prove the pipeline works too, but would leave
    the grader's agreement with its own gold untested.
    """

    answers: dict[str, str]
    provider: str = "stub"
    calls: list[str] = field(default_factory=list)

    async def complete(
        self, *, system: str, turns: tuple[Turn, ...], model: ModelSpec, seed: int
    ) -> ModelResponse:
        # Every turn, not just the last: a plan-review arm's final turn is the
        # engine's plan, and the question it is about was asked two turns back.
        conversation = "\n".join(turn.content for turn in turns)
        sql = next(
            (gold for asked, gold in self.answers.items() if asked in conversation),
            "SELECT 1",
        )
        self.calls.append(conversation)
        return ModelResponse(
            text=f"```sql\n{sql}\n```",
            tokens=TokenUsage(input_tokens=len(system) // 4, output_tokens=len(sql) // 4),
        )


async def live_executor() -> ClickHouseExecutor:
    target = ClickHouseTarget.from_env()
    try:
        client = await build_client(target)
    except Exception as exc:
        pytest.skip(f"no ClickHouse reachable ({exc}); start one with: make up")
    return ClickHouseExecutor(client=client)


@pytest.fixture
def gold_client() -> GoldReplayClient:
    suite = load_builtin("clickbench_nl").subset(3)
    return GoldReplayClient(answers={task.question: task.gold_sql for task in suite})


async def test_a_run_grades_itself_correct_against_the_live_table(
    tmp_path: Path, gold_client: GoldReplayClient
) -> None:
    """Gold in, gold out: the grader must agree with the answers it froze."""
    written: list[str] = []

    cells = await run_bench(
        BenchOptions(
            suite="clickbench_nl",
            engine="clickhouse",
            arms=("A0_baseline",),
            models=(MODEL,),
            seeds=(0,),
            limit=3,
            out=tmp_path,
        ),
        executor_factory=live_executor,
        client_factory=lambda: gold_client,
        write=written.append,
    )

    assert len(cells) == 3
    assert all(cell.score.execution_accuracy for cell in cells), [
        cell.task_id for cell in cells if not cell.score.execution_accuracy
    ]
    assert gold_client.calls, "the arm really did ask the model"


async def test_a_family_a_arm_loads_its_provider_and_carries_more_context(
    tmp_path: Path, gold_client: GoldReplayClient
) -> None:
    """A2 resolves agentdb by dotted path, connects, and the payload grows."""
    cells = await run_bench(
        BenchOptions(
            suite="clickbench_nl",
            engine="clickhouse",
            arms=("A0_baseline", "A2_layout"),
            models=(MODEL,),
            seeds=(0,),
            limit=2,
            out=tmp_path,
        ),
        executor_factory=live_executor,
        client_factory=lambda: gold_client,
        write=lambda _line: None,
    )

    assert {cell.system for cell in cells} == {"A0_baseline", "A2_layout"}
    assert all(cell.score.execution_accuracy for cell in cells)

    traces = [
        json.loads(line)
        for path in tmp_path.glob("*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    context_bytes = {record["system"]: record["context_bytes"] for record in traces}
    assert context_bytes["A2_layout"] > context_bytes["A0_baseline"], (
        "the layout arm must carry more grounding than the schema-only baseline"
    )


async def test_the_traces_a_run_writes_are_the_traces_the_report_reads(
    tmp_path: Path, gold_client: GoldReplayClient
) -> None:
    """The last mile of SPEC §18.2: every number traceable to a committed record."""
    await run_bench(
        BenchOptions(
            suite="clickbench_nl",
            engine="clickhouse",
            arms=("A0_baseline",),
            models=(MODEL,),
            seeds=(0,),
            limit=2,
            out=tmp_path,
        ),
        executor_factory=live_executor,
        client_factory=lambda: gold_client,
        write=lambda _line: None,
    )

    written: list[str] = []
    markdown = run_report(
        ReportOptions(from_raw=tmp_path, out=tmp_path / "REPORT.md", baseline="A0_baseline"),
        write=written.append,
    )

    assert "## Execution accuracy" in markdown
    assert "`A0_baseline`" in markdown
    assert "100.0%" in markdown
    assert (tmp_path / "REPORT.md").is_file()


async def test_the_whole_ladder_runs_through_the_runner_on_live_infrastructure(
    tmp_path: Path, gold_client: GoldReplayClient
) -> None:
    """A0 to A6 and S5, built by the CLI's own arm loader against real engines.

    The arms below A4 need only ClickHouse. A4 upward need the exemplar store
    too, and A6 additionally reads its committed reference workload — so this is
    the test that says the ladder the report will publish can actually be built,
    end to end, by the code path `make bench` uses.
    """
    arms = ("A0_baseline", "A2_layout", "A3_plan", "A4_memory", "A5_negmemory", "A6_full")

    cells = await run_bench(
        BenchOptions(
            suite="clickbench_nl",
            engine="clickhouse",
            arms=arms,
            models=(MODEL,),
            seeds=(0,),
            limit=1,
            out=tmp_path,
        ),
        executor_factory=live_executor,
        client_factory=lambda: gold_client,
        write=lambda _line: None,
    )

    assert {cell.system for cell in cells} == set(arms)
    assert all(cell.score.execution_accuracy for cell in cells), [
        cell.system for cell in cells if not cell.score.execution_accuracy
    ]

    traces = [
        json.loads(line)
        for path in tmp_path.glob("*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    context_bytes = {record["system"]: record["context_bytes"] for record in traces}
    assert context_bytes["A0_baseline"] < context_bytes["A2_layout"] < context_bytes["A6_full"], (
        "each rung of the ladder must carry strictly more than the one below it"
    )
    fingerprints = {record["system"]: record["config_fingerprint"] for record in traces}
    assert len(set(fingerprints.values())) == len(arms), "every arm is separately identified"
