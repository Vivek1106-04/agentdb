"""The run loop. Fairness properties here are structural, so they are tested."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from agenteval.models.base import ModelError
from agenteval.runner import Cell, RunConfigError, RunSpec, new_run_id, run
from agenteval.systems.base import ModelSpec
from agenteval.tasks import Task, TaskSuite
from agenteval.traces import TraceWriter, read_records
from tests.harness_fakes import MODEL, SYNTAX_ERROR, FakeExecutor, StubSystem, sample_task

SONNET = ModelSpec(provider="anthropic", name="claude-sonnet-5")


def _suite(*tasks: Task) -> TaskSuite:
    return TaskSuite(name="clickbench_nl", tasks=tasks or (sample_task(),))


def _spec(
    *,
    systems: tuple[StubSystem, ...],
    models: tuple[ModelSpec, ...] = (MODEL,),
    seeds: tuple[int, ...] = (0,),
    suite: TaskSuite | None = None,
) -> RunSpec:
    return RunSpec(
        suite=suite or _suite(),
        systems=systems,
        models=models,
        run_id="run-test",
        seeds=seeds,
    )


def test_a_run_id_is_utc_stamped() -> None:
    assert new_run_id().startswith("run-")
    assert new_run_id().endswith("Z")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"systems": ()}, "at least one system"),
        ({"models": ()}, "at least one model"),
        ({"seeds": ()}, "at least one seed"),
    ],
)
def test_an_empty_dimension_is_refused(kwargs: dict[str, tuple[object, ...]], message: str) -> None:
    defaults: dict[str, object] = {
        "suite": _suite(),
        "systems": (StubSystem(),),
        "models": (MODEL,),
        "seeds": (0,),
        "run_id": "r",
    }

    with pytest.raises(RunConfigError, match=message):
        RunSpec(**{**defaults, **kwargs})  # type: ignore[arg-type]


async def test_a_suite_with_no_tasks_for_the_engine_is_refused() -> None:
    postgres_only = _suite(replace(sample_task(), engines=("postgres",)))

    with pytest.raises(RunConfigError, match="no tasks targeting clickhouse"):
        await run(_spec(systems=(StubSystem(),), suite=postgres_only), FakeExecutor())


async def test_the_matrix_covers_every_system_model_and_seed() -> None:
    # Arrange
    systems = (StubSystem(name="A0_baseline"), StubSystem(name="S4_agentdb"))
    spec = _spec(systems=systems, models=(MODEL, SONNET), seeds=(0, 1, 2))

    # Act
    cells = await run(spec, FakeExecutor())

    # Assert — 2 systems x 2 models x 3 seeds
    assert len(cells) == 12
    assert {cell.system for cell in cells} == {"A0_baseline", "S4_agentdb"}
    assert {cell.model for cell in cells} == {
        "anthropic/claude-opus-5",
        "anthropic/claude-sonnet-5",
    }
    assert {cell.seed for cell in cells} == {0, 1, 2}


async def test_a_correct_answer_scores_as_correct() -> None:
    cells = await run(_spec(systems=(StubSystem(),)), FakeExecutor())

    assert cells == (
        Cell(
            system="S_stub",
            task_id="clickbench_nl_001",
            model="anthropic/claude-opus-5",
            seed=0,
            score=cells[0].score,
        ),
    )
    assert cells[0].score.execution_accuracy is True
    assert cells[0].score.verdict == "correct"


async def test_gold_is_resolved_once_and_shared_across_arms() -> None:
    # Arrange — two arms, three seeds: gold must not be re-run 6 times
    executor = FakeExecutor()
    systems = (StubSystem(name="a"), StubSystem(name="b"))

    await run(_spec(systems=systems, seeds=(0, 1, 2)), executor)

    assert executor.executed.count("SELECT count() FROM hits") == 1


async def test_a_system_that_picks_its_own_model_runs_once_per_seed() -> None:
    # Arrange — a managed service gets a footnote, not a silent like-for-like row
    managed = StubSystem(name="S3_clickhouse_agents", controls_model=False)

    cells = await run(
        _spec(systems=(managed,), models=(MODEL, SONNET), seeds=(0, 1)), FakeExecutor()
    )

    assert len(cells) == 2
    assert {cell.model for cell in cells} == {None}
    assert [call[1] for call in managed.calls] == [None, None]


async def test_a_crashing_system_becomes_a_recorded_failure_not_a_lost_run() -> None:
    # Arrange — a provider outage must not discard cells already earned
    broken = StubSystem(name="broken", error=ModelError("upstream is down"))
    healthy = StubSystem(name="healthy")

    cells = await run(_spec(systems=(broken, healthy)), FakeExecutor())

    failed = next(cell for cell in cells if cell.system == "broken")
    assert failed.score.verdict == "no_query"
    assert failed.score.execution_accuracy is False
    assert next(cell for cell in cells if cell.system == "healthy").score.verdict == "correct"


async def test_a_wrong_answer_is_scored_not_hidden() -> None:
    wrong = StubSystem(queries=(replace(SYNTAX_ERROR, sql="SELEC 1"),))

    cells = await run(_spec(systems=(wrong,)), FakeExecutor(outcomes=[]))

    assert cells[0].score.verdict == "execution_error"
    assert cells[0].score.error_class == "syntax"


async def test_every_cell_is_written_to_the_trace_file(tmp_path: Path) -> None:
    writer = TraceWriter(path=tmp_path / "raw.jsonl")
    spec = _spec(systems=(StubSystem(),), seeds=(0, 1))

    await run(spec, FakeExecutor(), writer=writer)

    records = read_records(writer.path)
    assert len(records) == 2
    assert {record["run_id"] for record in records} == {"run-test"}
    assert {record["seed"] for record in records} == {0, 1}
