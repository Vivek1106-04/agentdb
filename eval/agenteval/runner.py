"""The run loop: every system x task x model x seed, graded blind (SPEC §11.4).

The loop is deliberately sequential. Running arms concurrently against one
engine would let a busy arm's queries inflate a quiet arm's latency and bytes
read, and those are reported numbers — a faster run that produces uncomparable
timings is not a faster run.

Two rules are structural rather than promised:

* Gold is resolved once per task and shared by every arm, so no arm is graded
  against different truth than another.
* The scorer is handed ``attempt.blind()``, never the attempt, so grading cannot
  see which system it is grading.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from agenteval.execution import QueryExecutor
from agenteval.gold import GoldCache
from agenteval.scorer import Score, score_attempt
from agenteval.systems.base import Attempt, ModelSpec, SystemUnderTest
from agenteval.tasks import Task, TaskSuite
from agenteval.traces import TraceWriter, build_record

DEFAULT_SEEDS = (0, 1, 2, 3, 4)
"""SPEC §11.4 requires N_SEEDS >= 5 per (task, arm, model)."""


class RunConfigError(ValueError):
    """A run was requested that could not produce a meaningful table."""


def new_run_id() -> str:
    """A UTC-stamped id. Goes on every trace record so runs never interleave."""
    return datetime.now(UTC).strftime("run-%Y%m%dT%H%M%SZ")


@dataclass(frozen=True, slots=True)
class RunSpec:
    """What to run. Every field lands in the report, so none of it is implicit."""

    suite: TaskSuite
    systems: tuple[SystemUnderTest, ...]
    models: tuple[ModelSpec, ...]
    run_id: str
    seeds: tuple[int, ...] = DEFAULT_SEEDS

    def __post_init__(self) -> None:
        if not self.systems:
            raise RunConfigError("a run needs at least one system under test")
        if not self.models:
            raise RunConfigError("a run needs at least one model")
        if not self.seeds:
            raise RunConfigError("a run needs at least one seed")


@dataclass(frozen=True, slots=True)
class Cell:
    """One graded (system, task, model, seed). The unit the report aggregates."""

    system: str
    task_id: str
    model: str | None
    """``None`` when the system chose its own — footnoted, never silently compared."""

    seed: int
    score: Score


async def run(
    spec: RunSpec, executor: QueryExecutor, *, writer: TraceWriter | None = None
) -> tuple[Cell, ...]:
    """Execute the whole matrix, writing a trace per cell if ``writer`` is given."""
    tasks = spec.suite.for_engine(executor.engine)
    if not len(tasks):
        raise RunConfigError(f"suite {spec.suite.name!r} has no tasks targeting {executor.engine}")

    gold_cache = GoldCache(executor=executor)
    cells: list[Cell] = []

    for task in tasks:
        gold = await gold_cache.get(task)
        for system in spec.systems:
            for model in _models_for(system, spec.models):
                for seed in spec.seeds:
                    attempt = await _attempt(system, task, model, seed)
                    score = score_attempt(task, attempt.blind(), gold)
                    if writer is not None:
                        writer.append(
                            build_record(
                                run_id=spec.run_id,
                                engine=executor.engine,
                                task=task,
                                system=system,
                                attempt=attempt,
                                score=score,
                            )
                        )
                    cells.append(
                        Cell(
                            system=system.name,
                            task_id=task.id,
                            model=str(model) if model else None,
                            seed=seed,
                            score=score,
                        )
                    )

    return tuple(cells)


def _models_for(
    system: SystemUnderTest, models: Sequence[ModelSpec]
) -> tuple[ModelSpec | None, ...]:
    """A managed service that picks its own model runs once, not once per model."""
    if system.controls_model:
        return tuple(models)
    return (None,)


async def _attempt(
    system: SystemUnderTest, task: Task, model: ModelSpec | None, seed: int
) -> Attempt:
    """Ask one system for one answer, turning a crash into a recorded failure.

    A provider outage midway through a long run must not discard the cells
    already earned. The exception is never swallowed: it becomes a note on the
    attempt, which has no queries and therefore scores as ``no_query`` — visible
    in the report and in the trace rather than absent from both.
    """
    try:
        return await system.answer(task, model, seed)
    except Exception as exc:
        return Attempt(
            system=system.name,
            task_id=task.id,
            seed=seed,
            model=model,
            notes=(f"the system raised {type(exc).__name__}: {exc}",),
        )
