"""Resolving the gold answer for a task, and catching gold drift.

The gold result is produced by running the task's ``gold_sql`` against the same
engine every system under test is scored on. That makes the reference answer a
property of the data rather than of whoever wrote the task file — and it means
the harness can detect the failure mode that quietly invalidates text-to-SQL
benchmarks: the dataset changed, the gold query now returns something else, and
every number computed since is wrong.

``gold_result_hash`` in the task file is the tripwire. When present it is
checked on every run, and a mismatch stops the run rather than reporting a
plausible-looking number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agenteval.execution import QueryExecutor
from agenteval.scorer import GoldResult, has_top_level_order_by, result_hash
from agenteval.tasks import Task


class GoldError(RuntimeError):
    """The reference answer could not be trusted. Always fatal to the run."""


async def resolve_gold(executor: QueryExecutor, task: Task) -> GoldResult:
    """Run ``task.gold_sql`` and return its result, verifying any committed hash."""
    emitted = await executor.run(task.gold_sql)
    if not emitted.succeeded:
        raise GoldError(
            f"gold_sql for task {task.id!r} does not run on {executor.engine} "
            f"({emitted.error_class}): {emitted.error_text or 'no detail'}"
        )

    gold = GoldResult(columns=emitted.columns, rows=emitted.rows)
    if task.gold_result_hash is None:
        return gold

    digest = result_hash(gold.columns, gold.rows, ordered=has_top_level_order_by(task.gold_sql))
    if digest != task.gold_result_hash:
        raise GoldError(
            f"gold drift on task {task.id!r}: the file commits {task.gold_result_hash}, "
            f"the engine produced {digest}. Re-verify the data before trusting any result."
        )
    return gold


@dataclass
class GoldCache:
    """Gold results memoized per task for the length of one run.

    Every arm, model, and seed is graded against the *same* gold rows: resolving
    once is both faster and a fairness property, since a table that changed
    mid-run would otherwise score early arms against different truth than late
    ones.
    """

    executor: QueryExecutor
    _resolved: dict[str, GoldResult] = field(default_factory=dict)

    async def get(self, task: Task) -> GoldResult:
        cached = self._resolved.get(task.id)
        if cached is not None:
            return cached
        gold = await resolve_gold(self.executor, task)
        self._resolved[task.id] = gold
        return gold
