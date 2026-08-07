"""Per-task traces, committed to ``results/raw/*.jsonl`` (SPEC §11.4).

A reader must be able to audit any single claim in the report: which prompt was
sent, every query that came back, what the engine said, how long it took, what
it cost. That is the difference between a benchmark and a press release, so the
trace carries the whole attempt rather than a summary of it.

Records are line-delimited JSON with sorted keys, so a re-run produces a diff a
human can read instead of a reordered file.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agenteval.scorer import Score
from agenteval.systems.base import Attempt, EmittedQuery, SystemUnderTest
from agenteval.tasks import Engine, Task

TRACE_SCHEMA_VERSION = 1
"""Bumped when the record shape changes, so old traces stay interpretable."""

MAX_TRACE_ROWS = 20
"""Rows kept per query. A trace is evidence, not a data export: full result sets
from a 100M-row table would make the committed traces useless to read and
impossible to review."""


def build_record(
    *,
    run_id: str,
    engine: Engine,
    task: Task,
    system: SystemUnderTest,
    attempt: Attempt,
    score: Score,
) -> dict[str, Any]:
    """One auditable row: what was asked, what answered, and how it was graded."""
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "run_id": run_id,
        "engine": engine,
        "suite": task.suite,
        "task_id": task.id,
        "difficulty": task.difficulty,
        "tags": list(task.tags),
        "question": task.question,
        "gold_sql": task.gold_sql,
        "system": system.name,
        "system_version": system.version,
        "controls_model": system.controls_model,
        "config_fingerprint": system.config_fingerprint,
        "model": str(attempt.model) if attempt.model else None,
        "seed": attempt.seed,
        "prompt": attempt.prompt,
        "queries": [_query_record(query) for query in attempt.queries],
        "notes": list(attempt.notes),
        "verdict": score.verdict,
        "execution_accuracy": score.execution_accuracy,
        "accuracy_at_1": score.accuracy_at_1,
        "valid_sql": score.valid_sql,
        "error_class": score.error_class,
        "retries": score.retries,
        "order_sensitive": score.order_sensitive,
        "reason": score.reason,
        "bytes_read": score.bytes_read,
        "wall_clock_ms": attempt.wall_clock_ms,
        "input_tokens": score.input_tokens,
        "output_tokens": score.output_tokens,
        "context_bytes": score.context_bytes,
    }


def _query_record(query: EmittedQuery) -> dict[str, Any]:
    return {
        "sql": query.sql,
        "succeeded": query.succeeded,
        "error_class": query.error_class,
        "error_text": query.error_text,
        "columns": list(query.columns),
        "rows": [list(row) for row in query.rows[:MAX_TRACE_ROWS]],
        "rows_truncated": len(query.rows) > MAX_TRACE_ROWS,
        "row_count": query.row_count,
        "duration_ms": query.duration_ms,
        "rows_read": query.rows_read,
        "bytes_read": query.bytes_read,
    }


@dataclass(frozen=True, slots=True)
class TraceWriter:
    """Appends records to one JSONL file, creating it on first write."""

    path: Path

    def append(self, record: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, sort_keys=True, default=str)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(f"{line}\n")


def read_records(path: Path) -> tuple[dict[str, Any], ...]:
    """Load a trace file back — the input to ``make report``."""
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return tuple(json.loads(line) for line in lines)
