"""Result comparison and scoring (SPEC §11.1).

Ambiguity here is how benchmarks get discredited, so the rules are written down
once, implemented once, and tested directly:

* Set comparison is **order-insensitive** unless the gold query has a top-level
  ``ORDER BY``, in which case order is part of the answer.
* Rows are a **multiset**: duplicates are significant.
* **Column names are ignored**; the column *count* and the multiset of values
  must match.
* Floats compare at ``1e-6`` **relative** tolerance.
* ``NULL`` equals ``NULL``.

The scorer never sees which system produced an attempt — it takes a
:class:`~agenteval.systems.base.BlindAttempt` (SPEC §11.5).
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Literal

from agenteval.systems.base import BlindAttempt, EmittedQuery, ErrorClass
from agenteval.tasks import Task

FLOAT_RELATIVE_TOLERANCE = 1e-6
"""Relative tolerance for float comparison. SPEC §11.1."""

_FLOAT_HASH_PRECISION = 9
"""Significant digits used when hashing a float, comfortably finer than the tolerance."""

_LINE_COMMENT = re.compile(r"--[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_STRING_LITERAL = re.compile(r"'(?:''|[^'])*'", re.DOTALL)
_PAREN_OR_ORDER_BY = re.compile(r"[()]|\border\s+by\b", re.IGNORECASE)

Verdict = Literal["correct", "incorrect", "no_query", "execution_error"]


@dataclass(frozen=True, slots=True)
class Score:
    """The graded outcome of one attempt on one task."""

    task_id: str
    seed: int
    verdict: Verdict
    execution_accuracy: bool
    """The primary metric: the final query's result matches gold."""

    accuracy_at_1: bool
    """Whether the *first* emitted query was already correct."""

    valid_sql: bool
    """Whether the final query executed at all, correct or not."""

    error_class: ErrorClass
    retries: int
    order_sensitive: bool
    bytes_read: int | None = None
    duration_ms: int | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    context_bytes: int = 0
    reason: str | None = None
    """Why an incorrect verdict was reached, in one human-readable line."""


@dataclass(frozen=True, slots=True)
class GoldResult:
    """The reference answer a task is graded against."""

    columns: tuple[str, ...]
    rows: tuple[tuple[object, ...], ...]


def has_top_level_order_by(sql: str) -> bool:
    """Whether ``sql`` orders its final result.

    Only a top-level ``ORDER BY`` makes the answer order-sensitive: one inside a
    subquery or a window function does not survive into the result, and treating
    it as if it did would fail correct answers.
    """
    stripped = _strip_noise(sql)
    depth = 0
    for match in _PAREN_OR_ORDER_BY.finditer(stripped):
        token = match.group(0)
        if token == "(":
            depth += 1
        elif token == ")":
            depth = max(depth - 1, 0)
        elif depth == 0:
            return True
    return False


def results_match(
    gold: GoldResult,
    actual_columns: tuple[str, ...],
    actual_rows: tuple[tuple[object, ...], ...],
    *,
    ordered: bool,
) -> tuple[bool, str | None]:
    """Compare a result against gold under the SPEC §11.1 rules.

    Returns the verdict and, when it is negative, a one-line reason — a
    benchmark that only says "wrong" is far harder to audit than one that says
    which rule was violated.
    """
    if len(actual_columns) != len(gold.columns):
        return False, (
            f"column count differs: gold has {len(gold.columns)}, got {len(actual_columns)}"
        )
    if len(actual_rows) != len(gold.rows):
        return False, f"row count differs: gold has {len(gold.rows)}, got {len(actual_rows)}"

    left = [tuple(_normalize(value) for value in row) for row in gold.rows]
    right = [tuple(_normalize(value) for value in row) for row in actual_rows]

    if not ordered:
        left = sorted(left, key=_row_sort_key)
        right = sorted(right, key=_row_sort_key)

    for index, (gold_row, actual_row) in enumerate(zip(left, right, strict=True)):
        if not _rows_equal(gold_row, actual_row):
            position = f"at position {index}" if ordered else "in the row multiset"
            return False, f"row mismatch {position}: expected {gold_row!r}, got {actual_row!r}"
    return True, None


def result_hash(
    columns: tuple[str, ...], rows: tuple[tuple[object, ...], ...], *, ordered: bool
) -> str:
    """A stable ``sha256:`` digest of a result under the comparison rules.

    Column *names* are excluded and only the column count contributes, so the
    hash means the same thing the comparison does. Committed as
    ``gold_result_hash`` on every task, which is what makes gold drift
    detectable rather than a thing reviewers have to take on trust.
    """
    normalized = [tuple(_normalize(value) for value in row) for row in rows]
    if not ordered:
        normalized = sorted(normalized, key=_row_sort_key)

    digest = hashlib.sha256()
    digest.update(f"columns:{len(columns)}\n".encode())
    digest.update(f"ordered:{int(ordered)}\n".encode())
    for row in normalized:
        digest.update("\x1f".join(_hash_token(value) for value in row).encode())
        digest.update(b"\x1e")
    return f"sha256:{digest.hexdigest()}"


def score_attempt(task: Task, attempt: BlindAttempt, gold: GoldResult) -> Score:
    """Grade one blind attempt against the gold result for ``task``."""
    ordered = has_top_level_order_by(task.gold_sql)
    final = attempt.final_query
    first = attempt.first_query

    if final is None:
        return _empty_score(task, attempt, ordered)

    correct, reason = _grade(final, gold, ordered=ordered)
    at_1 = bool(first is not None and _grade(first, gold, ordered=ordered)[0])
    verdict: Verdict = (
        "correct" if correct else ("execution_error" if not final.succeeded else "incorrect")
    )

    return Score(
        task_id=task.id,
        seed=attempt.seed,
        verdict=verdict,
        execution_accuracy=correct,
        accuracy_at_1=at_1,
        valid_sql=final.succeeded,
        error_class=final.error_class,
        retries=attempt.retries,
        order_sensitive=ordered,
        bytes_read=final.bytes_read,
        duration_ms=attempt.wall_clock_ms,
        input_tokens=attempt.tokens.input_tokens,
        output_tokens=attempt.tokens.output_tokens,
        context_bytes=attempt.context_bytes,
        reason=reason,
    )


def _empty_score(task: Task, attempt: BlindAttempt, ordered: bool) -> Score:
    return Score(
        task_id=task.id,
        seed=attempt.seed,
        verdict="no_query",
        execution_accuracy=False,
        accuracy_at_1=False,
        valid_sql=False,
        error_class="none",
        retries=0,
        order_sensitive=ordered,
        input_tokens=attempt.tokens.input_tokens,
        output_tokens=attempt.tokens.output_tokens,
        context_bytes=attempt.context_bytes,
        duration_ms=attempt.wall_clock_ms,
        reason="the system emitted no query",
    )


def _grade(query: EmittedQuery, gold: GoldResult, *, ordered: bool) -> tuple[bool, str | None]:
    if not query.succeeded:
        return False, f"query failed ({query.error_class}): {query.error_text or 'no detail'}"
    return results_match(gold, query.columns, query.rows, ordered=ordered)


def _strip_noise(sql: str) -> str:
    """Remove comments and string literals so keyword scanning cannot be fooled."""
    without_blocks = _BLOCK_COMMENT.sub(" ", sql)
    without_lines = _LINE_COMMENT.sub(" ", without_blocks)
    return _STRING_LITERAL.sub("''", without_lines)


def _normalize(value: object) -> object:
    """Canonicalize one cell so equal answers compare equal across drivers."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, datetime | date):
        return value.isoformat()
    return str(value)


def _rows_equal(left: tuple[object, ...], right: tuple[object, ...]) -> bool:
    return all(_values_equal(a, b) for a, b in zip(left, right, strict=True))


def _values_equal(left: object, right: object) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, float) and isinstance(right, float):
        if math.isnan(left) or math.isnan(right):
            return math.isnan(left) and math.isnan(right)
        return math.isclose(left, right, rel_tol=FLOAT_RELATIVE_TOLERANCE, abs_tol=0.0)
    return left == right


def _row_sort_key(row: tuple[object, ...]) -> tuple[tuple[int, str], ...]:
    return tuple(_value_sort_key(value) for value in row)


def _value_sort_key(value: object) -> tuple[int, str]:
    """A total order over mixed-type cells, so multisets can be sorted at all."""
    if value is None:
        return (0, "")
    if isinstance(value, bool):
        return (1, str(int(value)))
    if isinstance(value, float):
        return (2, _hash_token(value))
    return (3, str(value))


def _hash_token(value: object) -> str:
    if value is None:
        return "\x00NULL"
    if isinstance(value, bool):
        return f"\x00BOOL:{int(value)}"
    if isinstance(value, float):
        if math.isnan(value):
            return "\x00NAN"
        if math.isinf(value):
            return f"\x00INF:{int(math.copysign(1, value))}"
        return f"\x00NUM:{value:.{_FLOAT_HASH_PRECISION}g}"
    return f"\x00STR:{value}"
