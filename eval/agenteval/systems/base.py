"""The system-under-test interface (SPEC §11.5).

agenteval scores whole stacks: an MCP server, a managed service, or a bare model
with a schema dump. Everything reaches the harness through :class:`SystemUnderTest`,
which is why the harness can measure systems this project did not write — the
property that makes it a scoreboard rather than a self-report.

Two rules are encoded here rather than promised in prose:

* ``controls_model`` is explicit, so a managed service that picks its own model
  is footnoted in the report instead of silently compared like-for-like.
* :meth:`Attempt.blind` strips the system's identity. The scorer only ever sees
  a :class:`BlindAttempt`, so grading cannot be influenced by who produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal, Protocol, runtime_checkable

from agenteval.tasks import Task

ErrorClass = Literal[
    "syntax", "semantic", "plan_rejection", "timeout", "permission", "limit_exceeded", "none"
]
"""Failure taxonomy the report distributes over (SPEC §11.1). Declared locally so
agenteval stays independent of any system it measures."""


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Which model a system should use, when the system lets the caller choose."""

    provider: str
    name: str
    temperature: float = 0.0
    max_output_tokens: int | None = None

    def __str__(self) -> str:
        return f"{self.provider}/{self.name}"


@dataclass(frozen=True, slots=True)
class EmittedQuery:
    """One SQL statement a system emitted, and what the engine did with it.

    Every attempt keeps all of them in order, so a reader auditing a trace sees
    the self-correction loop rather than only its outcome.
    """

    sql: str
    succeeded: bool
    error_class: ErrorClass = "none"
    error_text: str | None = None
    columns: tuple[str, ...] = ()
    rows: tuple[tuple[object, ...], ...] = ()
    row_count: int | None = None
    duration_ms: int | None = None
    rows_read: int | None = None
    bytes_read: int | None = None
    statement_id: str | None = None
    """The engine's own id for this execution, where it issues one.

    Databricks does, and it is what joins a number in this trace to the
    warehouse's own record without any string matching. ``None`` on ClickHouse,
    which attributes through ``log_comment`` instead."""

    files_read: int | None = None
    files_pruned: int | None = None
    """Measured file pruning, when the engine reported it after execution.

    Databricks only: its ``EXPLAIN`` carries no file counts at all, so unlike
    ClickHouse's granule numbers this cannot be known before the query runs. Both
    stay ``None`` when the result cache answered or when the statement opened no
    file — a zero here would read as "pruned everything" (SPEC §8.2)."""


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Context grounding is not free, and the report has to say so (SPEC §11.1)."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class BlindAttempt:
    """An attempt with the producing system removed. The only thing the scorer sees."""

    task_id: str
    seed: int
    queries: tuple[EmittedQuery, ...] = ()
    tokens: TokenUsage = TokenUsage()
    context_bytes: int = 0
    wall_clock_ms: int | None = None

    @property
    def final_query(self) -> EmittedQuery | None:
        """The last query emitted — the one that stands as the system's answer."""
        return self.queries[-1] if self.queries else None

    @property
    def first_query(self) -> EmittedQuery | None:
        """The first query emitted, for the EX@1 metric."""
        return self.queries[0] if self.queries else None

    @property
    def retries(self) -> int:
        """Failed attempts preceding the final one."""
        return sum(1 for query in self.queries[:-1] if not query.succeeded)


@dataclass(frozen=True, slots=True)
class Attempt:
    """One system's full response to one task at one seed.

    Committed verbatim to ``results/raw/*.jsonl`` so any single claim in the
    report can be audited back to the prompt and the engine's reply.
    """

    system: str
    task_id: str
    seed: int
    model: ModelSpec | None = None
    prompt: str | None = None
    """The grounded payload the system sent. Committed to the trace (SPEC §11.4)
    and deliberately absent from :class:`BlindAttempt`, since a prompt names the
    arm that wrote it."""

    queries: tuple[EmittedQuery, ...] = ()
    tokens: TokenUsage = TokenUsage()
    context_bytes: int = 0
    wall_clock_ms: int | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def blind(self) -> BlindAttempt:
        """Strip system identity before grading (SPEC §11.5: the scorer is blind)."""
        return BlindAttempt(
            task_id=self.task_id,
            seed=self.seed,
            queries=self.queries,
            tokens=self.tokens,
            context_bytes=self.context_bytes,
            wall_clock_ms=self.wall_clock_ms,
        )

    def with_note(self, note: str) -> Attempt:
        """A copy carrying an extra provenance note. Attempts are never mutated."""
        return replace(self, notes=(*self.notes, note))


@runtime_checkable
class SystemUnderTest(Protocol):
    """Anything agenteval can score.

    ``version`` and ``config_fingerprint`` are not bookkeeping: every row of the
    Family S leaderboard carries them, because a benchmark of a moving beta
    product is meaningless without saying exactly what was measured.

    They are declared read-only so an implementation can be a frozen dataclass:
    a system's identity must not drift halfway through a run.
    """

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def controls_model(self) -> bool: ...

    @property
    def config_fingerprint(self) -> str: ...

    async def answer(self, task: Task, model: ModelSpec | None, seed: int) -> Attempt:
        """Answer ``task`` once, returning every query emitted along the way."""
        ...
