"""Test doubles for the harness side of the project.

Kept apart from ``tests/fakes.py`` — that module imports ``agentdb``, and
anything exercising ``agenteval`` should be able to run without it, for the same
reason CI forbids the import.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from agenteval.execution import QueryExecutor
from agenteval.models.base import ModelClient, ModelResponse, Turn
from agenteval.systems.base import (
    Attempt,
    EmittedQuery,
    ModelSpec,
    SystemUnderTest,
    TokenUsage,
)
from agenteval.systems.raw_schema import RawSchemaSystem
from agenteval.tasks import Engine, Task

HITS_DDL = """CREATE TABLE hits (
    CounterID UInt32,
    EventDate Date,
    UserID UInt64,
    SearchEngineID UInt16,
    MobilePhone UInt8
) ENGINE = MergeTree ORDER BY (CounterID, EventDate, UserID)"""

OK = EmittedQuery(
    sql="",
    succeeded=True,
    columns=("count()",),
    rows=((99997497,),),
    row_count=1,
    rows_read=99997497,
    bytes_read=1024,
    duration_ms=12,
)

SYNTAX_ERROR = EmittedQuery(
    sql="",
    succeeded=False,
    error_class="syntax",
    error_text="Syntax error: failed at position 8",
)


def sample_task(task_id: str = "clickbench_nl_001") -> Task:
    return Task(
        id=task_id,
        suite="clickbench_nl",
        engines=("clickhouse",),
        question="How many rows are in the hits table?",
        gold_sql="SELECT count() FROM hits",
    )


@dataclass
class FakeExecutor:
    """A scripted :class:`QueryExecutor` that records everything it was asked."""

    engine: Engine = "clickhouse"
    schema: str = HITS_DDL
    outcomes: list[EmittedQuery] = field(default_factory=list)
    executed: list[str] = field(default_factory=list)
    namespaces: list[str] = field(default_factory=list)
    closed: bool = False
    """Set by :meth:`aclose`, so a test can assert the run released its engine."""

    async def schema_text(self, namespace: str) -> str:
        self.namespaces.append(namespace)
        return self.schema

    async def run(self, sql: str) -> EmittedQuery:
        self.executed.append(sql)
        outcome = self.outcomes.pop(0) if self.outcomes else OK
        return replace(outcome, sql=sql)

    async def aclose(self) -> None:
        self.closed = True


class ScriptExhaustedError(AssertionError):
    """The system asked for more completions than the test scripted."""


@dataclass
class ScriptedModelClient:
    """A :class:`ModelClient` that replays canned replies in order."""

    replies: list[str] = field(default_factory=list)
    provider: str = "fake"
    tokens: TokenUsage = field(
        default_factory=lambda: TokenUsage(input_tokens=100, output_tokens=20)
    )
    calls: list[tuple[str, tuple[Turn, ...], ModelSpec, int]] = field(default_factory=list)

    async def complete(
        self, *, system: str, turns: tuple[Turn, ...], model: ModelSpec, seed: int
    ) -> ModelResponse:
        if not self.replies:
            raise ScriptExhaustedError(f"no reply scripted for call {len(self.calls) + 1}")
        self.calls.append((system, turns, model, seed))
        return ModelResponse(text=self.replies.pop(0), tokens=self.tokens)


MODEL = ModelSpec(provider="anthropic", name="claude-opus-5")


@dataclass
class StubSystem:
    """A :class:`SystemUnderTest` that answers with the gold query, or explodes."""

    name: str = "S_stub"
    version: str = "0.1"
    controls_model: bool = True
    config_fingerprint: str = "sha256:stub"
    error: Exception | None = None
    queries: tuple[EmittedQuery, ...] | None = None
    calls: list[tuple[str, ModelSpec | None, int]] = field(default_factory=list)

    async def answer(self, task: Task, model: ModelSpec | None, seed: int) -> Attempt:
        self.calls.append((task.id, model, seed))
        if self.error is not None:
            raise self.error
        emitted = self.queries if self.queries is not None else (replace(OK, sql=task.gold_sql),)
        return Attempt(
            system=self.name,
            task_id=task.id,
            seed=seed,
            model=model,
            prompt=f"answer this: {task.question}",
            queries=emitted,
            tokens=TokenUsage(input_tokens=100, output_tokens=20),
            context_bytes=len(HITS_DDL),
            wall_clock_ms=5,
        )


# Import-time proof that the doubles really satisfy the protocols they stand in
# for; mypy checks these assignments, so a drifting protocol fails the build.
_EXECUTOR_CHECK: QueryExecutor = FakeExecutor()
_CLIENT_CHECK: ModelClient = ScriptedModelClient()
_SYSTEM_CHECK: SystemUnderTest = RawSchemaSystem.create(
    executor=FakeExecutor(), client=ScriptedModelClient()
)
_STUB_CHECK: SystemUnderTest = StubSystem()
