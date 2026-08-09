"""Any MCP server, scored (SPEC §11.5) — ``S1``, ``S2``, and ``S4``.

The arm that makes this project a scoreboard instead of a self-report. It knows
nothing about the server it drives: it discovers the tools, hands them to a
model, forwards whatever the model calls, and records everything.

**How a third-party system is graded.** The system answers by calling its own
tools against its own connection, so what comes back is text, not typed rows —
and grading text against a gold result set would be inventing a comparison.
Instead the harness takes the SQL the system actually emitted (read out of its
query-tool calls, or out of its final message) and re-executes it through the
harness's own read-only connection. Grading is then byte-identical across every
arm in the table.

That is a deliberate, stated limitation, not a papered-over one: it measures the
SQL a system produces, not its result formatting or its own execution path. The
full tool transcript is committed to the trace either way, so a reader can see
exactly what the server was asked and what it said.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from agenteval.execution import QueryExecutor
from agenteval.mcp.base import McpSession
from agenteval.mcp.config import McpServerConfig
from agenteval.models.base import ModelError
from agenteval.models.extract import extract_sql
from agenteval.models.tools import (
    ToolCall,
    ToolDefinition,
    ToolOutcome,
    ToolTurn,
    ToolUsingClient,
)
from agenteval.systems.base import Attempt, EmittedQuery, ModelSpec, TokenUsage
from agenteval.tasks import Engine, Task

DEFAULT_MAX_TURNS = 8
"""Tool round-trips allowed before the arm gives up. A server that loops forever
scores a no-answer rather than stalling the run."""

NO_TOOLS_NOTE = "the server advertised no usable tools"
NO_SQL_NOTE = "the system finished without emitting a SQL query"
TURN_LIMIT_NOTE = "the system hit the tool-call limit without answering"

_ENGINE_LABEL: Mapping[Engine, str] = {"clickhouse": "ClickHouse", "postgres": "PostgreSQL"}

_SYSTEM_PROMPT = """You are a data analyst answering questions about a {label} database.

Use the tools available to explore the schema and run queries.
When you are done, reply with the single SQL query that answers the question,
inside a ```sql fenced block."""

_QUESTION_TURN = "Question: {question}"


def build_system_prompt(engine: Engine) -> str:
    return _SYSTEM_PROMPT.format(label=_ENGINE_LABEL[engine])


def query_from_call(call: ToolCall, config: McpServerConfig) -> str | None:
    """The SQL a tool call carries, if that call is the system running a query."""
    if call.name not in config.query_tools:
        return None
    raw = call.arguments.get(config.query_argument)
    if not isinstance(raw, str) or not raw.strip():
        return None
    return raw.strip()


def select_tools(
    advertised: tuple[ToolDefinition, ...], config: McpServerConfig
) -> tuple[ToolDefinition, ...]:
    """Narrow to the configured allowlist, keeping the server's own order.

    An allowlist that names a tool the server does not offer is a config error
    the reader must see: silently running a differently-equipped system would
    make the row a lie.
    """
    if not config.tools:
        return advertised

    available = {tool.name for tool in advertised}
    missing = [name for name in config.tools if name not in available]
    if missing:
        raise ModelError(
            f"server {config.name!r} does not advertise tool(s) {', '.join(missing)}; "
            f"it offers {sorted(available)}"
        )
    return tuple(tool for tool in advertised if tool.name in set(config.tools))


@dataclass(frozen=True, slots=True)
class McpSystem:
    """One MCP server under test."""

    session: McpSession
    client: ToolUsingClient
    executor: QueryExecutor
    config: McpServerConfig
    tools: tuple[ToolDefinition, ...]
    max_turns: int = DEFAULT_MAX_TURNS

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def version(self) -> str:
        return self.config.version

    @property
    def controls_model(self) -> bool:
        return True

    @property
    def config_fingerprint(self) -> str:
        return self.config.fingerprint

    @classmethod
    async def create(
        cls,
        *,
        session: McpSession,
        client: ToolUsingClient,
        executor: QueryExecutor,
        config: McpServerConfig,
        max_turns: int = DEFAULT_MAX_TURNS,
    ) -> McpSystem:
        """Discover the server's tools once, at the start of the run."""
        if max_turns < 1:
            raise ValueError(f"max_turns must be >= 1, got {max_turns}")

        advertised = tuple(
            ToolDefinition(
                name=spec.name, description=spec.description, input_schema=spec.input_schema
            )
            for spec in await session.list_tools()
        )
        return cls(
            session=session,
            client=client,
            executor=executor,
            config=config,
            tools=select_tools(advertised, config),
            max_turns=max_turns,
        )

    async def answer(self, task: Task, model: ModelSpec | None, seed: int) -> Attempt:
        if model is None:
            raise ModelError(f"{self.name} chooses no model of its own; pass a ModelSpec")

        started = perf_counter()
        system_prompt = build_system_prompt(self.executor.engine)
        prompt = _QUESTION_TURN.format(question=task.question)

        if not self.tools:
            return self._attempt(task, model, seed, started, notes=(NO_TOOLS_NOTE,))

        history: list[ToolTurn] = []
        notes: list[str] = []
        context_bytes = 0
        input_tokens = 0
        output_tokens = 0
        last_query: str | None = None
        final_sql: str | None = None

        for turn_index in range(self.max_turns):
            response = await self.client.complete_with_tools(
                system=system_prompt,
                prompt=prompt,
                history=tuple(history),
                tools=self.tools,
                model=model,
                seed=seed,
            )
            input_tokens += response.tokens.input_tokens
            output_tokens += response.tokens.output_tokens

            if not response.wants_tools:
                final_sql = extract_sql(response.text) or last_query
                break

            outcomes: list[ToolOutcome] = []
            for call in response.calls:
                result = await self.session.call_tool(call.name, call.arguments)
                context_bytes += len(result.content.encode("utf-8"))
                outcomes.append(
                    ToolOutcome(call_id=call.id, content=result.content, is_error=result.is_error)
                )
                last_query = query_from_call(call, self.config) or last_query

            history.append(
                ToolTurn(text=response.text, calls=response.calls, outcomes=tuple(outcomes))
            )
            if turn_index == self.max_turns - 1:
                final_sql = last_query
                notes.append(TURN_LIMIT_NOTE)

        queries: tuple[EmittedQuery, ...] = ()
        if final_sql is None:
            notes.append(NO_SQL_NOTE)
        else:
            queries = (await self.executor.run(final_sql),)

        return self._attempt(
            task,
            model,
            seed,
            started,
            prompt=f"{system_prompt}\n\n{prompt}",
            queries=queries,
            tokens=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
            context_bytes=context_bytes,
            notes=tuple(notes),
            transcript=history,
        )

    def _attempt(
        self,
        task: Task,
        model: ModelSpec,
        seed: int,
        started: float,
        *,
        prompt: str | None = None,
        queries: tuple[EmittedQuery, ...] = (),
        tokens: TokenUsage | None = None,
        context_bytes: int = 0,
        notes: tuple[str, ...] = (),
        transcript: list[ToolTurn] | None = None,
    ) -> Attempt:
        return Attempt(
            system=self.name,
            task_id=task.id,
            seed=seed,
            model=model,
            prompt=prompt,
            queries=queries,
            tokens=tokens or TokenUsage(),
            context_bytes=context_bytes,
            wall_clock_ms=round((perf_counter() - started) * 1000),
            notes=(*notes, *_transcript_notes(transcript or [])),
        )


def _transcript_notes(history: list[ToolTurn]) -> tuple[str, ...]:
    """One note per tool call, so the trace shows what the server was asked."""
    return tuple(
        f"tool {call.name} {_compact(call.arguments)}" for turn in history for call in turn.calls
    )


def _compact(arguments: Mapping[str, Any]) -> str:
    return "{" + ", ".join(f"{key}={value!r}" for key, value in sorted(arguments.items())) + "}"
