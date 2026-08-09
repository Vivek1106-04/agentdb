"""Family S: driving a server the harness did not write, and grading it fairly."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pytest

from agenteval.mcp.base import ToolResult, ToolSpec
from agenteval.mcp.config import parse_server
from agenteval.models.base import ModelError
from agenteval.models.tools import (
    ToolCall,
    ToolDefinition,
    ToolResponse,
    ToolTurn,
)
from agenteval.systems.base import ModelSpec, SystemUnderTest, TokenUsage
from agenteval.systems.mcp_generic import (
    NO_SQL_NOTE,
    NO_TOOLS_NOTE,
    TURN_LIMIT_NOTE,
    McpSystem,
    query_from_call,
    select_tools,
)
from tests.harness_fakes import MODEL, FakeExecutor, sample_task

SERVER = {
    "name": "S1_mcp_clickhouse",
    "version": "0.1.12",
    "command": "uvx",
    "query_tools": ["run_select_query"],
}

RUN_QUERY = ToolSpec(name="run_select_query", description="run SQL")
LIST_TABLES = ToolSpec(name="list_tables", description="list tables")


@dataclass
class FakeSession:
    tools: tuple[ToolSpec, ...] = (RUN_QUERY, LIST_TABLES)
    results: dict[str, ToolResult] = field(default_factory=dict)
    calls: list[tuple[str, Mapping[str, Any]]] = field(default_factory=list)
    closed: bool = False

    async def list_tools(self) -> tuple[ToolSpec, ...]:
        return self.tools

    async def call_tool(self, name: str, arguments: Mapping[str, Any]) -> ToolResult:
        self.calls.append((name, arguments))
        return self.results.get(name, ToolResult(content="ok"))

    async def close(self) -> None:
        self.closed = True


@dataclass
class ScriptedToolClient:
    responses: list[ToolResponse] = field(default_factory=list)
    provider: str = "fake"
    seen_tools: list[tuple[ToolDefinition, ...]] = field(default_factory=list)
    histories: list[tuple[ToolTurn, ...]] = field(default_factory=list)

    async def complete_with_tools(
        self,
        *,
        system: str,
        prompt: str,
        history: tuple[ToolTurn, ...],
        tools: tuple[ToolDefinition, ...],
        model: ModelSpec,
        seed: int,
    ) -> ToolResponse:
        self.seen_tools.append(tools)
        self.histories.append(history)
        if not self.responses:
            raise AssertionError("no response scripted")
        return self.responses.pop(0)


def _text(sql: str) -> ToolResponse:
    return ToolResponse(text=f"```sql\n{sql}\n```", tokens=TokenUsage(10, 5))


def _call(name: str = "run_select_query", **arguments: Any) -> ToolResponse:
    return ToolResponse(
        text="let me look",
        calls=(ToolCall(id="c1", name=name, arguments=arguments),),
        tokens=TokenUsage(10, 5),
    )


async def _system(
    *responses: ToolResponse,
    session: FakeSession | None = None,
    executor: FakeExecutor | None = None,
    max_turns: int = 8,
    server: dict[str, Any] | None = None,
) -> tuple[McpSystem, FakeSession, FakeExecutor, ScriptedToolClient]:
    live = session or FakeSession()
    engine = executor or FakeExecutor()
    client = ScriptedToolClient(responses=list(responses))
    system = await McpSystem.create(
        session=live,
        client=client,
        executor=engine,
        config=parse_server(server or SERVER),
        max_turns=max_turns,
    )
    return system, live, engine, client


# --------------------------------------------------------------------------
# tool selection
# --------------------------------------------------------------------------


def test_an_empty_allowlist_means_the_servers_own_toolset() -> None:
    # Narrowing by default would measure a differently-equipped system than the
    # one a user actually installs
    advertised = (ToolDefinition("a", ""), ToolDefinition("b", ""))

    assert select_tools(advertised, parse_server(SERVER)) == advertised


def test_an_allowlist_narrows_but_keeps_the_servers_order() -> None:
    advertised = (ToolDefinition("a", ""), ToolDefinition("b", ""), ToolDefinition("c", ""))

    selected = select_tools(advertised, parse_server({**SERVER, "tools": ["c", "a"]}))

    assert [tool.name for tool in selected] == ["a", "c"]


def test_an_allowlist_naming_an_absent_tool_is_refused() -> None:
    with pytest.raises(ModelError, match="does not advertise tool\\(s\\) ghost"):
        select_tools((ToolDefinition("a", ""),), parse_server({**SERVER, "tools": ["ghost"]}))


def test_the_sql_is_read_out_of_a_query_tool_call() -> None:
    config = parse_server(SERVER)

    assert query_from_call(ToolCall("1", "run_select_query", {"query": " SELECT 1 "}), config) == (
        "SELECT 1"
    )


@pytest.mark.parametrize(
    "call",
    [
        ToolCall("1", "list_tables", {"query": "SELECT 1"}),
        ToolCall("1", "run_select_query", {}),
        ToolCall("1", "run_select_query", {"query": "   "}),
        ToolCall("1", "run_select_query", {"query": 42}),
    ],
)
def test_a_call_that_carries_no_query_yields_nothing(call: ToolCall) -> None:
    assert query_from_call(call, parse_server(SERVER)) is None


# --------------------------------------------------------------------------
# the arm
# --------------------------------------------------------------------------


async def test_it_is_a_system_under_test() -> None:
    system, _, _, _ = await _system()

    assert isinstance(system, SystemUnderTest)
    assert system.name == "S1_mcp_clickhouse"
    assert system.version == "0.1.12"
    assert system.controls_model is True
    assert system.config_fingerprint.startswith("sha256:")


async def test_tools_are_discovered_once_and_handed_to_the_model() -> None:
    system, _, _, client = await _system(_text("SELECT 1"))

    await system.answer(sample_task(), MODEL, seed=0)

    assert [tool.name for tool in client.seen_tools[0]] == ["run_select_query", "list_tables"]


async def test_a_negative_turn_budget_is_refused() -> None:
    with pytest.raises(ValueError, match="max_turns must be >= 1"):
        await _system(max_turns=0)


async def test_a_server_with_no_tools_is_recorded_not_crashed() -> None:
    system, _, _, _ = await _system(session=FakeSession(tools=()))

    attempt = await system.answer(sample_task(), MODEL, seed=0)

    assert attempt.notes == (NO_TOOLS_NOTE,)
    assert attempt.queries == ()


async def test_the_arm_refuses_to_invent_a_model() -> None:
    system, _, _, _ = await _system()

    with pytest.raises(ModelError, match="chooses no model of its own"):
        await system.answer(sample_task(), None, seed=0)


async def test_tool_calls_are_forwarded_and_their_results_returned() -> None:
    # Arrange — the server is asked, then the model answers
    session = FakeSession(results={"list_tables": ToolResult(content="hits")})
    system, live, _, client = await _system(
        _call("list_tables"), _text("SELECT count() FROM hits"), session=session
    )

    # Act
    attempt = await system.answer(sample_task(), MODEL, seed=0)

    # Assert
    assert live.calls == [("list_tables", {})]
    assert client.histories[1][0].outcomes[0].content == "hits"
    assert attempt.queries[0].sql == "SELECT count() FROM hits"


async def test_grading_re_executes_the_systems_sql_through_the_harness() -> None:
    # Arrange — the server returns text, so grading it directly would be inventing
    # a comparison; every arm is graded on the same connection instead
    system, _, executor, _ = await _system(
        _call(query="SELECT count() FROM hits"), _text("SELECT count() FROM hits")
    )

    attempt = await system.answer(sample_task(), MODEL, seed=0)

    assert executor.executed == ["SELECT count() FROM hits"]
    assert attempt.queries[0].succeeded is True
    assert attempt.queries[0].rows == ((99997497,),)


async def test_the_last_query_tool_call_stands_in_for_a_missing_final_block() -> None:
    # A system that ran the right query but did not repeat it in prose still
    # answered; scoring that as "no query" would grade formatting
    system, _, _, _ = await _system(
        _call(query="SELECT count() FROM hits"),
        ToolResponse(text="The answer is 99,997,497.", tokens=TokenUsage(1, 1)),
    )

    attempt = await system.answer(sample_task(), MODEL, seed=0)

    assert attempt.queries[0].sql == "SELECT count() FROM hits"


async def test_a_system_that_never_emits_sql_is_noted() -> None:
    system, _, _, _ = await _system(ToolResponse(text="I cannot answer that."))

    attempt = await system.answer(sample_task(), MODEL, seed=0)

    assert attempt.queries == ()
    assert NO_SQL_NOTE in attempt.notes


async def test_a_server_that_loops_forever_hits_the_turn_limit() -> None:
    system, live, _, _ = await _system(_call("list_tables"), _call("list_tables"), max_turns=2)

    attempt = await system.answer(sample_task(), MODEL, seed=0)

    assert len(live.calls) == 2
    assert TURN_LIMIT_NOTE in attempt.notes
    assert NO_SQL_NOTE in attempt.notes


async def test_a_loop_that_ran_a_query_is_graded_on_it() -> None:
    system, _, executor, _ = await _system(
        _call(query="SELECT 1"), _call(query="SELECT 2"), max_turns=2
    )

    attempt = await system.answer(sample_task(), MODEL, seed=0)

    assert TURN_LIMIT_NOTE in attempt.notes
    assert executor.executed == ["SELECT 2"]
    assert attempt.queries[0].sql == "SELECT 2"


async def test_the_trace_records_what_the_server_was_asked() -> None:
    system, _, _, _ = await _system(_call("list_tables", database="agentdb"), _text("SELECT 1"))

    attempt = await system.answer(sample_task(), MODEL, seed=0)

    assert "tool list_tables {database='agentdb'}" in attempt.notes


async def test_tool_output_counts_as_the_injected_context() -> None:
    # Grounding is not free, and Family S has to pay for it in the same column
    session = FakeSession(results={"list_tables": ToolResult(content="hits" * 100)})
    system, _, _, _ = await _system(_call("list_tables"), _text("SELECT 1"), session=session)

    attempt = await system.answer(sample_task(), MODEL, seed=0)

    assert attempt.context_bytes == 400


async def test_tokens_accumulate_across_tool_round_trips() -> None:
    system, _, _, _ = await _system(_call("list_tables"), _text("SELECT 1"))

    attempt = await system.answer(sample_task(), MODEL, seed=0)

    assert attempt.tokens.input_tokens == 20
    assert attempt.tokens.output_tokens == 10
