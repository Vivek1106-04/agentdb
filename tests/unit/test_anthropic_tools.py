"""Rendering the neutral tool conversation into Messages API blocks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

from agenteval.models.anthropic import Usage
from agenteval.models.anthropic_tools import (
    TOOL_RESULT,
    TOOL_USE,
    AnthropicToolClient,
    render_messages,
)
from agenteval.models.base import DEFAULT_MAX_OUTPUT_TOKENS, ModelError
from agenteval.models.tools import (
    ToolCall,
    ToolDefinition,
    ToolOutcome,
    ToolTurn,
    ToolUsingClient,
)
from agenteval.systems.base import ModelSpec

MODEL = ModelSpec(provider="anthropic", name="claude-opus-5")
RUN_QUERY = ToolDefinition(name="run_select_query", description="run SQL")


@dataclass
class FakeUsage:
    input_tokens: int = 7
    output_tokens: int = 3


@dataclass
class FakeText:
    text: str


@dataclass
class FakeToolUse:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class FakeMessage:
    content: Sequence[object]
    usage: Usage = field(default_factory=FakeUsage)
    stop_reason: str | None = "tool_use"


@dataclass
class RecordingCreate:
    message: Any
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self.message, Exception):
            raise self.message
        return self.message


def _client(message: Any) -> tuple[AnthropicToolClient, RecordingCreate]:
    create = RecordingCreate(message=message)
    return AnthropicToolClient(create=create), create


def test_it_satisfies_the_tool_using_client_protocol() -> None:
    client, _ = _client(FakeMessage(content=[]))

    checked: ToolUsingClient = client
    assert checked.provider == "anthropic"


def test_an_empty_history_is_just_the_question() -> None:
    assert render_messages("how many?", ()) == [{"role": "user", "content": "how many?"}]


def test_a_turn_becomes_an_assistant_message_and_a_tool_result_message() -> None:
    # Arrange — the two-message shape is exactly why the neutral form exists
    turn = ToolTurn(
        text="let me look",
        calls=(ToolCall(id="c1", name="list_tables", arguments={"db": "agentdb"}),),
        outcomes=(ToolOutcome(call_id="c1", content="hits"),),
    )

    messages = render_messages("how many?", [turn])

    assert messages[1] == {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "let me look"},
            {"type": TOOL_USE, "id": "c1", "name": "list_tables", "input": {"db": "agentdb"}},
        ],
    }
    assert messages[2] == {
        "role": "user",
        "content": [
            {
                "type": TOOL_RESULT,
                "tool_use_id": "c1",
                "content": "hits",
                "is_error": False,
            }
        ],
    }


def test_a_silent_tool_call_omits_the_empty_text_block() -> None:
    turn = ToolTurn(text="", calls=(ToolCall(id="c1", name="t"),), outcomes=())

    assistant = render_messages("q", [turn])[1]

    assert assistant["content"] == [{"type": TOOL_USE, "id": "c1", "name": "t", "input": {}}]


def test_a_turn_with_no_outcomes_adds_no_result_message() -> None:
    turn = ToolTurn(text="done", calls=(), outcomes=())

    assert len(render_messages("q", [turn])) == 2


def test_a_failing_tool_result_is_reported_as_such() -> None:
    turn = ToolTurn(
        text="",
        calls=(ToolCall(id="c1", name="t"),),
        outcomes=(ToolOutcome(call_id="c1", content="bad column", is_error=True),),
    )

    results = render_messages("q", [turn])[2]["content"]

    assert results[0]["is_error"] is True


async def test_a_reply_separates_text_from_tool_calls() -> None:
    client, _ = _client(
        FakeMessage(
            content=[
                FakeText("let me look"),
                FakeToolUse(id="c1", name="run_select_query", input={"query": "SELECT 1"}),
            ]
        )
    )

    response = await client.complete_with_tools(
        system="s", prompt="q", history=(), tools=(RUN_QUERY,), model=MODEL, seed=0
    )

    assert response.text == "let me look"
    assert response.wants_tools is True
    assert response.calls[0] == ToolCall(
        id="c1", name="run_select_query", arguments={"query": "SELECT 1"}
    )
    assert response.tokens.input_tokens == 7


async def test_a_reply_with_no_calls_is_a_final_answer() -> None:
    client, _ = _client(FakeMessage(content=[FakeText("```sql\nSELECT 1\n```")]))

    response = await client.complete_with_tools(
        system="s", prompt="q", history=(), tools=(RUN_QUERY,), model=MODEL, seed=0
    )

    assert response.wants_tools is False


async def test_tools_are_sent_with_a_schema_the_api_accepts() -> None:
    client, create = _client(FakeMessage(content=[]))

    await client.complete_with_tools(
        system="s", prompt="q", history=(), tools=(RUN_QUERY,), model=MODEL, seed=0
    )

    sent = create.calls[0]["tools"][0]
    assert sent["name"] == "run_select_query"
    assert sent["input_schema"] == {"type": "object", "properties": {}}
    assert create.calls[0]["max_tokens"] == DEFAULT_MAX_OUTPUT_TOKENS


async def test_a_declared_schema_is_passed_through() -> None:
    client, create = _client(FakeMessage(content=[]))
    tool = ToolDefinition(name="t", description="d", input_schema={"type": "object", "x": 1})

    await client.complete_with_tools(
        system="s", prompt="q", history=(), tools=(tool,), model=MODEL, seed=0
    )

    assert create.calls[0]["tools"][0]["input_schema"] == {"type": "object", "x": 1}


async def test_a_provider_failure_becomes_a_model_error_naming_the_seed() -> None:
    client, _ = _client(TimeoutError("upstream timed out"))

    with pytest.raises(ModelError, match="at seed 9: upstream timed out"):
        await client.complete_with_tools(
            system="s", prompt="q", history=(), tools=(RUN_QUERY,), model=MODEL, seed=9
        )
