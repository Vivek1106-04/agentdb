"""Anthropic tool use — the model half of every Family S arm.

Separate from :mod:`agenteval.models.anthropic` because it is a different call
shape, not a flag on the same one: a request carrying tools, a reply that may be
a request to call them, and a conversation that has to carry ``tool_use`` and
``tool_result`` blocks back. Keeping them apart means the A0 path cannot
accidentally acquire tools and stop being A0.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from agenteval.models.anthropic import PROVIDER, Message, TextBlock
from agenteval.models.base import DEFAULT_MAX_OUTPUT_TOKENS, ModelError
from agenteval.models.tools import ToolCall, ToolDefinition, ToolResponse, ToolTurn
from agenteval.systems.base import ModelSpec, TokenUsage

TOOL_USE = "tool_use"
TOOL_RESULT = "tool_result"


@runtime_checkable
class ToolUseBlock(Protocol):
    """A ``tool_use`` content block."""

    id: str
    name: str
    input: dict[str, Any]


class MessageCreateWithTools(Protocol):
    """``messages.create``, given tools."""

    async def __call__(
        self,
        *,
        model: str,
        max_tokens: int,
        temperature: float,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Message: ...


@dataclass(frozen=True, slots=True)
class AnthropicToolClient:
    """A :class:`ToolUsingClient` over the Messages API."""

    create: MessageCreateWithTools
    provider: str = PROVIDER

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
        try:
            message = await self.create(
                model=model.name,
                max_tokens=model.max_output_tokens or DEFAULT_MAX_OUTPUT_TOKENS,
                temperature=model.temperature,
                system=system,
                messages=render_messages(prompt, history),
                tools=[_render_tool(tool) for tool in tools],
            )
        except Exception as exc:
            raise ModelError(f"anthropic call failed for {model} at seed {seed}: {exc}") from exc

        return _to_response(message)


def render_messages(prompt: str, history: Sequence[ToolTurn]) -> list[dict[str, Any]]:
    """Turn the neutral conversation into Messages API blocks.

    A turn becomes two messages — the assistant's text and tool calls, then the
    tool results as a user message — which is the shape the API requires and the
    reason the neutral form exists at all.
    """
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    for turn in history:
        assistant: list[dict[str, Any]] = []
        if turn.text:
            assistant.append({"type": "text", "text": turn.text})
        assistant += [
            {"type": TOOL_USE, "id": call.id, "name": call.name, "input": dict(call.arguments)}
            for call in turn.calls
        ]
        messages.append({"role": "assistant", "content": assistant})

        if turn.outcomes:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": TOOL_RESULT,
                            "tool_use_id": outcome.call_id,
                            "content": outcome.content,
                            "is_error": outcome.is_error,
                        }
                        for outcome in turn.outcomes
                    ],
                }
            )
    return messages


def _render_tool(tool: ToolDefinition) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": dict(tool.input_schema) or {"type": "object", "properties": {}},
    }


def _to_response(message: Message) -> ToolResponse:
    texts = [block.text for block in message.content if isinstance(block, TextBlock)]
    calls = tuple(
        ToolCall(id=str(block.id), name=str(block.name), arguments=dict(block.input or {}))
        for block in message.content
        if isinstance(block, ToolUseBlock)
    )
    return ToolResponse(
        text="\n".join(texts),
        calls=calls,
        tokens=TokenUsage(
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
        ),
        stop_reason=message.stop_reason,
    )
