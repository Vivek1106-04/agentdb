"""Tool use, in provider-neutral terms.

Family S measures systems that answer by *calling tools*, not by emitting text.
The conversation shape that requires is modelled here once, so an arm describes
what happened — the assistant said this, called these tools, got these back —
without ever naming a provider's message format. The adapter renders it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from agenteval.systems.base import ModelSpec, TokenUsage


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """A tool offered to the model.

    Declared here rather than reusing ``mcp.ToolSpec`` so the model layer stays
    ignorant of where tools come from: a future arm could offer tools that are
    not MCP tools at all.
    """

    name: str
    description: str
    input_schema: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One tool invocation the model asked for."""

    id: str
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    """What one call returned, paired back to it by ``call_id``."""

    call_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class ToolTurn:
    """One completed exchange: what the model did, and what came back."""

    text: str
    calls: tuple[ToolCall, ...] = ()
    outcomes: tuple[ToolOutcome, ...] = ()


@dataclass(frozen=True, slots=True)
class ToolResponse:
    """A completion that may ask for tools before it answers."""

    text: str
    calls: tuple[ToolCall, ...] = ()
    tokens: TokenUsage = field(default_factory=TokenUsage)
    stop_reason: str | None = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.calls)


class ToolUsingClient(Protocol):
    """A model that can be handed tools."""

    @property
    def provider(self) -> str: ...

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
        """Continue the conversation, optionally calling tools."""
        ...
