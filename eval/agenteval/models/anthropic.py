"""Anthropic model adapter — the primary models for every arm (SPEC §11.3).

The SDK is reached through an injected ``create`` callable rather than imported
at module scope. That keeps ``anthropic`` an optional dependency (a reader
reproducing the open-weights arm should not need an API key to import the
harness) and makes the adapter testable without a network call.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Protocol, cast, runtime_checkable

from agenteval.models.base import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    ModelError,
    ModelResponse,
    Turn,
)
from agenteval.systems.base import ModelSpec, TokenUsage

API_KEY_ENV = "ANTHROPIC_API_KEY"

PROVIDER = "anthropic"


@runtime_checkable
class TextBlock(Protocol):
    """A content block carrying text. Others (tool use, thinking) are skipped."""

    text: str


class Usage(Protocol):
    input_tokens: int
    output_tokens: int


class Message(Protocol):
    content: Sequence[object]
    usage: Usage
    stop_reason: str | None


class MessageCreate(Protocol):
    """The one SDK entry point this adapter uses."""

    async def __call__(
        self,
        *,
        model: str,
        max_tokens: int,
        temperature: float,
        system: str,
        messages: list[dict[str, str]],
    ) -> Message: ...


Importer = Callable[[str], ModuleType]


def build_create(
    *, api_key: str | None = None, importer: Importer = importlib.import_module
) -> MessageCreate:
    """Bind ``messages.create`` on a live client, failing with an actionable message.

    Both failure modes — no package, no key — are things a reader hits while
    reproducing the benchmark, so neither is allowed to surface as a traceback
    from three frames deep in a vendor SDK.
    """
    key = api_key or os.environ.get(API_KEY_ENV)
    if not key:
        raise ModelError(f"{API_KEY_ENV} is not set; export it or pass api_key=")

    try:
        module = importer(PROVIDER)
    except ImportError as exc:
        raise ModelError(
            "the 'anthropic' package is not installed; install the optional "
            "extra with: uv sync --extra anthropic"
        ) from exc

    client = module.AsyncAnthropic(api_key=key)
    return cast(MessageCreate, client.messages.create)


@dataclass(frozen=True, slots=True)
class AnthropicClient:
    """A :class:`~agenteval.models.base.ModelClient` backed by the Messages API."""

    create: MessageCreate
    provider: str = PROVIDER

    async def complete(
        self,
        *,
        system: str,
        turns: tuple[Turn, ...],
        model: ModelSpec,
        seed: int,
    ) -> ModelResponse:
        """One completion.

        The Messages API exposes no sampling seed, so ``seed`` cannot be pushed
        down to the provider; the runner records it against the attempt and
        SPEC §11.4's repetitions supply the variance instead.
        """
        try:
            message = await self.create(
                model=model.name,
                max_tokens=model.max_output_tokens or DEFAULT_MAX_OUTPUT_TOKENS,
                temperature=model.temperature,
                system=system,
                messages=[{"role": turn.role, "content": turn.content} for turn in turns],
            )
        except Exception as exc:
            raise ModelError(f"anthropic call failed for {model} at seed {seed}: {exc}") from exc

        return _to_response(message)


def _to_response(message: Message) -> ModelResponse:
    text = "\n".join(block.text for block in message.content if isinstance(block, TextBlock))
    return ModelResponse(
        text=text,
        tokens=TokenUsage(
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
        ),
        stop_reason=message.stop_reason,
    )
