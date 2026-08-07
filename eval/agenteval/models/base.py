"""The model adapter interface (SPEC §11.3).

Model adapters are pluggable so a reader without an Anthropic budget can
reproduce the numbers with whatever they have — an open-weights model behind an
OpenAI-compatible endpoint, or a local runtime. Every arm records which model
answered it, so swapping one never silently changes what a table means.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from agenteval.systems.base import ModelSpec, TokenUsage

DEFAULT_MAX_OUTPUT_TOKENS = 2048
"""Enough for a long analytical query, short enough that a runaway generation
costs a task rather than a run."""


class ModelError(RuntimeError):
    """A model call could not be completed.

    Distinct from a *bad answer*: a wrong query is data, a failed API call is a
    hole in the run and the runner must be able to tell them apart.
    """


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """One completion, plus what it cost."""

    text: str
    tokens: TokenUsage = field(default_factory=TokenUsage)
    stop_reason: str | None = None


@dataclass(frozen=True, slots=True)
class Turn:
    """One message in the self-correction loop.

    ``role`` is limited to the two roles every provider agrees on, so an adapter
    never has to invent a mapping for a role a benchmark arm made up.
    """

    role: str
    content: str


@runtime_checkable
class ModelClient(Protocol):
    """Anything that can turn a conversation into text."""

    provider: str

    async def complete(
        self,
        *,
        system: str,
        turns: tuple[Turn, ...],
        model: ModelSpec,
        seed: int,
    ) -> ModelResponse:
        """Continue the conversation.

        ``seed`` is the run-repetition index (SPEC §11.4 requires ≥5 per cell).
        Providers that expose a sampling seed should pass it through; those that
        do not still receive it, because an adapter that quietly ignores it is
        better than a caller that quietly assumes determinism.
        """
        ...
