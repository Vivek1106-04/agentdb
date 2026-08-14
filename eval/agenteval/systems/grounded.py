"""Arms whose only difference from ``A0_baseline`` is a richer context payload.

``A1_stats``, ``A2_layout`` and every later Family A arm are this class with a
different :class:`ContextProvider`. The model, the prompt, the retry loop and the
token accounting are the ones ``A0_baseline`` uses, so a difference in score is
attributable to the payload and to nothing else (SPEC §11.3).

**The provider is reached structurally, never by import.** agenteval does not
know that agentdb exists, and this is the seam where that rule earns its keep: a
provider is anything with a ``context`` coroutine, so a third party can score
their own grounding service against the same tasks without either project
depending on the other (SPEC §4.1.6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from agenteval.execution import QueryExecutor
from agenteval.models.base import ModelClient, ModelError
from agenteval.systems.base import Attempt, ModelSpec
from agenteval.systems.fingerprint import config_fingerprint
from agenteval.systems.loop import answer_with_model
from agenteval.systems.raw_schema import build_question_turn, build_system_prompt
from agenteval.tasks import Task

VERSION = "1.0"
"""Bumped whenever the arm's prompt or loop changes, because both change the number."""

DEFAULT_MAX_RETRIES = 2


@runtime_checkable
class ContextProvider(Protocol):
    """Something that assembles the context payload for a question.

    ``fingerprint`` covers the provider's effective configuration — grounding
    level, sample fraction, column budget. It is committed with the run, because
    "A2 scored 61%" means nothing without saying which A2.
    """

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def fingerprint(self) -> str: ...

    async def context(self, *, namespace: str, question: str) -> str:
        """The payload to put in front of the model, as text."""
        ...


@dataclass(frozen=True, slots=True)
class GroundedSystem:
    """A model, a context provider, and the same retry loop every arm uses."""

    arm: str
    provider: ContextProvider
    executor: QueryExecutor
    client: ModelClient
    config_fingerprint: str
    max_retries: int = DEFAULT_MAX_RETRIES
    version: str = VERSION
    controls_model: bool = True

    @property
    def name(self) -> str:
        return self.arm

    @classmethod
    def create(
        cls,
        *,
        arm: str,
        provider: ContextProvider,
        executor: QueryExecutor,
        client: ModelClient,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> GroundedSystem:
        """Build the arm and derive its fingerprint from everything that moves the number."""
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")

        return cls(
            arm=arm,
            provider=provider,
            executor=executor,
            client=client,
            max_retries=max_retries,
            config_fingerprint=config_fingerprint(
                {
                    "arm": arm,
                    "version": VERSION,
                    "engine": executor.engine,
                    "max_retries": max_retries,
                    "provider": client.provider,
                    "system_prompt": build_system_prompt(executor.engine),
                    "context_provider": provider.name,
                    "context_provider_version": provider.version,
                    "context_fingerprint": provider.fingerprint,
                }
            ),
        )

    async def answer(self, task: Task, model: ModelSpec | None, seed: int) -> Attempt:
        """Answer ``task`` once, on the grounded payload instead of a schema dump."""
        if model is None:
            raise ModelError(f"{self.arm} chooses no model of its own; pass a ModelSpec")

        context = await self.provider.context(namespace=task.namespace, question=task.question)
        return await answer_with_model(
            system_name=self.arm,
            task=task,
            model=model,
            seed=seed,
            executor=self.executor,
            client=self.client,
            system_prompt=build_system_prompt(self.executor.engine),
            first_turn=build_question_turn(task, context),
            context_bytes=len(context.encode("utf-8")),
            max_retries=self.max_retries,
        )
