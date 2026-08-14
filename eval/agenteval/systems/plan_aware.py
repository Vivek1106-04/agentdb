"""``A3_plan`` — the arm that sees the plan before the query runs (SPEC §11.3).

A2 tells the model what the table looks like. A3 adds the one thing no amount of
schema description can supply: what the engine says it will *do* with the query
the model just wrote — how much it can prune, which index fires, what it will
read anyway.

Two deliberate choices, both stated because they shape the number:

* **The review is mandatory, not agent-elected.** The spec describes a tool the
  agent may call. Making it a fixed step keeps A3 on the same model interface as
  A0 to A2 — no tool-calling API, no provider-specific plumbing — so the arms stay
  comparable and any model with a text endpoint can be scored. What it measures
  is therefore "does plan feedback help", not "will an agent think to ask".
* **The review costs a model call, never an execution.** ``max_retries`` counts
  queries sent to the engine, so A3 gets exactly the attempts A2 gets. The extra
  cost shows up where it should: in the token column of the report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from agenteval.execution import QueryExecutor
from agenteval.models.base import ModelClient, ModelError
from agenteval.systems.base import Attempt, ModelSpec
from agenteval.systems.fingerprint import config_fingerprint
from agenteval.systems.grounded import DEFAULT_MAX_RETRIES, ContextProvider
from agenteval.systems.loop import answer_with_model
from agenteval.systems.raw_schema import build_question_turn, build_system_prompt
from agenteval.tasks import Task

VERSION = "1.0"

REVIEW_TURN = """Before this runs, here is what the engine says it would do with that query:

{plan}

Reply with your final SQL query — the same one if the plan looks right."""


@runtime_checkable
class PlanAdvisor(Protocol):
    """Something that can explain a draft query against the engine.

    Structural, like :class:`~agenteval.systems.grounded.ContextProvider`: the
    harness scores plan feedback from anyone who can produce it, and imports
    nobody to do so.
    """

    async def explain_plan(self, *, sql: str, namespace: str) -> str | None:
        """Feedback on ``sql``, or ``None`` when there is nothing worth saying."""
        ...


def build_review_turn(plan: str) -> str:
    """What the model is shown about its own draft."""
    return REVIEW_TURN.format(plan=plan)


@dataclass(frozen=True, slots=True)
class PlanAwareSystem:
    """A grounded arm that shows the model its plan before the query runs."""

    arm: str
    provider: ContextProvider
    advisor: PlanAdvisor
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
        advisor: PlanAdvisor,
        executor: QueryExecutor,
        client: ModelClient,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> PlanAwareSystem:
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")

        return cls(
            arm=arm,
            provider=provider,
            advisor=advisor,
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
                    "review_turn": REVIEW_TURN,
                    "context_provider": provider.name,
                    "context_provider_version": provider.version,
                    "context_fingerprint": provider.fingerprint,
                }
            ),
        )

    async def answer(self, task: Task, model: ModelSpec | None, seed: int) -> Attempt:
        if model is None:
            raise ModelError(f"{self.arm} chooses no model of its own; pass a ModelSpec")

        context = await self.provider.context(namespace=task.namespace, question=task.question)

        async def review(sql: str) -> str | None:
            plan = await self.advisor.explain_plan(sql=sql, namespace=task.namespace)
            return None if plan is None else build_review_turn(plan)

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
            review=review,
        )
