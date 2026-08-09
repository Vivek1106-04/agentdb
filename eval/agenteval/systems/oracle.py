"""``A7_oracle`` — the gold query in context (SPEC §11.3).

The ceiling, and the only arm whose job is to fail loudly. A7 is handed the
answer; if it does not score near 100%, the gap is not a model limitation, it is
the harness lying — extraction dropping a query, comparison rejecting an
equivalent result, the engine timing out. Every other arm's number is only
meaningful relative to this one.

Reporting A7 alongside the rest is the cheapest honesty mechanism in the
project: a reader can see exactly how much headroom the measurement itself
allows before reading anything into an arm's score.
"""

from __future__ import annotations

from dataclasses import dataclass

from agenteval.execution import QueryExecutor
from agenteval.models.base import ModelClient, ModelError
from agenteval.systems.base import Attempt, ModelSpec
from agenteval.systems.fingerprint import config_fingerprint
from agenteval.systems.loop import answer_with_model
from agenteval.systems.raw_schema import build_system_prompt
from agenteval.tasks import Task

ARM_NAME = "A7_oracle"

VERSION = "1.0"

DEFAULT_MAX_RETRIES = 2

_ORACLE_TURN = """These are the tables available:

{schema}

Question: {question}

A query that is known to answer this question correctly:

{gold_sql}"""


def build_oracle_turn(task: Task, schema: str) -> str:
    """A0's payload plus the gold query.

    The question is still asked, and the model still has to emit something
    runnable — an arm that skipped straight to executing gold would measure
    nothing about the harness, which is the only thing this arm exists to test.
    """
    return _ORACLE_TURN.format(schema=schema, question=task.question, gold_sql=task.gold_sql)


@dataclass(frozen=True, slots=True)
class OracleSystem:
    """The upper bound the rest of the table is read against."""

    executor: QueryExecutor
    client: ModelClient
    config_fingerprint: str
    max_retries: int = DEFAULT_MAX_RETRIES
    name: str = ARM_NAME
    version: str = VERSION
    controls_model: bool = True

    @classmethod
    def create(
        cls,
        *,
        executor: QueryExecutor,
        client: ModelClient,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> OracleSystem:
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")

        return cls(
            executor=executor,
            client=client,
            max_retries=max_retries,
            config_fingerprint=config_fingerprint(
                {
                    "arm": ARM_NAME,
                    "version": VERSION,
                    "engine": executor.engine,
                    "max_retries": max_retries,
                    "provider": client.provider,
                    "system_prompt": build_system_prompt(executor.engine),
                }
            ),
        )

    async def answer(self, task: Task, model: ModelSpec | None, seed: int) -> Attempt:
        if model is None:
            raise ModelError(f"{ARM_NAME} chooses no model of its own; pass a ModelSpec")

        schema = await self.executor.schema_text(task.namespace)
        first_turn = build_oracle_turn(task, schema)
        return await answer_with_model(
            system_name=self.name,
            task=task,
            model=model,
            seed=seed,
            executor=self.executor,
            client=self.client,
            system_prompt=build_system_prompt(self.executor.engine),
            first_turn=first_turn,
            context_bytes=len(first_turn.encode("utf-8")),
            max_retries=self.max_retries,
        )
