"""``A0_baseline`` — table DDL only, no server (SPEC §11.3).

The floor of the ablation ladder, and the arm every other number is read
against. It gives a model exactly what the official ClickHouse MCP server gives
one: the names and ``CREATE TABLE`` DDL of the tables, nothing about
cardinality, physical layout, or plans. If grounding does not beat this, the
context layer is not earning its cost, and the report has to say so.

It is also the only arm with no moving parts, which makes it the arm that tells
you whether a disappointing result came from the idea or from the harness.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from agenteval.execution import QueryExecutor
from agenteval.models.base import ModelClient, ModelError
from agenteval.systems.base import Attempt, ModelSpec
from agenteval.systems.fingerprint import config_fingerprint
from agenteval.systems.loop import answer_with_model
from agenteval.tasks import Engine, Task

ARM_NAME = "A0_baseline"

VERSION = "1.0"
"""Bumped whenever the prompt or the loop changes, because both change the number."""

DEFAULT_MAX_RETRIES = 2
"""Self-corrections allowed after the first query — the ``k`` in EX@k (SPEC §11.1)."""

_ENGINE_LABEL: Mapping[Engine, str] = {"clickhouse": "ClickHouse", "databricks": "Databricks SQL"}

_SYSTEM_PROMPT = """You are a data analyst writing SQL for a {label} database.

Answer the question with exactly one SQL query.
Reply with only that query inside a ```sql fenced block, and nothing else."""

_QUESTION_TURN = """These are the tables available:

{schema}

Question: {question}"""


def build_system_prompt(engine: Engine) -> str:
    """The system prompt for ``engine``. Identical across arms, so arms stay comparable."""
    return _SYSTEM_PROMPT.format(label=_ENGINE_LABEL[engine])


def build_question_turn(task: Task, schema: str) -> str:
    """The A0 payload: DDL and the question, and deliberately nothing else."""
    return _QUESTION_TURN.format(schema=schema, question=task.question)


@dataclass(frozen=True, slots=True)
class RawSchemaSystem:
    """A model, a schema dump, and a retry loop."""

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
    ) -> RawSchemaSystem:
        """Build the arm and derive its fingerprint from its effective config.

        The system prompt is part of that config: editing a prompt changes the
        number, so it has to change the fingerprint too.
        """
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
        """Answer ``task`` once, keeping every query the model emitted on the way."""
        if model is None:
            raise ModelError(f"{ARM_NAME} chooses no model of its own; pass a ModelSpec")

        schema = await self.executor.schema_text(task.namespace)
        return await answer_with_model(
            system_name=self.name,
            task=task,
            model=model,
            seed=seed,
            executor=self.executor,
            client=self.client,
            system_prompt=build_system_prompt(self.executor.engine),
            first_turn=build_question_turn(task, schema),
            context_bytes=len(schema.encode("utf-8")),
            max_retries=self.max_retries,
        )
