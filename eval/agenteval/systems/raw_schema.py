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
from time import perf_counter

from agenteval.execution import QueryExecutor
from agenteval.models.base import ModelClient, ModelError, Turn
from agenteval.models.extract import extract_sql
from agenteval.systems.base import Attempt, EmittedQuery, ModelSpec, TokenUsage
from agenteval.systems.fingerprint import config_fingerprint
from agenteval.tasks import Engine, Task

ARM_NAME = "A0_baseline"

VERSION = "1.0"
"""Bumped whenever the prompt or the loop changes, because both change the number."""

DEFAULT_MAX_RETRIES = 2
"""Self-corrections allowed after the first query — the ``k`` in EX@k (SPEC §11.1)."""

_ENGINE_LABEL: Mapping[Engine, str] = {"clickhouse": "ClickHouse", "postgres": "PostgreSQL"}

_SYSTEM_PROMPT = """You are a data analyst writing SQL for a {label} database.

Answer the question with exactly one SQL query.
Reply with only that query inside a ```sql fenced block, and nothing else."""

_QUESTION_TURN = """These are the tables available:

{schema}

Question: {question}"""

_RETRY_TURN = """That query failed.

error_class: {error_class}
error: {error_text}

Reply with one corrected SQL query."""


def build_system_prompt(engine: Engine) -> str:
    """The system prompt for ``engine``. Identical across arms, so arms stay comparable."""
    return _SYSTEM_PROMPT.format(label=_ENGINE_LABEL[engine])


def build_question_turn(task: Task, schema: str) -> str:
    """The A0 payload: DDL and the question, and deliberately nothing else."""
    return _QUESTION_TURN.format(schema=schema, question=task.question)


def build_retry_turn(query: EmittedQuery) -> str:
    """What the model is told after a failed query.

    It gets the engine's own error text: withholding it would measure blind
    guessing, and no real deployment withholds it.
    """
    return _RETRY_TURN.format(
        error_class=query.error_class, error_text=query.error_text or "no detail"
    )


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

        started = perf_counter()
        schema = await self.executor.schema_text(task.namespace)
        system_prompt = build_system_prompt(self.executor.engine)

        question_turn = build_question_turn(task, schema)
        turns: list[Turn] = [Turn(role="user", content=question_turn)]
        queries: list[EmittedQuery] = []
        notes: list[str] = []
        input_tokens = 0
        output_tokens = 0

        for _ in range(self.max_retries + 1):
            response = await self.client.complete(
                system=system_prompt, turns=tuple(turns), model=model, seed=seed
            )
            input_tokens += response.tokens.input_tokens
            output_tokens += response.tokens.output_tokens

            sql = extract_sql(response.text)
            if sql is None:
                notes.append("the model replied without a SQL query")
                break

            emitted = await self.executor.run(sql)
            queries.append(emitted)
            if emitted.succeeded:
                break

            turns.append(Turn(role="assistant", content=response.text))
            turns.append(Turn(role="user", content=build_retry_turn(emitted)))

        return Attempt(
            system=self.name,
            task_id=task.id,
            seed=seed,
            model=model,
            prompt=f"{system_prompt}\n\n{question_turn}",
            queries=tuple(queries),
            tokens=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
            context_bytes=len(schema.encode("utf-8")),
            wall_clock_ms=round((perf_counter() - started) * 1000),
            notes=tuple(notes),
        )
