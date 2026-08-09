"""The self-correction loop shared by every model-driven arm.

Arms differ in *what context they inject*, and in nothing else. Keeping the loop
in one place is what makes that true: if A0 and A7 retried differently, or
counted tokens differently, the gap between them would measure the loop instead
of the grounding, and the whole ablation would be uninterpretable.
"""

from __future__ import annotations

from time import perf_counter

from agenteval.execution import QueryExecutor
from agenteval.models.base import ModelClient, Turn
from agenteval.models.extract import extract_sql
from agenteval.systems.base import Attempt, EmittedQuery, ModelSpec, TokenUsage
from agenteval.tasks import Task

NO_SQL_NOTE = "the model replied without a SQL query"

_RETRY_TURN = """That query failed.

error_class: {error_class}
error: {error_text}

Reply with one corrected SQL query."""


def build_retry_turn(query: EmittedQuery) -> str:
    """What a model is told after a failed query.

    It gets the engine's own error text: withholding it would measure blind
    guessing, and no real deployment withholds it.
    """
    return _RETRY_TURN.format(
        error_class=query.error_class, error_text=query.error_text or "no detail"
    )


async def answer_with_model(
    *,
    system_name: str,
    task: Task,
    model: ModelSpec,
    seed: int,
    executor: QueryExecutor,
    client: ModelClient,
    system_prompt: str,
    first_turn: str,
    context_bytes: int,
    max_retries: int,
) -> Attempt:
    """Ask, execute, and self-correct up to ``max_retries`` times."""
    started = perf_counter()
    turns: list[Turn] = [Turn(role="user", content=first_turn)]
    queries: list[EmittedQuery] = []
    notes: list[str] = []
    input_tokens = 0
    output_tokens = 0

    for _ in range(max_retries + 1):
        response = await client.complete(
            system=system_prompt, turns=tuple(turns), model=model, seed=seed
        )
        input_tokens += response.tokens.input_tokens
        output_tokens += response.tokens.output_tokens

        sql = extract_sql(response.text)
        if sql is None:
            notes.append(NO_SQL_NOTE)
            break

        emitted = await executor.run(sql)
        queries.append(emitted)
        if emitted.succeeded:
            break

        turns.append(Turn(role="assistant", content=response.text))
        turns.append(Turn(role="user", content=build_retry_turn(emitted)))

    return Attempt(
        system=system_name,
        task_id=task.id,
        seed=seed,
        model=model,
        prompt=f"{system_prompt}\n\n{first_turn}",
        queries=tuple(queries),
        tokens=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        context_bytes=context_bytes,
        wall_clock_ms=round((perf_counter() - started) * 1000),
        notes=tuple(notes),
    )
