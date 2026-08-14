"""A0_baseline is the floor every other arm is read against, so it is pinned hard."""

from __future__ import annotations

import pytest

from agenteval.execution import QueryExecutor
from agenteval.models.base import ModelError
from agenteval.systems.base import EmittedQuery, SystemUnderTest
from agenteval.systems.loop import build_retry_turn
from agenteval.systems.raw_schema import (
    ARM_NAME,
    RawSchemaSystem,
    build_question_turn,
    build_system_prompt,
)
from tests.harness_fakes import (
    HITS_DDL,
    MODEL,
    OK,
    SYNTAX_ERROR,
    FakeExecutor,
    ScriptedModelClient,
    sample_task,
)

FENCED = "```sql\n{sql}\n```"


def _system(
    *replies: str, outcomes: list[EmittedQuery] | None = None, max_retries: int = 2
) -> tuple[RawSchemaSystem, FakeExecutor, ScriptedModelClient]:
    executor = FakeExecutor(outcomes=list(outcomes or []))
    client = ScriptedModelClient(replies=list(replies))
    system = RawSchemaSystem.create(executor=executor, client=client, max_retries=max_retries)
    return system, executor, client


def test_the_system_prompt_names_the_engine() -> None:
    assert "ClickHouse" in build_system_prompt("clickhouse")
    assert "Databricks SQL" in build_system_prompt("databricks")


def test_the_question_turn_carries_only_ddl_and_the_question() -> None:
    # Arrange — A0 is defined by what it withholds
    turn = build_question_turn(sample_task(), HITS_DDL)

    assert HITS_DDL in turn
    assert "How many rows are in the hits table?" in turn
    assert "cardinality" not in turn.lower()


def test_the_retry_turn_hands_back_the_engine_error() -> None:
    turn = build_retry_turn(SYNTAX_ERROR)

    assert "syntax" in turn
    assert "failed at position 8" in turn


def test_the_retry_turn_survives_an_error_with_no_detail() -> None:
    assert "no detail" in build_retry_turn(OK)


def test_it_is_a_system_under_test() -> None:
    system, _, _ = _system()

    assert isinstance(system, SystemUnderTest)
    assert system.name == ARM_NAME
    assert system.controls_model is True


def test_the_executor_double_satisfies_the_engine_protocol() -> None:
    assert isinstance(FakeExecutor(), QueryExecutor)


def test_the_fingerprint_tracks_the_effective_config() -> None:
    baseline, _, _ = _system()
    retried, _, _ = _system(max_retries=5)

    assert baseline.config_fingerprint.startswith("sha256:")
    assert baseline.config_fingerprint != retried.config_fingerprint


def test_negative_retry_budgets_are_rejected() -> None:
    with pytest.raises(ValueError, match="max_retries must be >= 0"):
        RawSchemaSystem.create(
            executor=FakeExecutor(), client=ScriptedModelClient(), max_retries=-1
        )


async def test_a_first_try_success_emits_one_query() -> None:
    # Arrange
    system, executor, client = _system(FENCED.format(sql="SELECT count() FROM hits"))

    # Act
    attempt = await system.answer(sample_task(), MODEL, seed=7)

    # Assert
    assert len(attempt.queries) == 1
    assert attempt.queries[0].sql == "SELECT count() FROM hits"
    assert attempt.queries[0].succeeded is True
    assert executor.executed == ["SELECT count() FROM hits"]
    assert len(client.calls) == 1


async def test_the_attempt_carries_the_run_identity_and_costs() -> None:
    system, _, _ = _system(FENCED.format(sql="SELECT 1"))

    attempt = await system.answer(sample_task("nl_042"), MODEL, seed=7)

    assert attempt.system == ARM_NAME
    assert attempt.task_id == "nl_042"
    assert attempt.seed == 7
    assert attempt.model == MODEL
    assert attempt.tokens.input_tokens == 100
    assert attempt.tokens.output_tokens == 20
    assert attempt.context_bytes == len(HITS_DDL.encode("utf-8"))
    assert attempt.wall_clock_ms is not None


async def test_the_schema_is_fetched_for_the_task_namespace() -> None:
    system, executor, _ = _system(FENCED.format(sql="SELECT 1"))

    await system.answer(sample_task(), MODEL, seed=0)

    assert executor.namespaces == ["agentdb"]


async def test_a_failed_query_is_retried_with_the_error_in_context() -> None:
    # Arrange — first query fails, second succeeds
    system, executor, client = _system(
        FENCED.format(sql="SELEC 1"),
        FENCED.format(sql="SELECT 1"),
        outcomes=[SYNTAX_ERROR, OK],
    )

    # Act
    attempt = await system.answer(sample_task(), MODEL, seed=0)

    # Assert — both queries are kept, so EX@1 and EX@k are both measurable
    assert [query.sql for query in attempt.queries] == ["SELEC 1", "SELECT 1"]
    assert attempt.blind().retries == 1
    assert executor.executed == ["SELEC 1", "SELECT 1"]

    second_call_turns = client.calls[1][1]
    assert [turn.role for turn in second_call_turns] == ["user", "assistant", "user"]
    assert "failed at position 8" in second_call_turns[-1].content


async def test_tokens_accumulate_across_retries() -> None:
    system, _, _ = _system(
        FENCED.format(sql="SELEC 1"),
        FENCED.format(sql="SELECT 1"),
        outcomes=[SYNTAX_ERROR, OK],
    )

    attempt = await system.answer(sample_task(), MODEL, seed=0)

    assert attempt.tokens.input_tokens == 200
    assert attempt.tokens.output_tokens == 40


async def test_the_retry_budget_is_honoured() -> None:
    system, executor, _ = _system(
        FENCED.format(sql="SELEC 1"),
        FENCED.format(sql="SELEC 2"),
        outcomes=[SYNTAX_ERROR, SYNTAX_ERROR],
        max_retries=1,
    )

    attempt = await system.answer(sample_task(), MODEL, seed=0)

    assert len(executor.executed) == 2
    assert attempt.queries[-1].succeeded is False


async def test_a_reply_with_no_sql_stops_the_loop_and_is_noted() -> None:
    # Arrange — scoring this as an execution error would misattribute the failure
    system, executor, _ = _system("I cannot answer that from this schema.")

    attempt = await system.answer(sample_task(), MODEL, seed=0)

    assert attempt.queries == ()
    assert attempt.notes == ("the model replied without a SQL query",)
    assert executor.executed == []


async def test_the_arm_refuses_to_invent_a_model() -> None:
    system, _, _ = _system()

    with pytest.raises(ModelError, match="chooses no model of its own"):
        await system.answer(sample_task(), None, seed=0)
