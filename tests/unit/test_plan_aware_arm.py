"""``A3_plan``: the review step, and the fairness properties around it.

The arm is only interpretable if it differs from A2 in exactly one way. The tests
that matter most here are therefore the accounting ones — that a review costs a
model call and not an attempt at the engine, and that a plan with nothing to say
costs nothing at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from agenteval.models.base import ModelError
from agenteval.systems.base import SystemUnderTest
from agenteval.systems.loop import REVIEWED_NOTE
from agenteval.systems.plan_aware import PlanAdvisor, PlanAwareSystem, build_review_turn
from tests.harness_fakes import (
    MODEL,
    SYNTAX_ERROR,
    FakeExecutor,
    ScriptedModelClient,
    sample_task,
)

DRAFT = "```sql\nSELECT count() FROM hits WHERE SearchEngineID = 2\n```"
FINAL = "```sql\nSELECT count() FROM hits WHERE CounterID = 1 AND SearchEngineID = 2\n```"

PLAN_TEXT = "Plan summary (clickhouse):\n- [warning] SORT_KEY_UNUSED: nothing prunes"


@dataclass
class FakeProvider:
    payload: str = "CREATE TABLE hits (...)"
    name: str = "agentdb/A3_plan"
    version: str = "1.0"
    fingerprint: str = "sha256:fake"

    async def context(self, *, namespace: str, question: str) -> str:
        return self.payload


@dataclass
class FakeAdvisor:
    """Returns canned plan feedback and records the drafts it was shown."""

    feedback: str | None = PLAN_TEXT
    reviewed: list[tuple[str, str]] = field(default_factory=list)

    async def explain_plan(self, *, sql: str, namespace: str) -> str | None:
        self.reviewed.append((sql, namespace))
        return self.feedback


def _system(
    *replies: str,
    advisor: FakeAdvisor | None = None,
    executor: FakeExecutor | None = None,
    max_retries: int = 2,
) -> tuple[PlanAwareSystem, FakeAdvisor, ScriptedModelClient]:
    resolved = advisor or FakeAdvisor()
    client = ScriptedModelClient(replies=list(replies))
    system = PlanAwareSystem.create(
        arm="A3_plan",
        provider=FakeProvider(),
        advisor=resolved,
        executor=executor or FakeExecutor(),
        client=client,
        max_retries=max_retries,
    )
    return system, resolved, client


async def test_the_draft_is_reviewed_before_anything_reaches_the_engine() -> None:
    system, advisor, _ = _system(DRAFT, FINAL)
    executor = system.executor
    assert isinstance(executor, FakeExecutor)

    attempt = await system.answer(sample_task(), MODEL, seed=0)

    assert advisor.reviewed == [("SELECT count() FROM hits WHERE SearchEngineID = 2", "agentdb")]
    assert executor.executed == [
        "SELECT count() FROM hits WHERE CounterID = 1 AND SearchEngineID = 2"
    ]
    assert REVIEWED_NOTE in attempt.notes


async def test_the_model_is_shown_the_plan_it_earned() -> None:
    system, _, client = _system(DRAFT, FINAL)

    await system.answer(sample_task(), MODEL, seed=0)

    (_, turns, _, _) = client.calls[1]
    assert turns[-1].content == build_review_turn(PLAN_TEXT)
    assert PLAN_TEXT in turns[-1].content


async def test_a_plan_with_nothing_to_say_costs_no_extra_turn() -> None:
    system, _, client = _system(DRAFT, advisor=FakeAdvisor(feedback=None))

    attempt = await system.answer(sample_task(), MODEL, seed=0)

    assert len(client.calls) == 1
    assert attempt.notes == ()
    assert attempt.queries[0].succeeded


async def test_the_review_does_not_spend_an_attempt_at_the_engine() -> None:
    executor = FakeExecutor(outcomes=[SYNTAX_ERROR, SYNTAX_ERROR, SYNTAX_ERROR])
    system, _, _ = _system(DRAFT, FINAL, FINAL, FINAL, executor=executor, max_retries=2)

    attempt = await system.answer(sample_task(), MODEL, seed=0)

    assert len(attempt.queries) == 3  # one first attempt plus two retries, as in every arm
    assert all(not query.succeeded for query in attempt.queries)


async def test_a_reviewed_arm_still_self_corrects_on_an_engine_error() -> None:
    executor = FakeExecutor(outcomes=[SYNTAX_ERROR])
    system, _, _ = _system(DRAFT, FINAL, FINAL, executor=executor)

    attempt = await system.answer(sample_task(), MODEL, seed=0)

    assert len(attempt.queries) == 2
    assert attempt.queries[-1].succeeded


async def test_a_model_that_replies_without_sql_is_recorded_not_reviewed() -> None:
    system, advisor, _ = _system("I would need more information.")

    attempt = await system.answer(sample_task(), MODEL, seed=0)

    assert advisor.reviewed == []
    assert attempt.queries == ()


def test_the_arm_is_a_system_under_test_and_names_its_review_in_the_fingerprint() -> None:
    reviewed, _, _ = _system()
    other, _, _ = _system(advisor=FakeAdvisor(feedback="different"))

    assert isinstance(reviewed, SystemUnderTest)
    assert reviewed.name == "A3_plan"
    assert reviewed.config_fingerprint == other.config_fingerprint  # the advisor is not the config
    assert isinstance(reviewed.advisor, PlanAdvisor)


def test_a_negative_retry_budget_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="max_retries"):
        _system(max_retries=-1)


async def test_an_arm_that_controls_its_model_refuses_to_run_without_one() -> None:
    system, _, _ = _system()

    with pytest.raises(ModelError):
        await system.answer(sample_task(), None, seed=0)
