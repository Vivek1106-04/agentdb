"""``S5_claude_code`` is a Family S row, and the tests hold it to that.

The arm sends the same payload as A0 — same prompt, same schema dump, same retry
loop — because a comparison across arms means nothing if the question differs.
What it must never do is pretend to be A0: the channel carries the product's own
context, so the channel is in the name, in the fingerprint, and enforced at
construction.
"""

from __future__ import annotations

import pytest

from agenteval.models.base import ModelError
from agenteval.models.claude_cli import PROVIDER as CLI_PROVIDER
from agenteval.systems.base import ModelSpec
from agenteval.systems.claude_code import ARM_NAME, SCAFFOLDING_NOTE, ClaudeCodeSystem
from agenteval.systems.raw_schema import ARM_NAME as BASELINE_ARM
from agenteval.systems.raw_schema import RawSchemaSystem
from tests.harness_fakes import FakeExecutor, ScriptedModelClient, sample_task

MODEL = ModelSpec(provider=CLI_PROVIDER, name="sonnet")


def _cli_model(*replies: str) -> ScriptedModelClient:
    """A stub answering on the subscription channel rather than the API."""
    return ScriptedModelClient(replies=list(replies), provider=CLI_PROVIDER)


def _system(**overrides: object) -> ClaudeCodeSystem:
    return ClaudeCodeSystem.create(
        executor=FakeExecutor(),
        client=_cli_model("```sql\nSELECT count() FROM hits\n```"),
        **overrides,  # type: ignore[arg-type]
    )


def test_the_arm_is_named_after_the_product_not_the_ablation_floor() -> None:
    system = _system()

    # a shipped agent product measured through its own CLI is Family S; putting
    # this number in A0 would place a product in the bare-model floor
    assert system.name == ARM_NAME
    assert system.name != BASELINE_ARM
    assert system.controls_model is True


def test_the_arm_refuses_a_channel_that_is_not_the_product_it_names() -> None:
    with pytest.raises(ModelError, match="needs the 'claude-cli' channel"):
        ClaudeCodeSystem.create(executor=FakeExecutor(), client=ScriptedModelClient(replies=[""]))


def test_a_negative_retry_budget_is_refused() -> None:
    with pytest.raises(ValueError, match="max_retries must be >= 0"):
        _system(max_retries=-1)


async def test_the_arm_answers_with_the_same_payload_a0_would_send() -> None:
    system = _system()

    attempt = await system.answer(sample_task(), MODEL, seed=0)

    assert attempt.system == ARM_NAME
    assert attempt.queries
    assert attempt.queries[-1].sql.startswith("SELECT count()")


async def test_the_arm_needs_a_model_named_by_the_run() -> None:
    with pytest.raises(ModelError, match="chooses no model of its own"):
        await _system().answer(sample_task(), None, seed=0)


def test_the_fingerprint_covers_the_instruction_files_that_shaped_the_answer() -> None:
    # the operator's ~/.claude content reaches the model on every call, so two
    # machines are not running the same configuration and must not claim to
    system = _system()
    other = ClaudeCodeSystem.create(executor=FakeExecutor(), client=_cli_model(""), max_retries=0)

    assert system.config_fingerprint != other.config_fingerprint
    assert system.config_fingerprint


def test_the_arm_does_not_share_a_fingerprint_with_the_baseline() -> None:
    a0 = RawSchemaSystem.create(executor=FakeExecutor(), client=ScriptedModelClient(replies=[""]))

    assert _system().config_fingerprint != a0.config_fingerprint


async def test_a_reply_with_no_sql_is_recorded_rather_than_crashing() -> None:
    system = ClaudeCodeSystem.create(
        executor=FakeExecutor(), client=_cli_model("I cannot answer that.")
    )

    attempt = await system.answer(sample_task(), MODEL, seed=0)

    assert attempt.queries == ()
    assert attempt.notes


def test_the_stub_channel_matches_the_real_one() -> None:
    assert _cli_model("").provider == "claude-cli"


async def test_every_attempt_records_the_context_the_product_added() -> None:
    # the arm's own context is the schema dump; this is everything the product
    # sent on top, and it is far larger
    client = _cli_model("```sql\nSELECT count() FROM hits\n```")
    client.scaffolding_tokens = 18_830  # type: ignore[attr-defined]
    system = ClaudeCodeSystem.create(executor=FakeExecutor(), client=client)

    attempt = await system.answer(sample_task(), MODEL, seed=0)

    assert f"{SCAFFOLDING_NOTE}=18830" in attempt.notes


async def test_a_channel_that_reports_no_scaffolding_records_zero_not_silence() -> None:
    attempt = await _system().answer(sample_task(), MODEL, seed=0)

    assert f"{SCAFFOLDING_NOTE}=0" in attempt.notes
