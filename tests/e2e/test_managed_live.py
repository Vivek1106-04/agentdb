"""Live checks against the managed services (SPEC §11.5.1, §11.5.2).

Everything the Genie and ClickHouse Agents clients know about their vendors'
reply shapes was read from documentation, and the unit tier proves only that the
readers do what this project *believes* those shapes are. These tests are where
that becomes an observation: one real question to a real space or agent, and the
answer parsed by the same code a benchmark run uses.

They **skip loudly**, naming the variable that was missing. A managed arm that
silently never ran would leave `S3`/`S4a`/`S4b` at zero cells in the report while
CI stayed green, which is exactly the failure §11.5 exists to prevent.

Run with::

    uv sync --extra databricks
    export AGENTEVAL_DBX_HOST=... AGENTEVAL_DBX_WAREHOUSE_ID=... AGENTEVAL_DBX_TOKEN=...
    export AGENTEVAL_GENIE_SPACE_ID=...
    uv run pytest -m e2e tests/e2e/test_managed_live.py

**Read the beta terms before running either of these** (§11.5.1). If they
prohibit publishing benchmark results, stop and ask the vendor rather than
publishing anyway. Nothing here writes: a conversation turn is a read, and the
SQL that comes back is executed by the harness under its own read-only role.
"""

from __future__ import annotations

import os

import pytest

from agenteval.engines.clickhouse import ClickHouseExecutor
from agenteval.engines.connect import ClickHouseTarget, DatabricksTarget, build_client
from agenteval.suites import load_builtin
from agenteval.systems.clickhouse_agents import ClickHouseAgentsTarget, build_conversation
from agenteval.systems.genie import build_genie_conversation
from agenteval.systems.managed import ManagedConfig, ManagedSystem

pytestmark = pytest.mark.e2e

GENIE_REQUIRED = ("AGENTEVAL_DBX_HOST", "AGENTEVAL_DBX_WAREHOUSE_ID", "AGENTEVAL_GENIE_SPACE_ID")
AGENTS_REQUIRED = ("AGENTEVAL_CH_AGENTS_HOST", "AGENTEVAL_CH_AGENTS_AGENT_ID")

QUESTION = "How many rows are in the region table?"
"""Deliberately not a suite question. A live probe that asked one would put a
graded task through a space before the run that measures it."""


def _require(names: tuple[str, ...]) -> None:
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        pytest.skip(
            f"managed-service e2e skipped: {', '.join(missing)} unset. "
            "Export them once the space or agent exists, and read the vendor's beta "
            "terms before measuring anything (SPEC §11.5.1)."
        )


# --------------------------------------------------------------------------
# Databricks AI/BI Genie
# --------------------------------------------------------------------------


async def test_a_genie_space_answers_and_the_reply_parses() -> None:
    """The shape the client reads is the shape the SDK actually returns."""
    _require(GENIE_REQUIRED)
    conversation = build_genie_conversation(DatabricksTarget.from_env())

    answer = await conversation.ask(os.environ["AGENTEVAL_GENIE_SPACE_ID"], QUESTION)

    assert any(note.startswith("genie message status:") for note in answer.notes)
    assert answer.sql or answer.text, (
        "Genie returned neither SQL nor prose, which means the attachment shape "
        "this client reads has changed"
    )


async def test_the_committed_genie_configurations_name_a_reachable_space() -> None:
    """A space id in `eval/managed.yaml` that answers nothing is an unmeasurable row."""
    _require(GENIE_REQUIRED)
    conversation = build_genie_conversation(DatabricksTarget.from_env())

    answer = await conversation.ask(os.environ["AGENTEVAL_GENIE_SPACE_ID"], QUESTION)

    if answer.sql is None:
        pytest.skip(f"the space declined the probe: {answer.text!r}")
    assert "region" in answer.sql.lower()


# --------------------------------------------------------------------------
# ClickHouse Agents
# --------------------------------------------------------------------------


async def test_a_clickhouse_agent_answers_where_the_committed_config_says() -> None:
    """The configured request path and response paths match the live beta.

    A failure here is the useful kind: it says the beta's payload moved, and the
    fix is an `eval/managed.yaml` diff that a reader can see in the run that used
    it — not a silent column of zeros.
    """
    _require(AGENTS_REQUIRED)
    conversation = build_conversation({}, target=ClickHouseAgentsTarget.from_env())

    answer = await conversation.ask(os.environ["AGENTEVAL_CH_AGENTS_AGENT_ID"], QUESTION)

    assert answer.sql or answer.text, (
        "the agent returned neither SQL nor prose at the configured paths; "
        "update the arm's `response` mapping in eval/managed.yaml"
    )


# --------------------------------------------------------------------------
# the whole arm, end to end
# --------------------------------------------------------------------------


async def test_a_managed_arm_scores_a_real_task_through_the_harness() -> None:
    """One graded cell, exactly as a run produces it.

    The service writes SQL, the *harness* executes it under its own read-only
    role, and the attempt carries a query the grader could score. That last step
    is the one that makes a Family S row comparable to every other row.
    """
    _require(AGENTS_REQUIRED)
    try:
        client = await build_client(ClickHouseTarget.from_env())
    except Exception as exc:
        pytest.skip(f"no ClickHouse reachable ({exc}); start one with: make up")

    executor = ClickHouseExecutor(client=client, context_id="e2e")
    task = load_builtin("clickbench_nl").subset(1).tasks[0]
    system = ManagedSystem.create(
        config=ManagedConfig(
            name="S3_clickhouse_agents",
            kind="clickhouse_agents",
            version="beta-2026-05",
            target_id=os.environ["AGENTEVAL_CH_AGENTS_AGENT_ID"],
        ),
        conversation=build_conversation({}, target=ClickHouseAgentsTarget.from_env()),
        executor=executor,
        tasks=[task],
    )

    try:
        attempt = await system.answer(task, None, seed=0)
    finally:
        await executor.aclose()

    assert len(attempt.queries) == 1
    assert attempt.model is None, "a managed service picks its own model"
    assert attempt.tokens.total == 0, "a managed service reports no tokens of its own"
