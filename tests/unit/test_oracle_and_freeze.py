"""A7_oracle (the harness ceiling) and freezing gold into a committed lock file."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from agenteval.freeze import compute_gold_hashes, write_gold_lock
from agenteval.gold import GoldError
from agenteval.models.base import ModelError
from agenteval.scorer import result_hash
from agenteval.systems.base import SystemUnderTest
from agenteval.systems.loop import NO_SQL_NOTE
from agenteval.systems.oracle import ARM_NAME, OracleSystem, build_oracle_turn
from agenteval.tasks import GOLD_LOCK_NAME, TaskSuite, load_suite
from tests.harness_fakes import MODEL, SYNTAX_ERROR, FakeExecutor, ScriptedModelClient, sample_task

FENCED = "```sql\n{sql}\n```"


def _oracle(*replies: str) -> tuple[OracleSystem, FakeExecutor, ScriptedModelClient]:
    executor = FakeExecutor()
    client = ScriptedModelClient(replies=list(replies))
    return OracleSystem.create(executor=executor, client=client), executor, client


# --------------------------------------------------------------------------
# A7_oracle
# --------------------------------------------------------------------------


def test_the_oracle_is_a_system_under_test() -> None:
    system, _, _ = _oracle()

    assert isinstance(system, SystemUnderTest)
    assert system.name == ARM_NAME
    assert system.config_fingerprint.startswith("sha256:")


def test_the_oracle_turn_carries_the_gold_query() -> None:
    # Arrange — this arm exists to measure the harness, so it is handed the answer
    turn = build_oracle_turn(sample_task(), "CREATE TABLE hits (...)")

    assert "SELECT count() FROM hits" in turn
    assert "known to answer this question correctly" in turn


def test_the_oracle_still_asks_the_question() -> None:
    # An arm that executed gold directly would measure nothing about extraction,
    # comparison, or the engine — which is all this arm is for
    turn = build_oracle_turn(sample_task(), "ddl")

    assert "How many rows are in the hits table?" in turn


def test_a_negative_retry_budget_is_refused() -> None:
    with pytest.raises(ValueError, match="max_retries must be >= 0"):
        OracleSystem.create(executor=FakeExecutor(), client=ScriptedModelClient(), max_retries=-1)


async def test_the_oracle_refuses_to_invent_a_model() -> None:
    system, _, _ = _oracle()

    with pytest.raises(ModelError, match="chooses no model of its own"):
        await system.answer(sample_task(), None, seed=0)


async def test_the_oracle_answers_and_counts_its_own_context() -> None:
    system, executor, _ = _oracle(FENCED.format(sql="SELECT count() FROM hits"))

    attempt = await system.answer(sample_task(), MODEL, seed=0)

    assert attempt.system == ARM_NAME
    assert attempt.queries[0].succeeded is True
    assert executor.executed == ["SELECT count() FROM hits"]
    # the gold query is part of the injected payload, so it is part of the cost
    assert attempt.context_bytes > len(executor.schema.encode("utf-8"))


# --------------------------------------------------------------------------
# the shared loop, exercised through the arm that reuses it
# --------------------------------------------------------------------------


async def test_both_arms_share_one_retry_loop() -> None:
    # Arrange — if arms retried differently, the gap between them would measure
    # the loop instead of the grounding
    executor = FakeExecutor(outcomes=[SYNTAX_ERROR, replace(SYNTAX_ERROR, succeeded=True)])
    client = ScriptedModelClient(replies=[FENCED.format(sql="SELEC 1"), FENCED.format(sql="q")])
    system = OracleSystem.create(executor=executor, client=client)

    attempt = await system.answer(sample_task(), MODEL, seed=0)

    assert len(attempt.queries) == 2
    assert attempt.blind().retries == 1


async def test_a_reply_with_no_sql_is_noted_identically() -> None:
    system, _, _ = _oracle("I would rather not.")

    attempt = await system.answer(sample_task(), MODEL, seed=0)

    assert attempt.queries == ()
    assert attempt.notes == (NO_SQL_NOTE,)


# --------------------------------------------------------------------------
# freezing gold
# --------------------------------------------------------------------------


def _suite() -> TaskSuite:
    return TaskSuite(name="clickbench_nl", tasks=(sample_task("t1"), sample_task("t2")))


async def test_freezing_hashes_every_gold_result() -> None:
    hashes = await compute_gold_hashes(FakeExecutor(), _suite())

    assert set(hashes) == {"t1", "t2"}
    assert hashes["t1"] == result_hash(("count()",), ((99997497,),), ordered=False)


async def test_freezing_a_suite_the_engine_cannot_serve_is_refused() -> None:
    databricks_only = TaskSuite(name="s", tasks=(replace(sample_task(), engines=("databricks",)),))

    with pytest.raises(GoldError, match="no tasks targeting clickhouse"):
        await compute_gold_hashes(FakeExecutor(), databricks_only)


async def test_freezing_re_checks_existing_hashes_rather_than_papering_over_drift() -> None:
    # Arrange — a stale committed hash must fail the freeze, not be overwritten
    drifted = TaskSuite(
        name="s",
        tasks=(replace(sample_task(), gold_result_hashes=(("clickhouse", "sha256:stale"),)),),
    )

    with pytest.raises(GoldError, match="gold drift"):
        await compute_gold_hashes(FakeExecutor(), drifted)


def test_the_lock_file_is_sorted_and_self_documenting(tmp_path: Path) -> None:
    path = write_gold_lock(
        tmp_path, "clickbench_nl", {"t2": "sha256:b", "t1": "sha256:a"}, engine="clickhouse"
    )

    text = path.read_text(encoding="utf-8")
    assert path.name == GOLD_LOCK_NAME
    assert "freeze-gold --suite clickbench_nl" in text
    assert text.index("t1:") < text.index("t2:")
    assert yaml.safe_load(text) == {
        "t1": {"clickhouse": "sha256:a"},
        "t2": {"clickhouse": "sha256:b"},
    }


def test_freezing_one_engine_does_not_erase_the_other(tmp_path: Path) -> None:
    # a cross-engine suite is frozen twice, and the second freeze must not
    # silently disarm drift detection on the first engine
    write_gold_lock(tmp_path, "tpch_nl", {"t1": "sha256:ch"}, engine="clickhouse")
    path = write_gold_lock(tmp_path, "tpch_nl", {"t1": "sha256:dbx"}, engine="databricks")

    assert yaml.safe_load(path.read_text(encoding="utf-8")) == {
        "t1": {"clickhouse": "sha256:ch", "databricks": "sha256:dbx"}
    }


def test_a_flat_lock_file_is_read_as_covering_every_engine(tmp_path: Path) -> None:
    # the single-engine form authors wrote by hand stays valid
    (tmp_path / GOLD_LOCK_NAME).write_text("t1: sha256:flat\n", encoding="utf-8")

    path = write_gold_lock(tmp_path, "s", {"t1": "sha256:dbx"}, engine="databricks")

    assert yaml.safe_load(path.read_text(encoding="utf-8")) == {
        "t1": {"clickhouse": "sha256:flat", "databricks": "sha256:dbx"}
    }


def test_a_lock_file_that_is_not_a_mapping_cannot_be_merged_into(tmp_path: Path) -> None:
    (tmp_path / GOLD_LOCK_NAME).write_text("- just\n- a list\n", encoding="utf-8")

    with pytest.raises(GoldError, match="mapping of task id"):
        write_gold_lock(tmp_path, "s", {"t1": "sha256:x"}, engine="databricks")


# --------------------------------------------------------------------------
# the lock file feeding back into task loading
# --------------------------------------------------------------------------


def _write_task(directory: Path, task_id: str, extra: str = "") -> None:
    (directory / f"{task_id}.yaml").write_text(
        f"id: {task_id}\n"
        f"suite: s\n"
        f"engine: [clickhouse]\n"
        f'question: "how many?"\n'
        f"gold_sql: SELECT count() FROM hits\n" + extra,
        encoding="utf-8",
    )


def test_a_locked_hash_is_attached_to_its_task(tmp_path: Path) -> None:
    _write_task(tmp_path, "t1")
    write_gold_lock(tmp_path, "s", {"t1": "sha256:frozen"}, engine="clickhouse")

    task = load_suite(tmp_path).by_id("t1")
    assert task.gold_hash_for("clickhouse") == "sha256:frozen"
    # frozen on one engine only: the other has nothing to check against
    assert task.gold_hash_for("databricks") is None


def test_a_suite_with_no_lock_file_still_loads(tmp_path: Path) -> None:
    _write_task(tmp_path, "t1")

    assert load_suite(tmp_path).by_id("t1").gold_result_hashes == ()


def test_a_lock_naming_an_unknown_task_is_refused(tmp_path: Path) -> None:
    _write_task(tmp_path, "t1")
    write_gold_lock(tmp_path, "s", {"t9": "sha256:x"}, engine="clickhouse")

    with pytest.raises(Exception, match="names task\\(s\\) that do not exist: t9"):
        load_suite(tmp_path)


def test_a_hash_in_two_places_is_refused(tmp_path: Path) -> None:
    # One source of truth, or a reviewer cannot tell which one was verified
    _write_task(tmp_path, "t1", extra='gold_result_hash: "sha256:inline"\n')
    write_gold_lock(tmp_path, "s", {"t1": "sha256:frozen"}, engine="clickhouse")

    with pytest.raises(Exception, match="one source of truth"):
        load_suite(tmp_path)


def test_an_empty_lock_file_is_harmless(tmp_path: Path) -> None:
    _write_task(tmp_path, "t1")
    (tmp_path / GOLD_LOCK_NAME).write_text("", encoding="utf-8")

    assert load_suite(tmp_path).by_id("t1").gold_result_hashes == ()


def test_a_lock_file_that_is_not_a_mapping_is_refused(tmp_path: Path) -> None:
    _write_task(tmp_path, "t1")
    (tmp_path / GOLD_LOCK_NAME).write_text("- just a list\n", encoding="utf-8")

    with pytest.raises(Exception, match="must be a mapping"):
        load_suite(tmp_path)
