"""Task loading is strict on purpose: a benchmark cannot silently skip tasks."""

from __future__ import annotations

from pathlib import Path

import pytest

from agenteval.tasks import (
    GOLD_LOCK_NAME,
    Task,
    TaskLoadError,
    TaskSuite,
    gold_sql_fingerprint,
    load_suite,
    parse_task,
)

VALID = {
    "id": "clickbench_nl_017",
    "suite": "clickbench_nl",
    "engine": ["clickhouse"],
    "question": "How many unique visitors used each search engine?",
    "gold_sql": "SELECT SearchEngineID, uniq(UserID) FROM hits GROUP BY SearchEngineID",
}


def test_parses_a_minimal_task_with_documented_defaults() -> None:
    task = parse_task(VALID)

    assert task.id == "clickbench_nl_017"
    assert task.engines == ("clickhouse",)
    assert task.difficulty == "medium"
    assert task.namespace == "agentdb"
    assert task.tags == ()
    assert task.gold_result_hashes == ()


def test_parses_the_full_task_shape_from_the_spec() -> None:
    task = parse_task(
        {
            **VALID,
            "difficulty": "hard",
            "tags": ["group_by", "high_cardinality"],
            "gold_result_hash": "sha256:abc",
            "notes": "MobilePhone is 0/non-zero, not boolean",
            "namespace": "bench",
        },
        source_path="suites/clickbench_nl/017.yaml",
    )

    assert task.tags == ("group_by", "high_cardinality")
    assert task.notes is not None
    assert task.source_path == "suites/clickbench_nl/017.yaml"
    assert task.targets("clickhouse") is True
    assert task.targets("databricks") is False


def test_a_scalar_engine_is_accepted_and_deduplicated() -> None:
    assert parse_task({**VALID, "engine": "databricks"}).engines == ("databricks",)
    assert parse_task({**VALID, "engine": ["databricks", "databricks", "clickhouse"]}).engines == (
        "databricks",
        "clickhouse",
    )


@pytest.mark.parametrize("missing", ["id", "suite", "engine", "question", "gold_sql"])
def test_missing_required_fields_fail_loudly(missing: str) -> None:
    payload = {k: v for k, v in VALID.items() if k != missing}
    with pytest.raises(TaskLoadError, match=f"missing required field\\(s\\): {missing}"):
        parse_task(payload)


def test_unknown_fields_are_an_error_not_a_shrug() -> None:
    with pytest.raises(TaskLoadError, match="unknown field\\(s\\): gold_hash"):
        parse_task({**VALID, "gold_hash": "sha256:abc"})


def test_unknown_engine_is_rejected() -> None:
    with pytest.raises(TaskLoadError, match="unknown engine 'duckdb'"):
        parse_task({**VALID, "engine": "duckdb"})


def test_empty_engine_list_is_rejected() -> None:
    with pytest.raises(TaskLoadError, match="declares no engine"):
        parse_task({**VALID, "engine": []})


def test_unknown_difficulty_is_rejected() -> None:
    with pytest.raises(TaskLoadError, match="unknown difficulty 'trivial'"):
        parse_task({**VALID, "difficulty": "trivial"})


@pytest.mark.parametrize(
    ("field_name", "message"), [("gold_sql", "gold_sql"), ("question", "question")]
)
def test_blank_content_is_rejected(field_name: str, message: str) -> None:
    with pytest.raises(TaskLoadError, match=f"empty {message}"):
        parse_task({**VALID, field_name: "   "})


def test_blank_optional_strings_become_none() -> None:
    task = parse_task({**VALID, "notes": "  ", "gold_result_hash": None})
    assert task.notes is None
    assert task.gold_result_hashes == ()


def _write(directory: Path, name: str, body: str) -> None:
    (directory / name).write_text(body, encoding="utf-8")


def _task_yaml(task_id: str, suite: str = "clickbench_nl", engine: str = "clickhouse") -> str:
    return (
        f"id: {task_id}\n"
        f"suite: {suite}\n"
        f"engine: [{engine}]\n"
        f'question: "how many?"\n'
        f"gold_sql: SELECT count() FROM hits\n"
    )


def test_loads_and_sorts_a_directory_of_tasks(tmp_path: Path) -> None:
    # Arrange — filesystem order deliberately differs from id order
    _write(tmp_path, "b.yaml", _task_yaml("nl_002"))
    _write(tmp_path, "a.yaml", _task_yaml("nl_001"))

    # Act
    suite = load_suite(tmp_path)

    # Assert — task order is a property of the data, so seeds mean the same thing everywhere
    assert suite.name == "clickbench_nl"
    assert [task.id for task in suite] == ["nl_001", "nl_002"]
    assert len(suite) == 2


def test_a_file_may_hold_a_list_of_tasks(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "all.yaml",
        "- id: nl_001\n"
        "  suite: clickbench_nl\n"
        "  engine: [clickhouse]\n"
        '  question: "how many?"\n'
        "  gold_sql: SELECT count() FROM hits\n"
        "- id: nl_002\n"
        "  suite: clickbench_nl\n"
        "  engine: [clickhouse]\n"
        '  question: "how many again?"\n'
        "  gold_sql: SELECT count() FROM hits\n",
    )

    assert len(load_suite(tmp_path)) == 2


def test_missing_directory_is_reported(tmp_path: Path) -> None:
    with pytest.raises(TaskLoadError, match="task directory does not exist"):
        load_suite(tmp_path / "nope")


def test_empty_directory_is_reported(tmp_path: Path) -> None:
    with pytest.raises(TaskLoadError, match="no tasks found"):
        load_suite(tmp_path)


def test_empty_file_is_reported(tmp_path: Path) -> None:
    _write(tmp_path, "a.yaml", "")
    with pytest.raises(TaskLoadError, match="is empty"):
        load_suite(tmp_path)


def test_malformed_yaml_is_reported(tmp_path: Path) -> None:
    _write(tmp_path, "a.yaml", "id: [unclosed\n")
    with pytest.raises(TaskLoadError, match="not valid YAML"):
        load_suite(tmp_path)


def test_non_mapping_document_is_reported(tmp_path: Path) -> None:
    _write(tmp_path, "a.yaml", "- just a string\n")
    with pytest.raises(TaskLoadError, match="contains a str, expected a mapping"):
        load_suite(tmp_path)


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    _write(tmp_path, "a.yaml", _task_yaml("nl_001"))
    _write(tmp_path, "b.yaml", _task_yaml("nl_001"))
    with pytest.raises(TaskLoadError, match="duplicate task id 'nl_001'"):
        load_suite(tmp_path)


def test_one_directory_holds_one_suite(tmp_path: Path) -> None:
    _write(tmp_path, "a.yaml", _task_yaml("nl_001"))
    _write(tmp_path, "b.yaml", _task_yaml("nl_002", suite="tpch_nl"))
    with pytest.raises(TaskLoadError, match="mixes suites"):
        load_suite(tmp_path)


def _suite() -> TaskSuite:
    return TaskSuite(
        name="tpch_nl",
        tasks=(
            Task(
                id="t1",
                suite="tpch_nl",
                engines=("clickhouse",),
                question="q",
                gold_sql="SELECT 1",
            ),
            Task(
                id="t2",
                suite="tpch_nl",
                engines=("databricks", "clickhouse"),
                question="q",
                gold_sql="SELECT 1",
            ),
        ),
    )


def test_suite_filters_by_engine() -> None:
    assert [task.id for task in _suite().for_engine("databricks")] == ["t2"]
    assert len(_suite().for_engine("clickhouse")) == 2


def test_suite_subset_backs_the_five_minute_reproduction() -> None:
    assert [task.id for task in _suite().subset(1)] == ["t1"]


def test_suite_subset_must_be_positive() -> None:
    with pytest.raises(ValueError, match="subset size must be > 0"):
        _suite().subset(0)


def test_suite_lookup_by_id() -> None:
    assert _suite().by_id("t2").engines == ("databricks", "clickhouse")


def test_suite_lookup_reports_a_missing_id() -> None:
    with pytest.raises(KeyError, match="no task 't9'"):
        _suite().by_id("t9")


# --------------------------------------------------------------------------
# a committed hash belongs to the question it was taken against
# --------------------------------------------------------------------------


def _locked_suite(tmp_path: Path, gold_sql: str, lock: str) -> Task:
    (tmp_path / "s.yaml").write_text(
        f"- id: t1\n  suite: s\n  engine: [clickhouse]\n  question: q\n  gold_sql: {gold_sql}\n",
        encoding="utf-8",
    )
    (tmp_path / GOLD_LOCK_NAME).write_text(lock, encoding="utf-8")
    return load_suite(tmp_path).by_id("t1")


def test_a_hash_frozen_against_the_same_sql_is_attached(tmp_path: Path) -> None:
    fingerprint = gold_sql_fingerprint("SELECT 1")
    task = _locked_suite(
        tmp_path, "SELECT 1", f"t1:\n  clickhouse: sha256:frozen\n  clickhouse_sql: {fingerprint}\n"
    )

    assert task.gold_hash_for("clickhouse") == "sha256:frozen"


def test_a_hash_frozen_against_different_sql_is_dropped(tmp_path: Path) -> None:
    # Editing a question must not be reported as drifted data. Before the
    # fingerprint the only way past that was to hand-delete the lock entry.
    stale = gold_sql_fingerprint("SELECT 2")
    task = _locked_suite(
        tmp_path, "SELECT 1", f"t1:\n  clickhouse: sha256:frozen\n  clickhouse_sql: {stale}\n"
    )

    assert task.gold_result_hashes == ()


def test_a_lock_written_before_fingerprints_still_applies(tmp_path: Path) -> None:
    task = _locked_suite(tmp_path, "SELECT 1", "t1:\n  clickhouse: sha256:frozen\n")

    assert task.gold_hash_for("clickhouse") == "sha256:frozen"


def test_the_fingerprint_ignores_surrounding_whitespace() -> None:
    assert gold_sql_fingerprint("SELECT 1") == gold_sql_fingerprint("  SELECT 1\n")


def test_a_flat_single_engine_lock_entry_carries_no_fingerprint(tmp_path: Path) -> None:
    # The oldest lock form is a bare hash, with nowhere to record the SQL.
    task = _locked_suite(tmp_path, "SELECT 1", "t1: sha256:frozen\n")

    assert task.gold_hash_for("clickhouse") == "sha256:frozen"


def test_one_engines_fingerprint_does_not_revalidate_anothers_stale_hash(tmp_path: Path) -> None:
    # A cross-engine suite is frozen once per engine, at different times. With a
    # single shared fingerprint, re-freezing ClickHouse after a question was
    # rewritten stamped the new SQL over the entry and silently re-validated the
    # Databricks hash still held from the old question.
    (tmp_path / "s.yaml").write_text(
        "- id: t1\n"
        "  suite: s\n"
        "  engine: [clickhouse, databricks]\n"
        "  question: q\n"
        "  gold_sql: SELECT 1\n",
        encoding="utf-8",
    )
    current = gold_sql_fingerprint("SELECT 1")
    stale = gold_sql_fingerprint("SELECT 2")
    (tmp_path / GOLD_LOCK_NAME).write_text(
        "t1:\n"
        "  clickhouse: sha256:fresh\n"
        f"  clickhouse_sql: {current}\n"
        "  databricks: sha256:stale\n"
        f"  databricks_sql: {stale}\n",
        encoding="utf-8",
    )

    task = load_suite(tmp_path).by_id("t1")

    assert task.gold_hash_for("clickhouse") == "sha256:fresh"
    assert task.gold_hash_for("databricks") is None
