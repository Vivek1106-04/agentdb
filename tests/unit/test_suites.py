"""The shipped suites are data, and a benchmark's data gets checked like code."""

from __future__ import annotations

import pytest

from agenteval.suites import SUITES_DIR, builtin_suite_names, load_builtin
from agenteval.tasks import TaskLoadError, TaskSuite


@pytest.fixture(scope="module")
def clickbench() -> TaskSuite:
    return load_builtin("clickbench_nl")


def test_clickbench_nl_ships_with_the_harness() -> None:
    assert "clickbench_nl" in builtin_suite_names()


def test_an_unknown_suite_names_the_ones_that_exist() -> None:
    with pytest.raises(TaskLoadError, match="unknown suite 'bird_subset'"):
        load_builtin("bird_subset")


def test_m1_ships_at_least_twenty_clickhouse_tasks(clickbench: TaskSuite) -> None:
    # SPEC 15: M1 is 20 hand-written ClickHouse tasks
    assert len(clickbench) >= 20
    assert all(task.targets("clickhouse") for task in clickbench)


def test_task_ids_are_unique_and_sorted(clickbench: TaskSuite) -> None:
    ids = [task.id for task in clickbench]

    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)


def test_every_task_carries_a_question_and_a_gold_query(clickbench: TaskSuite) -> None:
    for task in clickbench:
        assert task.question.strip()
        assert task.gold_sql.strip().upper().startswith(("SELECT", "WITH"))


def test_no_gold_query_limits_without_ordering(clickbench: TaskSuite) -> None:
    # A LIMIT with no ORDER BY has no single correct answer to grade against
    for task in clickbench:
        sql = task.gold_sql.upper()
        if "LIMIT" in sql:
            assert "ORDER BY" in sql, f"{task.id} limits an unordered result"


def test_clickbench_provenance_is_traceable(clickbench: TaskSuite) -> None:
    # Ids keep the ClickBench query number so a reader can check the translation
    numbered = [task.id for task in clickbench if task.id.startswith("clickbench_nl_")]

    assert len(numbered) == len(clickbench)
    assert all(task.id.rsplit("_", 1)[-1].isdigit() for task in clickbench)


@pytest.fixture(scope="module")
def tpch() -> TaskSuite:
    return load_builtin("tpch_nl")


def test_tpch_nl_ships_as_the_cross_engine_suite() -> None:
    # clickbench_nl cannot cross: hits is ~100M rows over 105 columns and will
    # not fit a Databricks Free Edition workspace (spec 1.2)
    assert "tpch_nl" in builtin_suite_names()


def test_every_tpch_task_is_shaped_like_a_benchmark_task(tpch: TaskSuite) -> None:
    ids = [task.id for task in tpch]

    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)
    for task in tpch:
        assert task.question.strip()
        assert task.gold_sql.strip().upper().startswith(("SELECT", "WITH"))
        assert task.notes, f"{task.id} does not say what it probes"


def test_tpch_gold_never_limits_an_unordered_result(tpch: TaskSuite) -> None:
    for task in tpch:
        sql = task.gold_sql.upper()
        if "LIMIT" in sql and "ORDER BY DESC" not in sql:
            assert "ORDER BY" in sql, f"{task.id} limits an unordered result"


def test_tpch_tasks_share_one_namespace_so_both_engines_resolve_them(
    tpch: TaskSuite,
) -> None:
    # "tpch" is a ClickHouse database and, with the adapter's configured
    # catalog, the Databricks samples.tpch schema
    assert {task.namespace for task in tpch} == {"tpch"}


def test_tpch_gold_avoids_engine_specific_syntax(tpch: TaskSuite) -> None:
    # gold that runs on one engine only is not cross-engine gold (SPEC 18.6)
    clickhouse_only = ("TOPK(", "UNIQCOMBINED", "TODATE(", "TOYEAR(", "ARRAYJOIN", "FINAL")
    databricks_only = ("TABLESAMPLE", "LATERAL VIEW", "DATE_TRUNC(")
    for task in tpch:
        sql = task.gold_sql.upper()
        for token in (*clickhouse_only, *databricks_only):
            assert token not in sql, f"{task.id} uses engine-specific {token}"


def test_the_suite_documents_itself() -> None:
    # SPEC 11.2: clickbench_nl is published as a standalone, citable dataset
    readme = SUITES_DIR / "clickbench_nl" / "README.md"

    assert readme.is_file()
    assert "ClickBench" in readme.read_text(encoding="utf-8")


def test_the_cross_engine_suite_states_its_data_requirement() -> None:
    # identical questions over different data are two experiments, not one
    readme = (SUITES_DIR / "tpch_nl" / "README.md").read_text(encoding="utf-8")

    assert "samples.tpch" in readme
    assert "scale factor" in readme.lower()
