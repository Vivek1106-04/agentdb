"""The report is a pure function of the traces, and says what it cannot compare."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agenteval.report import ReportError, compare_to_baseline, load_run, render, summarize


def record(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "run_id": "run-1",
        "engine": "clickhouse",
        "suite": "clickbench_nl",
        "task_id": "t1",
        "seed": 0,
        "system": "A0_baseline",
        "system_version": "1.0",
        "controls_model": True,
        "config_fingerprint": "sha256:aaa",
        "model": "anthropic/claude-opus-5",
        "execution_accuracy": True,
        "accuracy_at_1": True,
        "valid_sql": True,
        "retries": 0,
        "error_class": "none",
        "input_tokens": 1000,
        "output_tokens": 50,
        "context_bytes": 4096,
    }
    return {**base, **overrides}


def test_a_summary_is_one_row_per_arm_and_model() -> None:
    summaries = summarize(
        [record(), record(model="anthropic/claude-sonnet-5"), record(system="A7_oracle")]
    )

    assert len(summaries) == 3
    assert {summary.system for summary in summaries} == {"A0_baseline", "A7_oracle"}


def test_a_summary_carries_the_rates_the_spec_requires() -> None:
    summaries = summarize(
        [
            record(task_id="t1", execution_accuracy=True),
            record(task_id="t2", execution_accuracy=False, accuracy_at_1=False, retries=2),
        ]
    )

    only = summaries[0]
    assert only.execution_accuracy.mean == pytest.approx(0.5)
    assert only.accuracy_at_1 == pytest.approx(0.5)
    assert only.valid_sql == pytest.approx(1.0)
    assert only.mean_retries == pytest.approx(1.0)
    assert only.mean_input_tokens == pytest.approx(1000)


def test_successful_cells_do_not_pollute_the_error_taxonomy() -> None:
    summaries = summarize([record(), record(task_id="t2", error_class="syntax")])

    assert dict(summaries[0].errors) == {"syntax": 1}


def test_a_managed_arm_is_labelled_as_choosing_its_own_model() -> None:
    summaries = summarize([record(system="S3", controls_model=False, model=None)])

    assert summaries[0].label == "S3 (system-chosen)"


# --------------------------------------------------------------------------
# paired comparison
# --------------------------------------------------------------------------


def test_arms_are_compared_only_on_the_cells_both_ran() -> None:
    # Arrange — the oracle skipped t2, so t2 must not enter the comparison
    records = [
        record(task_id="t1", execution_accuracy=False),
        record(task_id="t2", execution_accuracy=False),
        record(task_id="t1", system="A7_oracle", execution_accuracy=True),
    ]

    comparisons = compare_to_baseline(records, "A0_baseline")

    assert len(comparisons) == 1
    assert comparisons[0].system == "A7_oracle"
    assert comparisons[0].paired_cells == 1
    assert comparisons[0].test.only_second == 1


def test_an_arm_sharing_no_cells_is_left_out_rather_than_faked() -> None:
    records = [
        record(task_id="t1"),
        record(task_id="t9", system="A7_oracle"),
    ]

    assert compare_to_baseline(records, "A0_baseline") == ()


def test_a_missing_baseline_is_reported() -> None:
    with pytest.raises(ReportError, match="no records for baseline arm 'A9'"):
        compare_to_baseline([record()], "A9")


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def test_the_report_states_its_provenance() -> None:
    markdown = render([record()])

    assert "# agentdb benchmark results" in markdown
    assert "run-1" in markdown
    assert "clickbench_nl" in markdown
    assert "graded cells: 1" in markdown


def test_the_leaderboard_carries_the_interval_and_the_cost() -> None:
    markdown = render([record(), record(task_id="t2", execution_accuracy=False)])

    assert "EX (95% CI)" in markdown
    assert "in tok" in markdown
    assert "`A0_baseline`" in markdown


def test_a_clean_run_says_so_instead_of_printing_an_empty_table() -> None:
    assert "No failures recorded." in render([record()])


def test_the_error_table_appears_when_there_are_errors() -> None:
    markdown = render([record(), record(task_id="t2", error_class="timeout")])

    assert "## Error taxonomy" in markdown
    assert "timeout" in markdown


def test_a_second_arm_produces_a_paired_comparison_section() -> None:
    markdown = render(
        [
            record(task_id="t1", execution_accuracy=False),
            record(task_id="t1", system="A7_oracle", execution_accuracy=True),
        ]
    )

    assert "## Paired comparisons" in markdown
    assert "McNemar" in markdown


def test_a_single_arm_report_has_no_comparison_section() -> None:
    assert "## Paired comparisons" not in render([record()])


def test_every_arm_is_footnoted_with_its_pinned_config() -> None:
    markdown = render([record()])

    assert "sha256:aaa" in markdown
    assert "v1.0" in markdown


def test_a_managed_arm_gets_a_footnote_not_a_silent_asterisk() -> None:
    markdown = render([record(system="S3", controls_model=False, model=None)])

    assert "choose their own model" in markdown
    assert "`S3`" in markdown


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def test_records_load_from_every_trace_file(tmp_path: Path) -> None:
    (tmp_path / "a.jsonl").write_text(json.dumps(record()) + "\n", encoding="utf-8")
    (tmp_path / "b.jsonl").write_text(json.dumps(record(task_id="t2")) + "\n", encoding="utf-8")

    assert len(load_run(tmp_path)) == 2


def test_a_missing_directory_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ReportError, match="no trace directory"):
        load_run(tmp_path / "nope")


def test_an_empty_directory_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ReportError, match="no trace records found"):
        load_run(tmp_path)


def test_a_comparison_section_with_nothing_comparable_renders_empty() -> None:
    # Two arms that share no cell: the section exists but claims nothing
    markdown = render([record(task_id="t1"), record(task_id="t9", system="A7_oracle")])

    assert "## Paired comparisons" not in markdown
