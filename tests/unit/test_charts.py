"""Charts are a function of the traces, and are absent when the traces cannot support them."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agenteval.charts import build_charts, write_charts


def record(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "run_id": "run-20260815T082806Z",
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
        "bytes_read": None,
        "input_tokens": 1000,
        "output_tokens": 50,
        "context_bytes": 4096,
    }
    return {**base, **overrides}


def filenames(records: list[dict[str, Any]]) -> set[str]:
    return {chart.filename for chart in build_charts(records)}


def test_a_family_a_run_gets_the_ablation_chart_not_the_leaderboard() -> None:
    names = filenames([record(system="A0_baseline"), record(system="A7_oracle")])

    assert "02-family-a-ablation.svg" in names
    assert "01-family-s-leaderboard.svg" not in names


def test_anything_not_named_like_an_arm_lands_on_the_leaderboard() -> None:
    names = filenames([record(system="S1_mcp_clickhouse"), record(system="genie")])

    assert "01-family-s-leaderboard.svg" in names
    assert "02-family-a-ablation.svg" not in names


def test_the_cross_engine_chart_needs_two_engines_on_the_shared_suite() -> None:
    one_engine = [record(suite="tpch_nl"), record(suite="tpch_nl", task_id="t2")]
    assert "03-cross-engine-tpch.svg" not in filenames(one_engine)

    both = [*one_engine, record(suite="tpch_nl", engine="databricks")]
    assert "03-cross-engine-tpch.svg" in filenames(both)


def test_a_clean_run_publishes_no_error_chart() -> None:
    assert "04-error-classes.svg" not in filenames([record()])
    assert "04-error-classes.svg" in filenames(
        [record(execution_accuracy=False, error_class="wrong_result")]
    )


def test_bytes_are_charted_only_when_the_engine_reported_them() -> None:
    assert "05-bytes-read.svg" not in filenames([record()])

    charts = build_charts([record(bytes_read=4096), record(task_id="t2", bytes_read=8192)])
    bytes_chart = next(chart for chart in charts if chart.filename == "05-bytes-read.svg")
    assert "6.0 KB" in bytes_chart.svg  # the mean of the two, not the sum


def test_the_cost_scatter_carries_one_point_per_arm_and_suite() -> None:
    charts = build_charts(
        [record(system="A0_baseline"), record(system="A3_grounded", suite="tpch_nl")]
    )
    scatter = next(chart for chart in charts if chart.filename == "06-accuracy-vs-cost.svg")

    assert scatter.svg.count("<circle") == 2
    assert "clickbench_nl" in scatter.svg
    assert "tpch_nl" in scatter.svg


def test_no_records_means_no_charts() -> None:
    assert build_charts([]) == ()


def test_the_headline_chart_states_its_measurement_dates() -> None:
    charts = build_charts([record(system="S1_mcp_clickhouse")])

    assert "2026-08-15" in charts[0].svg


def test_a_run_id_without_a_date_says_so_rather_than_inventing_one() -> None:
    charts = build_charts([record(system="S1_mcp_clickhouse", run_id="local")])

    assert "date not recorded in the run id" in charts[0].svg


def test_writing_charts_creates_the_directory_and_returns_the_paths(tmp_path: Path) -> None:
    charts = build_charts([record(execution_accuracy=False, error_class="wrong_result")])

    written = write_charts(charts, tmp_path / "charts")

    assert len(written) == len(charts)
    assert all(path.read_text(encoding="utf-8").startswith("<svg") for path in written)
