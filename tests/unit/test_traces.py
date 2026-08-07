"""A trace has to let a reader audit one claim end to end."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from agenteval.scorer import Score
from agenteval.systems.base import Attempt, EmittedQuery, TokenUsage
from agenteval.traces import MAX_TRACE_ROWS, TraceWriter, build_record, read_records
from tests.harness_fakes import MODEL, OK, SYNTAX_ERROR, StubSystem, sample_task

SCORE = Score(
    task_id="clickbench_nl_001",
    seed=3,
    verdict="correct",
    execution_accuracy=True,
    accuracy_at_1=True,
    valid_sql=True,
    error_class="none",
    retries=0,
    order_sensitive=False,
    bytes_read=1024,
    input_tokens=100,
    output_tokens=20,
    context_bytes=512,
)

ATTEMPT = Attempt(
    system="A0_baseline",
    task_id="clickbench_nl_001",
    seed=3,
    model=MODEL,
    prompt="tables: hits\n\nHow many rows?",
    queries=(replace(OK, sql="SELECT count() FROM hits"),),
    tokens=TokenUsage(input_tokens=100, output_tokens=20),
    context_bytes=512,
    wall_clock_ms=42,
)


def _record() -> dict[str, object]:
    return build_record(
        run_id="run-20260807T000000Z",
        engine="clickhouse",
        task=sample_task(),
        system=StubSystem(name="A0_baseline", version="1.0"),
        attempt=ATTEMPT,
        score=SCORE,
    )


def test_the_record_identifies_the_run_the_task_and_the_system() -> None:
    record = _record()

    assert record["run_id"] == "run-20260807T000000Z"
    assert record["engine"] == "clickhouse"
    assert record["task_id"] == "clickbench_nl_001"
    assert record["suite"] == "clickbench_nl"
    assert record["system"] == "A0_baseline"
    assert record["system_version"] == "1.0"
    assert record["config_fingerprint"] == "sha256:stub"
    assert record["model"] == "anthropic/claude-opus-5"
    assert record["seed"] == 3


def test_the_record_carries_the_prompt_and_every_query() -> None:
    # Arrange — SPEC 11.4: a reader must be able to audit any single claim
    record = _record()

    assert record["prompt"] == "tables: hits\n\nHow many rows?"
    queries = record["queries"]
    assert isinstance(queries, list)
    assert queries[0]["sql"] == "SELECT count() FROM hits"
    assert queries[0]["rows"] == [[99997497]]
    assert queries[0]["bytes_read"] == 1024


def test_the_record_carries_the_verdict_and_its_reason() -> None:
    record = build_record(
        run_id="r",
        engine="clickhouse",
        task=sample_task(),
        system=StubSystem(),
        attempt=replace(ATTEMPT, queries=(replace(SYNTAX_ERROR, sql="SELEC 1"),)),
        score=replace(
            SCORE, verdict="execution_error", execution_accuracy=False, reason="query failed"
        ),
    )

    assert record["verdict"] == "execution_error"
    assert record["execution_accuracy"] is False
    assert record["reason"] == "query failed"
    queries = record["queries"]
    assert isinstance(queries, list)
    assert queries[0]["error_class"] == "syntax"


def test_a_system_that_picks_its_own_model_records_no_model() -> None:
    record = build_record(
        run_id="r",
        engine="clickhouse",
        task=sample_task(),
        system=StubSystem(controls_model=False),
        attempt=replace(ATTEMPT, model=None),
        score=SCORE,
    )

    assert record["model"] is None
    assert record["controls_model"] is False


def test_large_result_sets_are_truncated_and_flagged() -> None:
    # Arrange — a trace is evidence, not a data export
    rows = tuple((n,) for n in range(MAX_TRACE_ROWS + 5))
    attempt = replace(ATTEMPT, queries=(EmittedQuery(sql="q", succeeded=True, rows=rows),))

    record = build_record(
        run_id="r",
        engine="clickhouse",
        task=sample_task(),
        system=StubSystem(),
        attempt=attempt,
        score=SCORE,
    )

    queries = record["queries"]
    assert isinstance(queries, list)
    assert len(queries[0]["rows"]) == MAX_TRACE_ROWS
    assert queries[0]["rows_truncated"] is True


def test_short_result_sets_are_not_flagged_as_truncated() -> None:
    queries = _record()["queries"]
    assert isinstance(queries, list)
    assert queries[0]["rows_truncated"] is False


def test_records_round_trip_through_a_jsonl_file(tmp_path: Path) -> None:
    # Arrange — the file is created on first write, nested directory and all
    writer = TraceWriter(path=tmp_path / "raw" / "clickbench.jsonl")

    writer.append(_record())
    writer.append(_record())

    records = read_records(writer.path)
    assert len(records) == 2
    assert records[0]["task_id"] == "clickbench_nl_001"


def test_blank_lines_in_a_trace_file_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_text('{"task_id": "a"}\n\n{"task_id": "b"}\n', encoding="utf-8")

    assert [record["task_id"] for record in read_records(path)] == ["a", "b"]
