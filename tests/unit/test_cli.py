"""The CLI. Every flag here can change a published number, so every flag is tested."""

from __future__ import annotations

from pathlib import Path

import pytest

from agenteval.__main__ import EXIT_OK, EXIT_USAGE, main
from agenteval.cli import (
    DEFAULT_MODEL,
    QUICK_TASKS,
    CliError,
    Options,
    build_systems,
    default_client,
    default_executor,
    parse_args,
    parse_model,
    run_options,
    summarize,
)
from agenteval.engines.connect import EngineConnectionError
from agenteval.execution import QueryExecutor
from agenteval.models.base import ModelClient, ModelError
from agenteval.runner import Cell
from agenteval.scorer import Score
from agenteval.systems.raw_schema import ARM_NAME
from agenteval.traces import read_records
from tests.harness_fakes import FakeExecutor, ScriptedModelClient

# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------


def test_the_defaults_are_a_full_five_seed_run() -> None:
    options = parse_args([])

    assert options.suite == "clickbench_nl"
    assert options.arms == (ARM_NAME,)
    assert options.models == (DEFAULT_MODEL,)
    assert options.seeds == (0, 1, 2, 3, 4)
    assert options.limit is None


def test_arms_and_models_are_repeatable() -> None:
    options = parse_args(["--arm", "A0_baseline", "--model", "a/b", "--model", "a/c"])

    assert options.arms == ("A0_baseline",)
    assert options.models == ("a/b", "a/c")


def test_quick_trades_statistical_power_for_five_minutes() -> None:
    options = parse_args(["--quick"])

    assert options.seeds == (0,)
    assert options.limit == QUICK_TASKS


def test_seed_count_and_limit_are_explicit() -> None:
    options = parse_args(["--seeds", "3", "--limit", "2", "--out", "/tmp/x"])

    assert options.seeds == (0, 1, 2)
    assert options.limit == 2
    assert options.out == Path("/tmp/x")


def test_a_zero_seed_run_is_refused() -> None:
    with pytest.raises(CliError, match="--seeds must be >= 1"):
        parse_args(["--seeds", "0"])


# --------------------------------------------------------------------------
# building the run
# --------------------------------------------------------------------------


def test_a_model_is_written_provider_slash_name() -> None:
    spec = parse_model("anthropic/claude-opus-5")

    assert spec.provider == "anthropic"
    assert spec.name == "claude-opus-5"


@pytest.mark.parametrize("bad", ["claude-opus-5", "/name", "anthropic/"])
def test_a_malformed_model_is_refused(bad: str) -> None:
    with pytest.raises(CliError, match="provider/name"):
        parse_model(bad)


def test_a_provider_with_no_adapter_is_refused() -> None:
    with pytest.raises(CliError, match="no adapter for provider 'openai'"):
        parse_model("openai/gpt-5")


def test_known_arms_are_constructed() -> None:
    systems = build_systems([ARM_NAME], FakeExecutor(), ScriptedModelClient())

    assert [system.name for system in systems] == [ARM_NAME]


def test_an_arm_that_does_not_exist_yet_is_named() -> None:
    with pytest.raises(CliError, match="unknown arm\\(s\\) \\['A9_future'\\]"):
        build_systems(["A9_future"], FakeExecutor(), ScriptedModelClient())


def test_the_summary_reports_execution_accuracy_per_arm() -> None:
    def cell(system: str, correct: bool) -> Cell:
        return Cell(
            system=system,
            task_id="t",
            model="m",
            seed=0,
            score=Score(
                task_id="t",
                seed=0,
                verdict="correct" if correct else "incorrect",
                execution_accuracy=correct,
                accuracy_at_1=correct,
                valid_sql=True,
                error_class="none",
                retries=0,
                order_sensitive=False,
            ),
        )

    lines = summarize([cell("A0", True), cell("A0", False), cell("S4", True)])

    assert lines == ("A0: EX 1/2 = 50.0%", "S4: EX 1/1 = 100.0%")


# --------------------------------------------------------------------------
# running end to end
# --------------------------------------------------------------------------


async def test_a_run_writes_traces_and_reports_a_number(tmp_path: Path) -> None:
    # Arrange — a full pass with no server and no API key
    executor = FakeExecutor()
    client = ScriptedModelClient(replies=["```sql\nSELECT count() FROM hits\n```"] * 4)
    lines: list[str] = []

    async def make_executor() -> QueryExecutor:
        return executor

    def make_client() -> ModelClient:
        return client

    options = Options(arms=(ARM_NAME,), seeds=(0,), limit=2, out=tmp_path)

    # Act
    cells = await run_options(
        options,
        executor_factory=make_executor,
        client_factory=make_client,
        write=lines.append,
    )

    # Assert
    assert len(cells) == 2
    assert any("EX" in line for line in lines)
    written = list(tmp_path.glob("*.jsonl"))
    assert len(written) == 1
    assert len(read_records(written[0])) == 2


async def test_the_limit_is_applied_to_the_suite(tmp_path: Path) -> None:
    executor = FakeExecutor()
    client = ScriptedModelClient(replies=["```sql\nSELECT count() FROM hits\n```"] * 10)

    async def make_executor() -> QueryExecutor:
        return executor

    cells = await run_options(
        Options(seeds=(0,), limit=1, out=tmp_path),
        executor_factory=make_executor,
        client_factory=lambda: client,
        write=lambda _: None,
    )

    assert len(cells) == 1


# --------------------------------------------------------------------------
# the process entry point
# --------------------------------------------------------------------------


async def test_the_default_executor_needs_a_driver_and_a_server() -> None:
    # The optional extra is not installed in CI, which is the point: the harness
    # imports and its tests run without one
    with pytest.raises(EngineConnectionError):
        await default_executor()


def test_the_default_client_needs_an_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(ModelError):
        default_client()


def test_a_setup_failure_exits_distinctly_from_a_crash(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Arrange — no ANTHROPIC_API_KEY and no engine: the reader gets one line
    code = main(["--seeds", "0"])

    assert code == EXIT_USAGE
    assert "error:" in capsys.readouterr().err


def test_the_success_path_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run_options(*args: object, **kwargs: object) -> tuple[()]:
        return ()

    monkeypatch.setattr("agenteval.__main__.run_options", fake_run_options)

    assert main(["--quick"]) == EXIT_OK


def test_argv_defaults_to_the_process_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run_options(*args: object, **kwargs: object) -> tuple[()]:
        return ()

    monkeypatch.setattr("agenteval.__main__.run_options", fake_run_options)
    monkeypatch.setattr("sys.argv", ["agenteval", "--quick"])

    assert main() == EXIT_OK
