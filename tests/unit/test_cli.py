"""The CLI. Every flag here can change a published number, so every flag is tested."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

from agenteval.__main__ import EXIT_OK, EXIT_USAGE, main
from agenteval.cli import (
    DEFAULT_MODEL,
    QUICK_TASKS,
    BenchOptions,
    CliError,
    FreezeOptions,
    ReportOptions,
    build_systems,
    default_client,
    default_executor,
    default_tool_client,
    executor_factory_for,
    load_grounded_configs,
    load_server_configs,
    parse_args,
    parse_model,
    run_bench,
    run_report,
    summarize,
)
from agenteval.engines.connect import EngineConnectionError
from agenteval.engines.databricks import DatabricksExecutor
from agenteval.execution import QueryExecutor
from agenteval.mcp.base import McpSession, ToolResult, ToolSpec
from agenteval.mcp.config import McpServerConfig, parse_server
from agenteval.models.base import ModelClient, ModelError
from agenteval.models.tools import ToolResponse
from agenteval.report import ReportError
from agenteval.runner import Cell
from agenteval.scorer import Score
from agenteval.systems.base import SystemUnderTest
from agenteval.systems.oracle import ARM_NAME as ORACLE_ARM
from agenteval.systems.providers import ProviderConfig
from agenteval.systems.raw_schema import ARM_NAME
from agenteval.traces import read_records
from tests.harness_fakes import FakeExecutor, ScriptedModelClient

GOOD_REPLY = "```sql\nSELECT count() FROM hits\n```"


def _bench(argv: list[str]) -> BenchOptions:
    options = parse_args(argv)
    assert isinstance(options, BenchOptions)
    return options


# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------


def test_no_subcommand_still_means_bench() -> None:
    options = _bench([])

    assert options.suite == "clickbench_nl"
    assert options.arms == (ARM_NAME,)
    assert options.models == (DEFAULT_MODEL,)
    assert options.seeds == (0, 1, 2, 3, 4)
    assert options.limit is None


def test_a_bare_flag_still_means_bench() -> None:
    assert _bench(["--quick"]).limit == QUICK_TASKS


def test_arms_and_models_are_repeatable() -> None:
    options = _bench(["bench", "--arm", ARM_NAME, "--model", "a/b", "--model", "a/c"])

    assert options.arms == (ARM_NAME,)
    assert options.models == ("a/b", "a/c")


def test_quick_trades_statistical_power_for_five_minutes() -> None:
    options = _bench(["--quick"])

    assert options.seeds == (0,)
    assert options.limit == QUICK_TASKS


def test_seed_count_and_limit_are_explicit() -> None:
    options = _bench(["bench", "--seeds", "3", "--limit", "2", "--out", "/tmp/x"])

    assert options.seeds == (0, 1, 2)
    assert options.limit == 2
    assert options.out == Path("/tmp/x")


def test_a_zero_seed_run_is_refused() -> None:
    with pytest.raises(CliError, match="--seeds must be >= 1"):
        parse_args(["bench", "--seeds", "0"])


def test_report_is_its_own_command() -> None:
    options = parse_args(["report", "--from-raw", "traces", "--baseline", ORACLE_ARM])

    assert isinstance(options, ReportOptions)
    assert options.from_raw == Path("traces")
    assert options.baseline == ORACLE_ARM


def test_freeze_gold_is_its_own_command() -> None:
    options = parse_args(["freeze-gold", "--suite", "clickbench_nl"])

    assert isinstance(options, FreezeOptions)
    assert options.suite == "clickbench_nl"


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


async def _build(arms: list[str], **overrides: object) -> tuple[SystemUnderTest, ...]:
    defaults: dict[str, Any] = {
        "client": ScriptedModelClient(),
        "tool_client": StubToolClient(),
        "servers": {},
        "providers": {},
        "connector": _refuse_to_connect,
        "sessions": [],
    }
    return await build_systems(arms, FakeExecutor(), **{**defaults, **overrides})


async def _refuse_to_connect(config: McpServerConfig) -> McpSession:
    raise AssertionError("no MCP session should be opened for a local arm")


@dataclass
class StubToolClient:
    provider: str = "fake"

    async def complete_with_tools(self, **kwargs: object) -> ToolResponse:
        return ToolResponse(text="")


async def test_both_shipped_arms_are_constructible() -> None:
    systems = await _build([ARM_NAME, ORACLE_ARM])

    assert [system.name for system in systems] == [ARM_NAME, ORACLE_ARM]


@dataclass
class StubProvider:
    """A context provider the CLI can resolve by dotted path, with no engine behind it."""

    name: str = "agentdb/A1_stats"
    version: str = "1.0"
    fingerprint: str = "sha256:stub-provider"

    async def context(self, *, namespace: str, question: str) -> str:
        return f"grounded context for {namespace}"


def stub_provider_factory(**options: object) -> StubProvider:
    """Named in a :class:`ProviderConfig` so the loader resolves this module."""
    return StubProvider(name=str(options.get("name", "agentdb/A1_stats")))


async def test_a_family_a_arm_is_built_from_its_provider_config() -> None:
    config = ProviderConfig(
        arm="A1_stats",
        provider="tests.unit.test_cli:stub_provider_factory",
        options={"name": "agentdb/A1_stats"},
    )

    systems = await _build(["A1_stats"], providers={"A1_stats": config})

    assert [system.name for system in systems] == ["A1_stats"]


def test_a_missing_provider_catalogue_simply_means_no_family_a_arms(tmp_path: Path) -> None:
    assert load_grounded_configs(tmp_path / "absent.yaml") == {}


def test_the_shipped_provider_catalogue_is_keyed_by_arm() -> None:
    configs = load_grounded_configs(Path("eval/providers.yaml"))

    assert sorted(configs) == ["A1_stats", "A2_layout", "A3_plan"]


async def test_an_arm_that_does_not_exist_yet_is_named() -> None:
    with pytest.raises(CliError, match="unknown arm\\(s\\) \\['A9_future'\\]"):
        await _build(["A9_future"])


async def test_a_family_s_arm_is_built_from_its_server_config() -> None:
    opened: list[McpSession] = []
    server = parse_server({"name": "S1_mcp_clickhouse", "version": "0.1.12", "command": "uvx"})

    async def connector(config: McpServerConfig) -> McpSession:
        return StubSession()

    systems = await _build(
        ["S1_mcp_clickhouse"],
        servers={"S1_mcp_clickhouse": server},
        connector=connector,
        sessions=opened,
    )

    assert [system.name for system in systems] == ["S1_mcp_clickhouse"]
    assert len(opened) == 1


async def test_every_opened_session_is_handed_back_for_closing() -> None:
    # A leaked server process poisons the next arm's timings
    sessions: list[McpSession] = []
    server = parse_server({"name": "S1", "version": "1", "command": "uvx"})

    async def connector(config: McpServerConfig) -> McpSession:
        return StubSession()

    await _build(["S1"], servers={"S1": server}, connector=connector, sessions=sessions)

    assert len(sessions) == 1


@dataclass
class StubSession:
    closed: bool = False

    async def list_tools(self) -> tuple[ToolSpec, ...]:
        return (ToolSpec(name="run_select_query"),)

    async def call_tool(self, name: str, arguments: Mapping[str, Any]) -> ToolResult:
        return ToolResult(content="")

    async def close(self) -> None:
        self.closed = True


def test_server_configs_load_from_the_shipped_file() -> None:
    servers = load_server_configs(Path("eval/servers.yaml"))

    assert "S1_mcp_clickhouse" in servers
    assert servers["S1_mcp_clickhouse"].query_tools == ("run_select_query",)


def test_a_missing_server_file_simply_means_no_family_s(tmp_path: Path) -> None:
    assert load_server_configs(tmp_path / "nope.yaml") == {}


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


async def _run(options: BenchOptions, lines: list[str]) -> tuple[Cell, ...]:
    executor = FakeExecutor()
    client = ScriptedModelClient(replies=[GOOD_REPLY] * 40)

    async def make_executor() -> QueryExecutor:
        return executor

    def make_client() -> ModelClient:
        return client

    return await run_bench(
        options,
        executor_factory=make_executor,
        client_factory=make_client,
        write=lines.append,
        tool_client_factory=StubToolClient,
        connector=_refuse_to_connect,
    )


async def test_a_run_writes_traces_and_reports_a_number(tmp_path: Path) -> None:
    lines: list[str] = []

    cells = await _run(BenchOptions(seeds=(0,), limit=2, out=tmp_path), lines)

    assert len(cells) == 2
    assert any("EX" in line for line in lines)
    written = list(tmp_path.glob("*.jsonl"))
    assert len(written) == 1
    assert len(read_records(written[0])) == 2


async def test_both_arms_run_in_one_pass(tmp_path: Path) -> None:
    cells = await _run(
        BenchOptions(arms=(ARM_NAME, ORACLE_ARM), seeds=(0,), limit=1, out=tmp_path), []
    )

    assert {cell.system for cell in cells} == {ARM_NAME, ORACLE_ARM}


async def test_the_limit_is_applied_to_the_suite(tmp_path: Path) -> None:
    assert len(await _run(BenchOptions(seeds=(0,), limit=1, out=tmp_path), [])) == 1


def test_the_report_command_writes_markdown_from_traces(tmp_path: Path) -> None:
    # Arrange
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "run.jsonl").write_text(_trace_line(), encoding="utf-8")
    out = tmp_path / "REPORT.md"
    lines: list[str] = []

    # Act
    markdown = run_report(ReportOptions(from_raw=raw, out=out), write=lines.append)

    # Assert
    assert "# agentdb benchmark results" in markdown
    assert out.read_text(encoding="utf-8") == markdown
    assert "1 records" in lines[0]


def test_the_report_command_reports_a_missing_trace_directory(tmp_path: Path) -> None:
    with pytest.raises(ReportError, match="no trace directory"):
        run_report(ReportOptions(from_raw=tmp_path / "nope"), write=lambda _: None)


def _trace_line() -> str:
    import json

    return json.dumps(
        {
            "run_id": "r",
            "engine": "clickhouse",
            "suite": "clickbench_nl",
            "task_id": "t1",
            "seed": 0,
            "system": ARM_NAME,
            "system_version": "1.0",
            "controls_model": True,
            "config_fingerprint": "sha256:x",
            "model": "anthropic/claude-opus-5",
            "execution_accuracy": True,
            "accuracy_at_1": True,
            "valid_sql": True,
            "retries": 0,
            "error_class": "none",
            "input_tokens": 10,
            "output_tokens": 2,
            "context_bytes": 100,
        }
    )


# --------------------------------------------------------------------------
# the process entry point
# --------------------------------------------------------------------------


async def test_the_default_executor_needs_a_driver_and_a_server() -> None:
    with pytest.raises(EngineConnectionError):
        await default_executor()


async def test_the_databricks_executor_needs_a_configured_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AGENTEVAL_DBX_HOST", raising=False)
    monkeypatch.delenv("AGENTEVAL_DBX_WAREHOUSE_ID", raising=False)

    # refusing to start beats measuring a workspace nobody chose
    with pytest.raises(EngineConnectionError, match="AGENTEVAL_DBX_HOST"):
        await default_executor("databricks")


async def test_the_databricks_executor_is_built_against_the_configured_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENTEVAL_DBX_HOST", "https://dbc-test.cloud.databricks.com")
    monkeypatch.setenv("AGENTEVAL_DBX_WAREHOUSE_ID", "abc123")
    monkeypatch.setenv("AGENTEVAL_DBX_CATALOG", "main")

    async def fake_client(target: Any) -> Any:
        return object()

    monkeypatch.setattr("agenteval.cli.build_databricks_client", fake_client)

    executor = await default_executor("databricks")

    assert executor.engine == "databricks"
    assert isinstance(executor, DatabricksExecutor)
    assert executor.catalog == "main"


async def test_the_engine_flag_selects_which_executor_a_run_builds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AGENTEVAL_DBX_HOST", raising=False)
    factory = executor_factory_for("databricks")

    with pytest.raises(EngineConnectionError, match="AGENTEVAL_DBX_HOST"):
        await factory()


def test_bench_and_freeze_both_take_an_engine_defaulting_to_clickhouse() -> None:
    bench = parse_args(["bench", "--engine", "databricks", "--suite", "tpch_nl"])
    freeze = parse_args(["freeze-gold", "--engine", "databricks"])

    assert isinstance(bench, BenchOptions)
    assert bench.engine == "databricks"
    assert isinstance(freeze, FreezeOptions)
    assert freeze.engine == "databricks"
    assert cast(BenchOptions, parse_args(["bench"])).engine == "clickhouse"


def test_the_default_client_needs_an_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(ModelError):
        default_client()


def test_the_default_tool_client_needs_an_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(ModelError):
        default_tool_client()


def test_a_setup_failure_exits_distinctly_from_a_crash(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["bench", "--seeds", "0"])

    assert code == EXIT_USAGE
    assert "error:" in capsys.readouterr().err


def test_the_bench_path_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run(*args: object, **kwargs: object) -> tuple[()]:
        return ()

    monkeypatch.setattr("agenteval.__main__.run_bench", fake_run)

    assert main(["--quick"]) == EXIT_OK


def test_the_report_path_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agenteval.__main__.run_report", lambda *a, **k: "")

    assert main(["report"]) == EXIT_OK


def test_the_freeze_path_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_freeze(*args: object, **kwargs: object) -> Path:
        return Path("gold.lock.yaml")

    monkeypatch.setattr("agenteval.__main__.run_freeze", fake_freeze)

    assert main(["freeze-gold"]) == EXIT_OK


def test_argv_defaults_to_the_process_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run(*args: object, **kwargs: object) -> tuple[()]:
        return ()

    monkeypatch.setattr("agenteval.__main__.run_bench", fake_run)
    monkeypatch.setattr("sys.argv", ["agenteval", "--quick"])

    assert main() == EXIT_OK


async def test_the_freeze_command_writes_a_lock_beside_the_suite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange — redirect the write so a test never edits a shipped suite
    from agenteval.cli import FreezeOptions as _FreezeOptions
    from agenteval.cli import run_freeze

    monkeypatch.setattr("agenteval.cli.SUITES_DIR", tmp_path)
    (tmp_path / "clickbench_nl").mkdir()
    executor = FakeExecutor()
    lines: list[str] = []

    async def make_executor() -> QueryExecutor:
        return executor

    # Act
    path = await run_freeze(
        _FreezeOptions(suite="clickbench_nl"), executor_factory=make_executor, write=lines.append
    )

    # Assert — one hash per shipped ClickHouse task
    assert path.name == "gold.lock.yaml"
    assert "froze 20 gold result(s)" in lines[0]


@dataclass
class StubAdvisingProvider(StubProvider):
    """A provider that can also explain a plan — what an A3 arm requires."""

    async def explain_plan(self, *, sql: str, namespace: str) -> str | None:
        return f"plan for {sql}"


def advising_provider_factory(**options: object) -> StubAdvisingProvider:
    return StubAdvisingProvider(name=str(options.get("name", "agentdb/A3_plan")))


async def test_a_plan_review_arm_is_built_when_the_provider_can_explain_one() -> None:
    config = ProviderConfig(
        arm="A3_plan",
        provider="tests.unit.test_cli:advising_provider_factory",
        plan_review=True,
    )

    systems = await _build(["A3_plan"], providers={"A3_plan": config})

    assert [system.name for system in systems] == ["A3_plan"]


async def test_asking_for_plan_review_from_a_provider_that_cannot_is_refused() -> None:
    config = ProviderConfig(
        arm="A3_plan",
        provider="tests.unit.test_cli:stub_provider_factory",
        plan_review=True,
    )

    with pytest.raises(CliError, match="cannot explain a plan"):
        await _build(["A3_plan"], providers={"A3_plan": config})
