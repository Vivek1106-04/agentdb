"""``python -m agenteval`` — the entry point behind ``make bench``.

Three commands, because there are three distinct things a reader does:

* ``bench`` — spend model calls and produce traces.
* ``report`` — turn committed traces into ``REPORT.md``. No model, no engine.
* ``freeze-gold`` — verify gold against trusted data once, and commit the hashes.

Every knob that changes a published number is an explicit flag with a recorded
default. Nothing is inferred from the environment except *where* the engine and
the API key live, because a run whose shape depends on the machine it ran on is
not reproducible.
"""

from __future__ import annotations

import argparse
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from agenteval.engines.clickhouse import ClickHouseExecutor
from agenteval.engines.connect import (
    ClickHouseTarget,
    DatabricksTarget,
    build_client,
    build_databricks_client,
)
from agenteval.engines.databricks import DatabricksExecutor
from agenteval.execution import QueryExecutor
from agenteval.freeze import compute_gold_hashes, write_gold_lock
from agenteval.mcp.base import McpSession
from agenteval.mcp.config import McpServerConfig, load_servers
from agenteval.mcp.stdio import connect
from agenteval.models.anthropic import PROVIDER, AnthropicClient, build_create
from agenteval.models.anthropic_tools import AnthropicToolClient, MessageCreateWithTools
from agenteval.models.base import ModelClient
from agenteval.models.claude_cli import PROVIDER as CLI_PROVIDER
from agenteval.models.claude_cli import ClaudeCliClient
from agenteval.models.tools import ToolUsingClient
from agenteval.report import load_run, render
from agenteval.runner import Cell, RunSpec, new_run_id, run
from agenteval.suites import SUITES_DIR, load_builtin
from agenteval.systems.base import ModelSpec, SystemUnderTest
from agenteval.systems.claude_code import ARM_NAME as CLAUDE_CODE_ARM
from agenteval.systems.claude_code import ClaudeCodeSystem
from agenteval.systems.grounded import GroundedSystem
from agenteval.systems.mcp_generic import McpSystem
from agenteval.systems.oracle import ARM_NAME as ORACLE_ARM
from agenteval.systems.oracle import OracleSystem
from agenteval.systems.plan_aware import PlanAdvisor, PlanAwareSystem
from agenteval.systems.providers import ProviderConfig, load_provider, load_provider_configs
from agenteval.systems.raw_schema import ARM_NAME as BASELINE_ARM
from agenteval.systems.raw_schema import RawSchemaSystem
from agenteval.tasks import Engine, gold_sql_fingerprint
from agenteval.traces import TraceWriter

DEFAULT_SUITE = "clickbench_nl"
DEFAULT_ENGINE: Engine = "clickhouse"
ENGINES: tuple[Engine, ...] = ("clickhouse", "databricks")
DEFAULT_MODEL = f"{PROVIDER}/claude-opus-5"
MODEL_PROVIDERS: frozenset[str] = frozenset({PROVIDER, CLI_PROVIDER})
"""Model channels a run may name. ``claude-cli`` drives Claude Code through a
subscription and carries the product's own context (SPEC §11.5); it is a Family
S measurement and never a Family A one."""
DEFAULT_RAW = Path("results/raw")
DEFAULT_SERVERS = Path("eval/servers.yaml")
DEFAULT_PROVIDERS = Path("eval/providers.yaml")
DEFAULT_REPORT = Path("results/REPORT.md")

QUICK_TASKS = 5
"""``--quick`` exists so a reader can reproduce *something* in five minutes
(SPEC §14.1). It is never the number that gets published."""

ARMS: dict[str, Callable[[QueryExecutor, ModelClient], SystemUnderTest]] = {
    BASELINE_ARM: lambda executor, client: RawSchemaSystem.create(executor=executor, client=client),
    ORACLE_ARM: lambda executor, client: OracleSystem.create(executor=executor, client=client),
    CLAUDE_CODE_ARM: lambda executor, client: ClaudeCodeSystem.create(
        executor=executor, client=client
    ),
}
"""Arms that can be constructed today. Grows as A1-A6 and S1-S4 land."""

Writer = Callable[[str], None]
"""Where progress goes. Injected so a test reads the output instead of stdout."""

ExecutorFactory = Callable[[], Awaitable[QueryExecutor]]
ClientFactory = Callable[[], ModelClient]
ToolClientFactory = Callable[[], ToolUsingClient]
Connector = Callable[[McpServerConfig], Awaitable[McpSession]]


class CliError(RuntimeError):
    """A run was asked for that cannot be built. Reported, never guessed around."""


@dataclass(frozen=True, slots=True)
class BenchOptions:
    """One fully-specified benchmark run."""

    suite: str = DEFAULT_SUITE
    engine: Engine = DEFAULT_ENGINE
    arms: tuple[str, ...] = (BASELINE_ARM,)
    models: tuple[str, ...] = (DEFAULT_MODEL,)
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    limit: int | None = None
    out: Path = DEFAULT_RAW
    servers: Path = DEFAULT_SERVERS
    providers: Path = DEFAULT_PROVIDERS


@dataclass(frozen=True, slots=True)
class ReportOptions:
    """Regenerating the report from committed evidence."""

    from_raw: Path = DEFAULT_RAW
    out: Path = DEFAULT_REPORT
    baseline: str = BASELINE_ARM


@dataclass(frozen=True, slots=True)
class FreezeOptions:
    """Committing verified gold hashes for a suite."""

    suite: str = DEFAULT_SUITE
    engine: Engine = DEFAULT_ENGINE


Command = BenchOptions | ReportOptions | FreezeOptions


def parse_args(argv: Sequence[str]) -> Command:
    """Turn a command line into the options for one command."""
    parser = argparse.ArgumentParser(prog="agenteval", description="The NL-to-SQL benchmark.")
    commands = parser.add_subparsers(dest="command", required=True)

    bench = commands.add_parser("bench", help="run the matrix and write traces")
    bench.add_argument("--suite", default=DEFAULT_SUITE, help="shipped suite to run")
    bench.add_argument(
        "--engine",
        default=DEFAULT_ENGINE,
        choices=ENGINES,
        help="engine to measure against; the suite is filtered to tasks targeting it",
    )
    bench.add_argument("--arm", action="append", dest="arms", help="repeatable; defaults to A0")
    bench.add_argument("--model", action="append", dest="models", help="provider/name, repeatable")
    bench.add_argument("--seeds", type=int, default=5, help="repetitions per cell (>=5)")
    bench.add_argument("--limit", type=int, default=None, help="first N tasks only")
    bench.add_argument("--quick", action="store_true", help=f"{QUICK_TASKS} tasks, one seed")
    bench.add_argument("--out", type=Path, default=DEFAULT_RAW, help="trace directory")
    bench.add_argument(
        "--servers", type=Path, default=DEFAULT_SERVERS, help="Family S server configs"
    )
    bench.add_argument(
        "--providers", type=Path, default=DEFAULT_PROVIDERS, help="Family A provider configs"
    )

    report = commands.add_parser("report", help="regenerate REPORT.md from traces")
    report.add_argument("--from-raw", type=Path, default=DEFAULT_RAW, dest="from_raw")
    report.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    report.add_argument("--baseline", default=BASELINE_ARM, help="arm others are compared to")

    freeze = commands.add_parser("freeze-gold", help="verify and commit gold hashes")
    freeze.add_argument("--suite", default=DEFAULT_SUITE)
    freeze.add_argument("--engine", default=DEFAULT_ENGINE, choices=ENGINES)

    args = parser.parse_args(_with_default_command(argv))

    if args.command == "report":
        return ReportOptions(from_raw=args.from_raw, out=args.out, baseline=args.baseline)
    if args.command == "freeze-gold":
        return FreezeOptions(suite=args.suite, engine=args.engine)

    if args.seeds < 1:
        raise CliError(f"--seeds must be >= 1, got {args.seeds}")
    return BenchOptions(
        suite=args.suite,
        engine=args.engine,
        arms=tuple(args.arms or (BASELINE_ARM,)),
        models=tuple(args.models or (DEFAULT_MODEL,)),
        seeds=(0,) if args.quick else tuple(range(args.seeds)),
        limit=QUICK_TASKS if args.quick else args.limit,
        out=args.out,
        servers=args.servers,
        providers=args.providers,
    )


def _with_default_command(argv: Sequence[str]) -> list[str]:
    """``python -m agenteval --quick`` still means ``bench``."""
    if not argv or argv[0].startswith("-"):
        return ["bench", *argv]
    return list(argv)


def parse_model(spec: str) -> ModelSpec:
    """``provider/name`` into a :class:`ModelSpec`."""
    provider, separator, name = spec.partition("/")
    if not separator or not provider or not name:
        raise CliError(f"model {spec!r} must be written provider/name, e.g. {DEFAULT_MODEL}")
    if provider not in MODEL_PROVIDERS:
        raise CliError(
            f"no adapter for provider {provider!r}; available: {', '.join(sorted(MODEL_PROVIDERS))}"
        )
    return ModelSpec(provider=provider, name=name)


def load_server_configs(path: Path) -> dict[str, McpServerConfig]:
    """Family S servers, by arm name. A missing file simply means no S arms."""
    if not path.is_file():
        return {}
    return {server.name: server for server in load_servers(path)}


def load_grounded_configs(path: Path) -> dict[str, ProviderConfig]:
    """Family A grounded arms, by arm name. A missing file means A0 and A7 only."""
    if not path.is_file():
        return {}
    return {config.arm: config for config in load_provider_configs(path)}


def load_provider_catalogue(path: Path) -> tuple[ProviderConfig, ...]:
    """Every Family A entry in ``path``. A missing file means A0 and A7 only."""
    return load_provider_configs(path) if path.is_file() else ()


def select_provider(
    arm: str, configs: Sequence[ProviderConfig], engine: Engine
) -> ProviderConfig | None:
    """The entry for ``arm`` on ``engine``: engine-specific first, then either-engine.

    An arm defined only for the other engine returns ``None`` and the caller
    refuses the run. Falling back to it would ground against ClickHouse while
    executing on a warehouse — a run that completes and reports numbers nobody
    could reproduce.
    """
    named = [config for config in configs if config.arm == arm]
    exact = [config for config in named if config.engine == engine]
    if exact:
        return exact[0]
    generic = [config for config in named if config.engine is None]
    return generic[0] if generic else None


async def build_systems(
    arms: Sequence[str],
    executor: QueryExecutor,
    *,
    client: ModelClient,
    tool_client: ToolClientFactory,
    servers: Mapping[str, McpServerConfig],
    providers: Sequence[ProviderConfig],
    connector: Connector,
    sessions: list[McpSession],
    engine: Engine = DEFAULT_ENGINE,
) -> tuple[SystemUnderTest, ...]:
    """Construct each named arm, refusing names that do not exist yet.

    Every MCP session opened is appended to ``sessions`` so the caller can close
    them even when the run fails: a leaked server process poisons the next
    arm's timings.

    ``tool_client`` is a factory rather than a client because building one
    requires an API key: a run of Family A arms alone must not fail on a
    credential no arm in it uses.
    """
    grounded = {config.arm for config in providers}
    known = {*ARMS, *servers, *grounded}
    unknown = [arm for arm in arms if arm not in known]
    if unknown:
        raise CliError(f"unknown arm(s) {unknown}; available: {sorted(known)}")

    built: list[SystemUnderTest] = []
    for arm in arms:
        if arm in ARMS:
            built.append(ARMS[arm](executor, client))
            continue
        if arm in grounded:
            config = select_provider(arm, providers, engine)
            if config is None:
                raise CliError(
                    f"arm {arm} has no provider configured for {engine}; "
                    "add an entry naming that engine rather than letting it ground "
                    "against the other one"
                )
            built.append(await build_grounded(arm, config, executor, client))
            continue
        session = await connector(servers[arm])
        sessions.append(session)
        built.append(
            await McpSystem.create(
                session=session, client=tool_client(), executor=executor, config=servers[arm]
            )
        )
    return tuple(built)


async def build_grounded(
    arm: str,
    config: ProviderConfig,
    executor: QueryExecutor,
    client: ModelClient,
) -> SystemUnderTest:
    """One Family A arm, from the provider its config names.

    A plan-review arm needs a provider that can actually explain a query; asking
    for one that cannot is a configuration error, not a silently weaker arm.
    """
    provider = await load_provider(config.provider, config.options)
    if not config.plan_review:
        return GroundedSystem.create(arm=arm, provider=provider, executor=executor, client=client)
    if not isinstance(provider, PlanAdvisor):
        raise CliError(
            f"arm {arm} asks for plan review but {config.provider} cannot explain a plan; "
            "it needs an async explain_plan(sql, namespace)"
        )
    return PlanAwareSystem.create(
        arm=arm, provider=provider, advisor=provider, executor=executor, client=client
    )


def summarize(cells: Sequence[Cell]) -> tuple[str, ...]:
    """Execution accuracy per arm — the headline, printed at the end of a run."""
    lines = []
    for arm in sorted({cell.system for cell in cells}):
        graded = [cell for cell in cells if cell.system == arm]
        correct = sum(1 for cell in graded if cell.score.execution_accuracy)
        lines.append(f"{arm}: EX {correct}/{len(graded)} = {correct / len(graded):.1%}")
    return tuple(lines)


async def run_bench(
    options: BenchOptions,
    *,
    executor_factory: ExecutorFactory,
    client_factory: ClientFactory,
    write: Writer,
    tool_client_factory: ToolClientFactory | None = None,
    connector: Connector = connect,
) -> tuple[Cell, ...]:
    """Build everything the options name, run it, and report."""
    tool_client_factory = tool_client_factory or default_tool_client
    suite = load_builtin(options.suite)
    if options.limit is not None:
        suite = suite.subset(options.limit)

    executor = await executor_factory()
    sessions: list[McpSession] = []
    try:
        systems = await build_systems(
            options.arms,
            executor,
            client=client_factory(),
            tool_client=tool_client_factory,
            servers=load_server_configs(options.servers),
            providers=load_provider_catalogue(options.providers),
            connector=connector,
            sessions=sessions,
            engine=options.engine,
        )
        run_id = new_run_id()
        spec = RunSpec(
            suite=suite,
            systems=systems,
            models=tuple(parse_model(model) for model in options.models),
            run_id=run_id,
            seeds=options.seeds,
        )
        writer = TraceWriter(path=options.out / f"{run_id}.jsonl")

        write(f"{run_id}: {len(suite)} tasks x {len(systems)} arm(s) x {len(spec.models)} model(s)")
        cells = await run(spec, executor, writer=writer)
    finally:
        for session in sessions:
            await session.close()
        await _close_providers(systems)
        await executor.aclose()

    for line in summarize(cells):
        write(line)
    write(f"traces: {writer.path}")
    return cells


async def _close_providers(systems: Sequence[SystemUnderTest]) -> None:
    """Release any connection a Family A provider opened for itself.

    Structural, like everything else across this seam: a provider that holds a
    connection says so by offering ``aclose``, and one that does not is left
    alone. Without this a full matrix leaks one engine connection per arm, which
    a five-seed run turns into a warning storm and an operator into a bug report.
    """
    for system in systems:
        provider = getattr(system, "provider", None)
        closer = getattr(provider, "aclose", None)
        if closer is not None:
            await closer()


def run_report(options: ReportOptions, *, write: Writer) -> str:
    """Regenerate the report. Deterministic, offline, and cheap to re-run."""
    records = load_run(options.from_raw)
    markdown = render(records, baseline=options.baseline)
    options.out.parent.mkdir(parents=True, exist_ok=True)
    options.out.write_text(markdown, encoding="utf-8")
    write(f"{len(records)} records -> {options.out}")
    return markdown


async def run_freeze(
    options: FreezeOptions, *, executor_factory: ExecutorFactory, write: Writer
) -> Path:
    """Verify every gold query against the live data and commit the hashes."""
    suite = load_builtin(options.suite)
    executor = await executor_factory()
    try:
        hashes = await compute_gold_hashes(executor, suite)
    finally:
        await executor.aclose()
    path = write_gold_lock(
        SUITES_DIR / options.suite,
        suite.name,
        hashes,
        engine=options.engine,
        fingerprints={task.id: gold_sql_fingerprint(task.gold_sql) for task in suite},
    )
    write(f"froze {len(hashes)} gold result(s) -> {path}")
    return path


async def default_executor(engine: Engine = "clickhouse") -> QueryExecutor:
    """The engine a run measures against, reached as a read-only principal.

    Two engines, one interface: the suite, the grader and the trace format do not
    change, which is what makes the cross-engine comparison a comparison rather
    than two unrelated experiments (SPEC §18.6).
    """
    if engine == "databricks":
        dbx = DatabricksTarget.from_env()
        return DatabricksExecutor(
            client=await build_databricks_client(dbx),
            catalog=dbx.catalog,
            context_id="agenteval",
        )
    target = ClickHouseTarget.from_env()
    return ClickHouseExecutor(client=await build_client(target), context_id="agenteval")


def executor_factory_for(engine: Engine) -> ExecutorFactory:
    """The factory ``bench`` and ``freeze-gold`` use for ``engine``."""

    async def factory() -> QueryExecutor:
        return await default_executor(engine)

    return factory


def client_for(models: Sequence[str]) -> ModelClient:
    """The adapter the named models require.

    One client per run, so a run cannot silently mix a subscription-driven
    product with a bare API model and report both under one arm.
    """
    providers = {parse_model(model).provider for model in models}
    if len(providers) > 1:
        raise CliError(
            f"a run measures one model provider at a time; got {sorted(providers)}. "
            "They are not the same channel: the CLI arm carries the product's own "
            "context, the API arm does not."
        )
    provider = next(iter(providers), PROVIDER)
    if provider == CLI_PROVIDER:
        return ClaudeCliClient()
    return AnthropicClient(create=build_create())


def default_client() -> ModelClient:
    return AnthropicClient(create=build_create())


def default_tool_client() -> ToolUsingClient:
    """The same Messages endpoint, typed for the call shape that carries tools."""
    return AnthropicToolClient(create=cast(MessageCreateWithTools, build_create()))
