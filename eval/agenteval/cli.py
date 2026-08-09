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
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from agenteval.engines.clickhouse import ClickHouseExecutor
from agenteval.engines.connect import ClickHouseTarget, build_client
from agenteval.execution import QueryExecutor
from agenteval.freeze import compute_gold_hashes, write_gold_lock
from agenteval.models.anthropic import PROVIDER, AnthropicClient, build_create
from agenteval.models.base import ModelClient
from agenteval.report import load_run, render
from agenteval.runner import Cell, RunSpec, new_run_id, run
from agenteval.suites import SUITES_DIR, load_builtin
from agenteval.systems.base import ModelSpec, SystemUnderTest
from agenteval.systems.oracle import ARM_NAME as ORACLE_ARM
from agenteval.systems.oracle import OracleSystem
from agenteval.systems.raw_schema import ARM_NAME as BASELINE_ARM
from agenteval.systems.raw_schema import RawSchemaSystem
from agenteval.traces import TraceWriter

DEFAULT_SUITE = "clickbench_nl"
DEFAULT_MODEL = f"{PROVIDER}/claude-opus-5"
DEFAULT_RAW = Path("results/raw")
DEFAULT_REPORT = Path("results/REPORT.md")

QUICK_TASKS = 5
"""``--quick`` exists so a reader can reproduce *something* in five minutes
(SPEC §14.1). It is never the number that gets published."""

ARMS: dict[str, Callable[[QueryExecutor, ModelClient], SystemUnderTest]] = {
    BASELINE_ARM: lambda executor, client: RawSchemaSystem.create(executor=executor, client=client),
    ORACLE_ARM: lambda executor, client: OracleSystem.create(executor=executor, client=client),
}
"""Arms that can be constructed today. Grows as A1-A6 and S1-S4 land."""

Writer = Callable[[str], None]
"""Where progress goes. Injected so a test reads the output instead of stdout."""

ExecutorFactory = Callable[[], Awaitable[QueryExecutor]]
ClientFactory = Callable[[], ModelClient]


class CliError(RuntimeError):
    """A run was asked for that cannot be built. Reported, never guessed around."""


@dataclass(frozen=True, slots=True)
class BenchOptions:
    """One fully-specified benchmark run."""

    suite: str = DEFAULT_SUITE
    arms: tuple[str, ...] = (BASELINE_ARM,)
    models: tuple[str, ...] = (DEFAULT_MODEL,)
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    limit: int | None = None
    out: Path = DEFAULT_RAW


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


Command = BenchOptions | ReportOptions | FreezeOptions


def parse_args(argv: Sequence[str]) -> Command:
    """Turn a command line into the options for one command."""
    parser = argparse.ArgumentParser(prog="agenteval", description="The NL-to-SQL benchmark.")
    commands = parser.add_subparsers(dest="command", required=True)

    bench = commands.add_parser("bench", help="run the matrix and write traces")
    bench.add_argument("--suite", default=DEFAULT_SUITE, help="shipped suite to run")
    bench.add_argument("--arm", action="append", dest="arms", help="repeatable; defaults to A0")
    bench.add_argument("--model", action="append", dest="models", help="provider/name, repeatable")
    bench.add_argument("--seeds", type=int, default=5, help="repetitions per cell (>=5)")
    bench.add_argument("--limit", type=int, default=None, help="first N tasks only")
    bench.add_argument("--quick", action="store_true", help=f"{QUICK_TASKS} tasks, one seed")
    bench.add_argument("--out", type=Path, default=DEFAULT_RAW, help="trace directory")

    report = commands.add_parser("report", help="regenerate REPORT.md from traces")
    report.add_argument("--from-raw", type=Path, default=DEFAULT_RAW, dest="from_raw")
    report.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    report.add_argument("--baseline", default=BASELINE_ARM, help="arm others are compared to")

    freeze = commands.add_parser("freeze-gold", help="verify and commit gold hashes")
    freeze.add_argument("--suite", default=DEFAULT_SUITE)

    args = parser.parse_args(_with_default_command(argv))

    if args.command == "report":
        return ReportOptions(from_raw=args.from_raw, out=args.out, baseline=args.baseline)
    if args.command == "freeze-gold":
        return FreezeOptions(suite=args.suite)

    if args.seeds < 1:
        raise CliError(f"--seeds must be >= 1, got {args.seeds}")
    return BenchOptions(
        suite=args.suite,
        arms=tuple(args.arms or (BASELINE_ARM,)),
        models=tuple(args.models or (DEFAULT_MODEL,)),
        seeds=(0,) if args.quick else tuple(range(args.seeds)),
        limit=QUICK_TASKS if args.quick else args.limit,
        out=args.out,
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
    if provider != PROVIDER:
        raise CliError(f"no adapter for provider {provider!r}; available: {PROVIDER}")
    return ModelSpec(provider=provider, name=name)


def build_systems(
    arms: Sequence[str], executor: QueryExecutor, client: ModelClient
) -> tuple[SystemUnderTest, ...]:
    """Construct each named arm, refusing names that do not exist yet."""
    unknown = [arm for arm in arms if arm not in ARMS]
    if unknown:
        raise CliError(f"unknown arm(s) {unknown}; available: {sorted(ARMS)}")
    return tuple(ARMS[arm](executor, client) for arm in arms)


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
) -> tuple[Cell, ...]:
    """Build everything the options name, run it, and report."""
    suite = load_builtin(options.suite)
    if options.limit is not None:
        suite = suite.subset(options.limit)

    executor = await executor_factory()
    systems = build_systems(options.arms, executor, client_factory())
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
    for line in summarize(cells):
        write(line)
    write(f"traces: {writer.path}")
    return cells


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
    hashes = await compute_gold_hashes(executor, suite)
    path = write_gold_lock(SUITES_DIR / options.suite, suite.name, hashes)
    write(f"froze {len(hashes)} gold result(s) -> {path}")
    return path


async def default_executor() -> QueryExecutor:
    """The compose ClickHouse, reached as the read-only role."""
    target = ClickHouseTarget.from_env()
    return ClickHouseExecutor(client=await build_client(target), context_id="agenteval")


def default_client() -> ModelClient:
    return AnthropicClient(create=build_create())
