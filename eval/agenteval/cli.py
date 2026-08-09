"""``python -m agenteval`` — the entry point behind ``make bench``.

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
from agenteval.models.anthropic import PROVIDER, AnthropicClient, build_create
from agenteval.models.base import ModelClient
from agenteval.runner import Cell, RunSpec, new_run_id, run
from agenteval.suites import load_builtin
from agenteval.systems.base import ModelSpec, SystemUnderTest
from agenteval.systems.raw_schema import ARM_NAME, RawSchemaSystem
from agenteval.traces import TraceWriter

DEFAULT_SUITE = "clickbench_nl"
DEFAULT_MODEL = f"{PROVIDER}/claude-opus-5"
DEFAULT_OUT = Path("results/raw")

QUICK_TASKS = 5
"""``--quick`` exists so a reader can reproduce *something* in five minutes
(SPEC §14.1). It is never the number that gets published."""

ARMS: dict[str, Callable[[QueryExecutor, ModelClient], SystemUnderTest]] = {
    ARM_NAME: lambda executor, client: RawSchemaSystem.create(executor=executor, client=client),
}
"""Arms that can be constructed today. Grows as A1-A7 and S1-S4 land."""


class CliError(RuntimeError):
    """A run was asked for that cannot be built. Reported, never guessed around."""


Writer = Callable[[str], None]
"""Where progress goes. Injected so a test reads the output instead of stdout."""


@dataclass(frozen=True, slots=True)
class Options:
    """One fully-specified run."""

    suite: str = DEFAULT_SUITE
    arms: tuple[str, ...] = (ARM_NAME,)
    models: tuple[str, ...] = (DEFAULT_MODEL,)
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    limit: int | None = None
    out: Path = DEFAULT_OUT


def parse_args(argv: Sequence[str]) -> Options:
    """Turn a command line into an :class:`Options`."""
    parser = argparse.ArgumentParser(prog="agenteval", description="Run the NL-to-SQL benchmark.")
    parser.add_argument("--suite", default=DEFAULT_SUITE, help="shipped suite to run")
    parser.add_argument("--arm", action="append", dest="arms", help="repeatable; defaults to A0")
    parser.add_argument("--model", action="append", dest="models", help="provider/name, repeatable")
    parser.add_argument(
        "--seeds", type=int, default=5, help="repetitions per cell (SPEC 11.4: >=5)"
    )
    parser.add_argument("--limit", type=int, default=None, help="first N tasks only")
    parser.add_argument("--quick", action="store_true", help=f"{QUICK_TASKS} tasks, one seed")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="trace directory")
    args = parser.parse_args(argv)

    if args.seeds < 1:
        raise CliError(f"--seeds must be >= 1, got {args.seeds}")

    return Options(
        suite=args.suite,
        arms=tuple(args.arms or (ARM_NAME,)),
        models=tuple(args.models or (DEFAULT_MODEL,)),
        seeds=(0,) if args.quick else tuple(range(args.seeds)),
        limit=QUICK_TASKS if args.quick else args.limit,
        out=args.out,
    )


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


async def run_options(
    options: Options,
    *,
    executor_factory: Callable[[], Awaitable[QueryExecutor]],
    client_factory: Callable[[], ModelClient],
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


async def default_executor() -> QueryExecutor:
    """The compose ClickHouse, reached as the read-only role."""
    target = ClickHouseTarget.from_env()
    return ClickHouseExecutor(client=await build_client(target), context_id="agenteval")


def default_client() -> ModelClient:
    return AnthropicClient(create=build_create())
