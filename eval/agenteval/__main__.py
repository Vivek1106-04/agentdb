"""``python -m agenteval`` — thin shell around :mod:`agenteval.cli`.

Kept separate so the CLI's behaviour is importable and testable, and only the
process plumbing (argv, stdout, exit codes) lives at the entry point.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Sequence

from agenteval.cli import (
    BenchOptions,
    CliError,
    FreezeOptions,
    ReportOptions,
    default_client,
    default_executor,
    parse_args,
    run_bench,
    run_freeze,
    run_report,
)
from agenteval.engines.connect import EngineConnectionError
from agenteval.gold import GoldError
from agenteval.models.base import ModelError
from agenteval.report import ReportError
from agenteval.stats import StatsError
from agenteval.tasks import TaskLoadError

EXIT_OK = 0
EXIT_USAGE = 2
"""Setup problems — a missing driver, a stopped engine, drifted gold — exit
distinctly from a crash, because they are things the reader can fix."""

_KNOWN_FAILURES = (
    CliError,
    EngineConnectionError,
    GoldError,
    ModelError,
    ReportError,
    StatsError,
    TaskLoadError,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one command, turning known setup failures into a clear line."""
    try:
        _dispatch(parse_args(sys.argv[1:] if argv is None else argv))
    except _KNOWN_FAILURES as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    return EXIT_OK


def _dispatch(options: BenchOptions | ReportOptions | FreezeOptions) -> None:
    if isinstance(options, ReportOptions):
        run_report(options, write=print)
        return
    if isinstance(options, FreezeOptions):
        asyncio.run(run_freeze(options, executor_factory=default_executor, write=print))
        return
    asyncio.run(
        run_bench(
            options,
            executor_factory=default_executor,
            client_factory=default_client,
            write=print,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
