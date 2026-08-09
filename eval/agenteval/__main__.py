"""``python -m agenteval`` — thin shell around :mod:`agenteval.cli`.

Kept separate so the CLI's behaviour is importable and testable, and only the
process plumbing (argv, stdout, exit codes) lives at the entry point.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Sequence

from agenteval.cli import CliError, default_client, default_executor, parse_args, run_options
from agenteval.engines.connect import EngineConnectionError
from agenteval.gold import GoldError
from agenteval.models.base import ModelError
from agenteval.tasks import TaskLoadError

EXIT_OK = 0
EXIT_USAGE = 2
"""Setup problems — a missing driver, a stopped engine, drifted gold — exit
distinctly from a crash, because they are things the reader can fix."""


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark, turning known setup failures into a clear line."""
    try:
        options = parse_args(sys.argv[1:] if argv is None else argv)
        asyncio.run(
            run_options(
                options,
                executor_factory=default_executor,
                client_factory=default_client,
                write=print,
            )
        )
    except (CliError, EngineConnectionError, GoldError, ModelError, TaskLoadError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
