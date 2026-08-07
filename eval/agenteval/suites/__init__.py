"""Task suites that ship with the harness (SPEC §11.2).

Suites are data. They live beside the code only so ``make bench`` works out of
the box; nothing here knows anything about the systems being measured, and no
task may be edited in response to how a system scored on it (SPEC §11.5.1).
"""

from __future__ import annotations

from pathlib import Path

from agenteval.tasks import TaskLoadError, TaskSuite, load_suite

SUITES_DIR = Path(__file__).parent


def builtin_suite_names() -> tuple[str, ...]:
    """Every suite directory that ships with agenteval, sorted."""
    return tuple(sorted(path.name for path in SUITES_DIR.iterdir() if _is_suite(path)))


def load_builtin(name: str) -> TaskSuite:
    """Load a shipped suite by directory name."""
    if name not in builtin_suite_names():
        raise TaskLoadError(
            f"unknown suite {name!r}; shipped suites are {list(builtin_suite_names())}"
        )
    return load_suite(SUITES_DIR / name)


def _is_suite(path: Path) -> bool:
    return path.is_dir() and not path.name.startswith(("_", "."))
