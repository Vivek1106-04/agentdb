"""Freezing gold results into a committed lock file.

Run once against a copy of the data you trust. From then on every benchmark run
verifies the gold query still produces the same answer, and a run against
drifted data fails loudly instead of publishing a plausible number.

Freezing is deliberately a separate, explicit command. If the harness wrote
these hashes automatically, drift would silently overwrite the record of what
the numbers were computed against — which is the failure it exists to catch.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from agenteval.execution import QueryExecutor
from agenteval.gold import GoldError, resolve_gold
from agenteval.scorer import has_top_level_order_by, result_hash
from agenteval.tasks import GOLD_LOCK_NAME, TaskSuite

ENGINES = ("clickhouse", "databricks")
"""Engines a flat, engine-less hash is taken to cover when the file is upgraded."""

LOCK_HEADER = """# Gold result hashes, verified against a trusted copy of the data.
# Regenerate with: python -m agenteval freeze-gold --suite {suite}
# A mismatch at run time stops the run: the data changed, and every number
# computed since is suspect (SPEC 11.2).
"""


async def compute_gold_hashes(executor: QueryExecutor, suite: TaskSuite) -> dict[str, str]:
    """Run every gold query and hash its result.

    Existing committed hashes are still checked as each task resolves, so
    re-freezing an unchanged suite cannot quietly paper over drift — it fails.
    """
    hashes = {}
    for task in suite.for_engine(executor.engine):
        gold = await resolve_gold(executor, task)
        hashes[task.id] = result_hash(
            gold.columns, gold.rows, ordered=has_top_level_order_by(task.gold_sql)
        )
    if not hashes:
        raise GoldError(f"suite {suite.name!r} has no tasks targeting {executor.engine}")
    return hashes


def write_gold_lock(
    directory: Path, suite_name: str, hashes: dict[str, str], *, engine: str
) -> Path:
    """Merge ``hashes`` into the sidecar under ``engine``, sorted for a readable diff.

    Merged rather than overwritten: freezing gold on one engine must not erase
    what was frozen on the other. A cross-engine suite is frozen twice, once per
    engine, and both records have to survive — otherwise the second freeze
    silently disarms drift detection on the first engine.
    """
    path = directory / GOLD_LOCK_NAME
    merged = _existing(path)
    for task_id, digest in hashes.items():
        merged.setdefault(task_id, {})[engine] = digest

    body = yaml.safe_dump(
        {
            task_id: dict(sorted(per_engine.items()))
            for task_id, per_engine in sorted(merged.items())
        },
        sort_keys=True,
        default_flow_style=False,
    )
    path.write_text(LOCK_HEADER.format(suite=suite_name) + body, encoding="utf-8")
    return path


def _existing(path: Path) -> dict[str, dict[str, str]]:
    """Read the committed lock file, accepting the flat single-engine form."""
    if not path.is_file():
        return {}
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(document, dict):
        raise GoldError(f"{path} must be a mapping of task id to hash")

    existing: dict[str, dict[str, str]] = {}
    for task_id, entry in document.items():
        if isinstance(entry, dict):
            existing[str(task_id)] = {str(k): str(v) for k, v in entry.items()}
        else:
            existing[str(task_id)] = {name: str(entry) for name in ENGINES}
    return existing
