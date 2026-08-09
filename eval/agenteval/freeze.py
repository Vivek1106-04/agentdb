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


def write_gold_lock(directory: Path, suite_name: str, hashes: dict[str, str]) -> Path:
    """Write the sidecar, sorted by task id so a diff is readable."""
    path = directory / GOLD_LOCK_NAME
    body = yaml.safe_dump(dict(sorted(hashes.items())), sort_keys=True, default_flow_style=False)
    path.write_text(LOCK_HEADER.format(suite=suite_name) + body, encoding="utf-8")
    return path
