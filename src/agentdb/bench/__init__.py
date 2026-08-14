"""The seam where agentdb is measured by a harness that does not import it.

Nothing in agentdb depends on ``agenteval``; nothing in ``agenteval`` depends on
agentdb. They meet through a structural interface loaded by dotted path, so the
benchmark scores agentdb the same way it scores anyone else (SPEC §4.1.6).
"""

from __future__ import annotations

from agentdb.bench.provider import (
    GroundedContextProvider,
    build_provider,
    clickhouse_provider,
)

__all__ = [
    "GroundedContextProvider",
    "build_provider",
    "clickhouse_provider",
]
