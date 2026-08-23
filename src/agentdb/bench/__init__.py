"""The seam where agentdb is measured by a harness that does not import it.

Nothing in agentdb depends on ``agenteval``; nothing in ``agenteval`` depends on
agentdb. They meet through a structural interface loaded by dotted path, so the
benchmark scores agentdb the same way it scores anyone else (SPEC §4.1.6).
"""

from __future__ import annotations

from agentdb.bench.advised_provider import (
    AdvisedContextProvider,
    build_advised_provider,
    clickhouse_advised_provider,
)
from agentdb.bench.memory_provider import (
    MemoryContextProvider,
    build_memory_provider,
    clickhouse_memory_provider,
)
from agentdb.bench.provider import (
    GroundedContextProvider,
    build_provider,
    clickhouse_provider,
    databricks_provider,
)

__all__ = [
    "AdvisedContextProvider",
    "GroundedContextProvider",
    "MemoryContextProvider",
    "build_advised_provider",
    "build_memory_provider",
    "build_provider",
    "clickhouse_advised_provider",
    "clickhouse_memory_provider",
    "clickhouse_provider",
    "databricks_provider",
]
