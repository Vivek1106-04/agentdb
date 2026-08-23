"""Physical-design advice, derived from evidence rather than from a prompt (SPEC §9).

The advisor is a component *under measurement* — arm ``A6_full`` — not a headline
feature. If A6 does not beat A5, the honest outcome is to say so in the README
and consider deleting this package.
"""

from __future__ import annotations

from agentdb.core.advisor.base import (
    ColumnDemand,
    Confidence,
    Demand,
    EffectEstimate,
    Evidence,
    Kind,
    Recommendation,
    demand_from_queries,
    rank,
    workload_shapes,
)
from agentdb.core.advisor.clickhouse_advisor import ClickHouseAdvisor

__all__ = [
    "ClickHouseAdvisor",
    "ColumnDemand",
    "Confidence",
    "Demand",
    "EffectEstimate",
    "Evidence",
    "Kind",
    "Recommendation",
    "demand_from_queries",
    "rank",
    "workload_shapes",
]
