"""agentdb core: engine-neutral reasoning over what an adapter can supply.

Core never imports a concrete adapter. Everything it knows about an engine
arrives through the :class:`~agentdb.adapters.Adapter` protocol and the
capability flags beside it, which is what keeps a third engine an afternoon's
work instead of a refactor (SPEC §4.1).
"""

from __future__ import annotations

from agentdb.core.context import (
    GroundedContext,
    GroundingLevel,
    RelationContext,
)
from agentdb.core.context_builder import ContextBuilder
from agentdb.core.explain import PlanExplainer
from agentdb.core.plan_analyzer import PlanParseError
from agentdb.core.plan_analyzer_databricks import (
    PlanParseError as DatabricksPlanParseError,
)
from agentdb.core.plan_ir import (
    PlanNode,
    PlanOp,
    PlanSummary,
    PlanWarning,
    Severity,
    WarningCode,
)
from agentdb.core.plan_rules import RelationFacts
from agentdb.core.query_shape import QueryShape

__all__ = [
    "ContextBuilder",
    "DatabricksPlanParseError",
    "GroundedContext",
    "GroundingLevel",
    "PlanExplainer",
    "PlanNode",
    "PlanOp",
    "PlanParseError",
    "PlanSummary",
    "PlanWarning",
    "QueryShape",
    "RelationContext",
    "RelationFacts",
    "Severity",
    "WarningCode",
]
