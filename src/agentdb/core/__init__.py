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

__all__ = [
    "ContextBuilder",
    "GroundedContext",
    "GroundingLevel",
    "RelationContext",
]
