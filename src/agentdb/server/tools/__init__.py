"""One module per group of the SPEC §13.1 tool catalog.

The groups are the spec's own: discovery, grounding, plan, execution, workload.
Advice and memory are absent rather than stubbed — a tool that advertises itself
and returns "not implemented" costs an agent a turn and a schema read to learn
nothing, and costs a reader of ``tools/list`` their trust in the rest.
"""

from __future__ import annotations

from agentdb.server.base import ServerContext, ToolDef
from agentdb.server.tools.discovery import discovery_tools
from agentdb.server.tools.execution import execution_tools
from agentdb.server.tools.grounding import grounding_tools
from agentdb.server.tools.plan import plan_tools
from agentdb.server.tools.workload import workload_tools

__all__ = [
    "ServerContext",
    "ToolDef",
    "all_tools",
    "discovery_tools",
    "execution_tools",
    "grounding_tools",
    "plan_tools",
    "workload_tools",
]


def all_tools(context: ServerContext) -> tuple[ToolDef, ...]:
    """Every tool this build serves, in the order SPEC §13.1 lists them."""
    return (
        *discovery_tools(context),
        *grounding_tools(context),
        *plan_tools(context),
        *execution_tools(context),
        *workload_tools(context),
    )
