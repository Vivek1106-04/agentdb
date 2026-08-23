"""One module per group of the SPEC §13.1 tool catalog.

The groups are the spec's own: discovery, grounding, plan, advice, execution,
memory, workload. Nothing is stubbed — a tool that advertises itself and returns
"not implemented" costs an agent a turn and a schema read to learn nothing, and
costs a reader of ``tools/list`` their trust in the rest. Memory follows that
rule one level down: its tools appear only where a store is configured to answer
them. Advice does not: both engines' advisors are always listed, and the one
that does not apply to this connection says so in a structured error, because a
client should not have to ask which engine it is talking to before it knows
which tools it has.
"""

from __future__ import annotations

from agentdb.server.base import ServerContext, ToolDef
from agentdb.server.tools.advice import advice_tools
from agentdb.server.tools.discovery import discovery_tools
from agentdb.server.tools.execution import execution_tools
from agentdb.server.tools.grounding import grounding_tools
from agentdb.server.tools.memory import memory_tools
from agentdb.server.tools.plan import plan_tools
from agentdb.server.tools.workload import workload_tools

__all__ = [
    "ServerContext",
    "ToolDef",
    "advice_tools",
    "all_tools",
    "discovery_tools",
    "execution_tools",
    "grounding_tools",
    "memory_tools",
    "plan_tools",
    "workload_tools",
]


def all_tools(context: ServerContext) -> tuple[ToolDef, ...]:
    """Every tool this build serves, in the order SPEC §13.1 lists them."""
    return (
        *discovery_tools(context),
        *grounding_tools(context),
        *plan_tools(context),
        *advice_tools(context),
        *execution_tools(context),
        *memory_tools(context),
        *workload_tools(context),
    )
