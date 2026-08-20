"""Workload tools: what this engine is actually asked to do (SPEC §13.1).

The log is the only place the real query mix lives. An advisor that proposes a
sort key from the query in front of it is guessing about every other query; one
that reads ``system.query_log`` or ``system.query.history`` first is not. This
tool is the read half of that, exposed on its own because an agent investigating
a slow table wants it directly.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Final

from agentdb.adapters import Capability, TimeWindow
from agentdb.server import serialize
from agentdb.server.base import ServerContext, ToolDef, ToolError, optional_int
from agentdb.server.schemas import JsonValue, array_of, object_schema

DEFAULT_WINDOW_HOURS: Final = 24
"""A day of traffic: long enough to hold a daily batch, short enough to reflect today."""

DEFAULT_TOP_N: Final = 20
"""Shapes returned by default. Beyond a couple of dozen, the tail is noise."""

MAX_WINDOW_HOURS: Final = 24 * 30
"""A month. Past this the log has usually rotated and the answer is quietly partial."""


def workload_tools(context: ServerContext) -> tuple[ToolDef, ...]:
    """Build the workload group against ``context``."""
    return (_mine_workload(context),)


def _mine_workload(context: ServerContext) -> ToolDef:
    async def handler(args: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        if not context.adapter.supports(Capability.WORKLOAD_LOG):
            raise ToolError(
                f"the {context.adapter.engine} connection cannot read the workload log",
                suggestion="grant the connection access to the engine's query log",
            )
        hours = optional_int(args, "hours") or DEFAULT_WINDOW_HOURS
        if hours > MAX_WINDOW_HOURS:
            raise ToolError(
                f"hours must be <= {MAX_WINDOW_HOURS}",
                suggestion="the query log has usually rotated past a month",
            )
        top_n = optional_int(args, "top_n") or DEFAULT_TOP_N
        end = datetime.now(UTC)
        window = TimeWindow(start=end - timedelta(hours=hours), end=end)
        entries = await context.adapter.workload(window, top_n)
        return {
            "window_start": window.start.isoformat(),
            "window_end": window.end.isoformat(),
            "entries": [serialize.workload_entry(entry) for entry in entries],
        }

    return ToolDef(
        name="mine_workload",
        title="Mine workload",
        description=(
            "Top-N costliest query shapes from the engine's own log — "
            "system.query_log on ClickHouse, system.query.history on Databricks "
            "— normalized so that queries differing only in their literals count "
            "as one shape. Note that system.query.history lags the warehouse by "
            "minutes to hours, so it describes yesterday's workload well and the "
            "last few minutes badly."
        ),
        input_schema=object_schema(
            {
                "hours": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_WINDOW_HOURS,
                    "description": f"Window length ending now. Defaults to {DEFAULT_WINDOW_HOURS}.",
                },
                "top_n": {
                    "type": "integer",
                    "minimum": 1,
                    "description": f"Shapes to return. Defaults to {DEFAULT_TOP_N}.",
                },
            },
            required=[],
        ),
        output_schema=object_schema(
            {
                "window_start": {"type": "string", "format": "date-time"},
                "window_end": {"type": "string", "format": "date-time"},
                "entries": array_of("workload_entry", "Costliest first."),
            },
            required=["window_start", "window_end", "entries"],
        ),
        handler=handler,
    )
