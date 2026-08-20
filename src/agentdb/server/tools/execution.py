"""Execution tools (SPEC §13.1, §13.3).

Read-only is a property of the connection, not of this module. ClickHouse
enforces it with a ``readonly = 1`` profile and Databricks with a Unity Catalog
principal granted ``SELECT`` and nothing else. There is deliberately no
SQL-string inspection here: string filtering is not a security boundary, and a
layer that looks like one invites exactly the false confidence that gets a
service principal over-granted.

What this tool *does* enforce is cost. Every execution carries a
:class:`~agentdb.adapters.models.Limits`, and the caller may lower a ceiling but
never raise one above the server's configuration.
"""

from __future__ import annotations

from collections.abc import Mapping

from agentdb.adapters import Limits
from agentdb.server import serialize
from agentdb.server.base import ServerContext, ToolDef, optional_int, require_str
from agentdb.server.schemas import JsonValue, definition_schema, object_schema


def execution_tools(context: ServerContext) -> tuple[ToolDef, ...]:
    """Build the execution group against ``context``."""
    return (_run_query(context),)


def _run_query(context: ServerContext) -> ToolDef:
    async def handler(args: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        sql = require_str(args, "sql")
        limits = _limits(context, args)
        result = await context.adapter.execute(sql, limits)
        return serialize.result_set(result)

    return ToolDef(
        name="run_query",
        title="Run query",
        description=(
            "Execute a read-only query under enforced limits: timeout, maximum "
            "rows returned, and maximum rows scanned. Read-only is enforced by "
            "the connection's own privileges, not by inspecting the SQL. Results "
            "report truncated=true when the row ceiling cut them, along with the "
            "rows and bytes the engine actually read — the numbers a plan can "
            "only estimate."
        ),
        input_schema=object_schema(
            {
                "sql": {
                    "type": "string",
                    "description": (
                        "The read-only query to execute. Qualify every relation "
                        "in it; the connection has one default namespace and it "
                        "may not be the one you are asking about."
                    ),
                },
                "max_rows": {
                    "type": "integer",
                    "minimum": 1,
                    "description": (
                        "Lower the row ceiling for this call. Values above the "
                        "server's configured maximum are clamped down to it."
                    ),
                },
                "timeout_s": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Lower the timeout for this call. Clamped the same way.",
                },
            },
            required=["sql"],
        ),
        output_schema=definition_schema("result_set"),
        handler=handler,
    )


def _limits(context: ServerContext, args: Mapping[str, JsonValue]) -> Limits:
    """The bounds this execution runs under: the caller's, or the server's, whichever is tighter.

    Clamping rather than rejecting is the right call for a ceiling an agent
    overshot: the agent asked for a bound and gets a bound, and the result says
    plainly when truncation happened. Silently *raising* a ceiling because the
    caller asked would make the configured limit advisory, which is not a limit.
    """
    config = context.config
    requested_rows = optional_int(args, "max_rows")
    requested_timeout = optional_int(args, "timeout_s")
    return Limits(
        timeout_s=min(requested_timeout or config.query_timeout_s, config.query_timeout_s),
        max_result_rows=min(requested_rows or config.max_result_rows, config.max_result_rows),
        max_rows_to_read=config.max_rows_to_read,
    )
