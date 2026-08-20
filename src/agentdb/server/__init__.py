"""The MCP server surface (SPEC §13).

Two layers, deliberately separated. :mod:`agentdb.server.app` is a plain
catalog of tools over an :class:`~agentdb.adapters.Adapter` — no MCP SDK, no
transport, no protocol — and :mod:`agentdb.server.transports` is the thin
wiring that hands that catalog to the official SDK. The catalog is therefore
provable against a fake adapter with the SDK uninstalled, which is the same
rule the rest of the codebase follows for vendor dependencies.

Every tool declares a full JSON Schema 2020-12 ``outputSchema`` and every one of
them is checked against a real response by a contract test. A tool whose output
schema is decorative is worse than no schema: it tells an agent it may rely on a
shape nobody verified.
"""

from __future__ import annotations

from agentdb.server.app import ToolCatalog, ToolDef, ToolError, build_catalog
from agentdb.server.schemas import SCHEMA_DIALECT, JsonValue

__all__ = [
    "SCHEMA_DIALECT",
    "JsonValue",
    "ToolCatalog",
    "ToolDef",
    "ToolError",
    "build_catalog",
]
