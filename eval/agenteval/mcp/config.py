"""How a measured MCP server is specified — and committed (SPEC §11.3).

Family S is only fair if every row says exactly what was measured. A server is
therefore described by data: the command that starts it, its pinned version, and
the environment it is given. That description is fingerprinted into every trace,
so "we ran mcp-clickhouse" becomes a claim a reader can check rather than take.

Secrets never live here. The config names the environment variables a server
needs; their values are read from the process environment at launch and are
never written to a trace.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from agenteval.systems.fingerprint import config_fingerprint

REDACTED = "<from-env>"
"""What a trace records in place of a secret's value."""


class McpConfigError(ValueError):
    """A server description that cannot be trusted or launched."""


@dataclass(frozen=True, slots=True)
class McpServerConfig:
    """One measured MCP server."""

    name: str
    """The arm name this server is scored under, e.g. ``S1_mcp_clickhouse``."""

    version: str
    """Pinned, and printed in the report. A benchmark of an unnamed beta is noise."""

    command: str
    args: tuple[str, ...] = ()
    env_passthrough: tuple[str, ...] = ()
    """Names of environment variables to forward. Values are never stored."""

    env: Mapping[str, str] = field(default_factory=dict)
    """Non-secret settings passed verbatim, and recorded in the trace."""

    tools: tuple[str, ...] = ()
    """Tools the arm may call. Empty means whatever the server advertises."""

    query_tools: tuple[str, ...] = ()
    """Tools whose invocation *is* the system emitting SQL. The harness reads the
    query out of the call so a third-party server's SQL lands in the trace."""

    query_argument: str = "query"
    """Which argument of a query tool carries the SQL."""

    def __post_init__(self) -> None:
        if not self.name:
            raise McpConfigError("an MCP server config needs a name")
        if not self.version:
            raise McpConfigError(f"server {self.name!r} needs a pinned version")
        if not self.command:
            raise McpConfigError(f"server {self.name!r} needs a command to launch")

    def resolve_env(self, environ: Mapping[str, str]) -> dict[str, str]:
        """The environment to launch with, failing loudly on a missing secret."""
        missing = [name for name in self.env_passthrough if name not in environ]
        if missing:
            raise McpConfigError(
                f"server {self.name!r} needs environment variable(s) "
                f"{', '.join(missing)}; export them before running the benchmark"
            )
        return {**self.env, **{name: environ[name] for name in self.env_passthrough}}

    def as_record(self) -> dict[str, Any]:
        """The committed description. Secret *names* appear; values never do."""
        return {
            "name": self.name,
            "version": self.version,
            "command": self.command,
            "args": list(self.args),
            "env": dict(self.env),
            "env_passthrough": dict.fromkeys(self.env_passthrough, REDACTED),
            "tools": list(self.tools),
            "query_tools": list(self.query_tools),
            "query_argument": self.query_argument,
        }

    @property
    def fingerprint(self) -> str:
        return config_fingerprint(self.as_record())


def parse_server(payload: Mapping[str, Any]) -> McpServerConfig:
    """Build one config from a mapping, rejecting anything unrecognised."""
    allowed = {
        "name",
        "version",
        "command",
        "args",
        "env",
        "env_passthrough",
        "tools",
        "query_tools",
        "query_argument",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise McpConfigError(f"MCP server config has unknown field(s): {', '.join(unknown)}")

    missing = [key for key in ("name", "version", "command") if key not in payload]
    if missing:
        raise McpConfigError(f"MCP server config is missing: {', '.join(missing)}")

    return McpServerConfig(
        name=str(payload["name"]),
        version=str(payload["version"]),
        command=str(payload["command"]),
        args=_strings(payload.get("args", ())),
        env_passthrough=_strings(payload.get("env_passthrough", ())),
        env={str(k): str(v) for k, v in dict(payload.get("env", {})).items()},
        tools=_strings(payload.get("tools", ())),
        query_tools=_strings(payload.get("query_tools", ())),
        query_argument=str(payload.get("query_argument", "query")),
    )


def load_servers(path: Path) -> tuple[McpServerConfig, ...]:
    """Load a YAML list of server descriptions, e.g. ``eval/servers.yaml``."""
    if not path.is_file():
        raise McpConfigError(f"no MCP server config at {path}")

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, Sequence) or isinstance(document, str):
        raise McpConfigError(f"{path} must contain a list of server configs")

    servers = tuple(parse_server(entry) for entry in document)
    names = [server.name for server in servers]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise McpConfigError(f"{path} defines {', '.join(duplicates)} more than once")
    return servers


def _strings(raw: object) -> tuple[str, ...]:
    if isinstance(raw, str) or not isinstance(raw, Sequence):
        raise McpConfigError(f"expected a list of strings, got {raw!r}")
    return tuple(str(item) for item in raw)
