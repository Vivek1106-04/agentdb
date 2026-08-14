"""Building a live ClickHouse client without making the driver a hard dependency.

Importing agentdb, and running its unit tests, must work on a machine with no
database driver and no server. Only the server process and a benchmark run need
either, so the driver is reached through an injected importer and named in an
optional extra.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import ModuleType

from agentdb.adapters.base import EngineConnectionError
from agentdb.adapters.clickhouse import ClickHouseClient

Importer = Callable[[str], ModuleType]

DRIVER = "clickhouse_connect"

DEFAULT_USER = "agentdb_ro"
"""The read-only account from ``docker/seed/clickhouse``. Read-only is a property
of this account, not of anything agentdb does with the SQL (SPEC §13.3)."""

DEFAULT_PORT = 58123


@dataclass(frozen=True, slots=True)
class ClickHouseTarget:
    """Where to connect, and as whom."""

    host: str = "localhost"
    port: int = DEFAULT_PORT
    username: str = DEFAULT_USER
    password: str = ""
    database: str = "agentdb"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ClickHouseTarget:
        """Read ``AGENTDB_CLICKHOUSE_*``. Hosts and ports differ per machine."""
        source = os.environ if env is None else env
        port = source.get("AGENTDB_CLICKHOUSE_PORT", str(DEFAULT_PORT))
        if not port.isdigit():
            raise EngineConnectionError(f"AGENTDB_CLICKHOUSE_PORT must be a number, got {port!r}")
        return cls(
            host=source.get("AGENTDB_CLICKHOUSE_HOST", "localhost"),
            port=int(port),
            username=source.get("AGENTDB_CLICKHOUSE_USER", DEFAULT_USER),
            password=source.get("AGENTDB_CLICKHOUSE_PASSWORD", ""),
            database=source.get("AGENTDB_CLICKHOUSE_DATABASE", "agentdb"),
        )


async def build_client(
    target: ClickHouseTarget, *, importer: Importer = importlib.import_module
) -> ClickHouseClient:
    """Connect, failing with a message that says what to install or start."""
    try:
        module = importer(DRIVER)
    except ImportError as exc:
        raise EngineConnectionError(
            "the 'clickhouse-connect' driver is not installed",
            suggestion="install the optional extra with: uv sync --extra clickhouse",
        ) from exc

    try:
        client = await module.get_async_client(
            host=target.host,
            port=target.port,
            username=target.username,
            password=target.password,
            database=target.database,
        )
    except Exception as exc:
        raise EngineConnectionError(
            f"cannot reach ClickHouse at {target.host}:{target.port} as {target.username!r}: {exc}",
            suggestion="is `make up` running?",
        ) from exc

    connected: ClickHouseClient = client
    return connected
