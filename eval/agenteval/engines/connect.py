"""Building a live client, without making the driver a hard dependency.

Same shape as the model adapters: the SDK is reached through an injected
importer, so the harness imports and its unit tests run on a machine with no
database driver and no server. Only ``make bench`` needs either.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType

from agenteval.engines.clickhouse import ClickHouseClient

Importer = Callable[[str], ModuleType]

DRIVER = "clickhouse_connect"

DEFAULT_USER = "agentdb_ro"
"""The read-only role from ``docker/seed/clickhouse``. Read-only is a property of
this account, not of anything the harness does with the SQL (SPEC §13.3)."""


class EngineConnectionError(RuntimeError):
    """The engine could not be reached, or the driver is missing."""


@dataclass(frozen=True, slots=True)
class ClickHouseTarget:
    """Where to connect, and as whom."""

    host: str = "localhost"
    port: int = 58123
    username: str = DEFAULT_USER
    password: str = ""
    database: str = "agentdb"

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> ClickHouseTarget:
        """Read ``AGENTEVAL_CLICKHOUSE_*``. Ports and hosts differ per machine."""
        source = os.environ if env is None else env
        port = source.get("AGENTEVAL_CLICKHOUSE_PORT", "58123")
        if not port.isdigit():
            raise EngineConnectionError(f"AGENTEVAL_CLICKHOUSE_PORT must be a number, got {port!r}")
        return cls(
            host=source.get("AGENTEVAL_CLICKHOUSE_HOST", "localhost"),
            port=int(port),
            username=source.get("AGENTEVAL_CLICKHOUSE_USER", DEFAULT_USER),
            password=source.get("AGENTEVAL_CLICKHOUSE_PASSWORD", ""),
            database=source.get("AGENTEVAL_CLICKHOUSE_DATABASE", "agentdb"),
        )


async def build_client(
    target: ClickHouseTarget, *, importer: Importer = importlib.import_module
) -> ClickHouseClient:
    """Connect, failing with a message that says what to install or start."""
    try:
        module = importer(DRIVER)
    except ImportError as exc:
        raise EngineConnectionError(
            "the 'clickhouse-connect' driver is not installed; install the "
            "optional extra with: uv sync --extra clickhouse"
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
            f"cannot reach ClickHouse at {target.host}:{target.port} as "
            f"{target.username!r}: {exc}. Is `make up` running?"
        ) from exc

    return_value: ClickHouseClient = client
    return return_value
