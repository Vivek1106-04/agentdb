"""Live engine clients used to run benchmark SQL.

These implement :class:`~agenteval.execution.QueryExecutor` and nothing else.
They are not a database abstraction layer — the systems under test bring their
own connections, and the harness only needs enough to show a schema, run a
query, and record what the server said.
"""

from agenteval.engines.clickhouse import (
    ClickHouseClient,
    ClickHouseExecutor,
    ClickHouseLimits,
    SchemaError,
)
from agenteval.engines.connect import (
    ClickHouseTarget,
    EngineConnectionError,
    build_client,
)
from agenteval.engines.errors import clickhouse_error_class

__all__ = [
    "ClickHouseClient",
    "ClickHouseExecutor",
    "ClickHouseLimits",
    "ClickHouseTarget",
    "EngineConnectionError",
    "SchemaError",
    "build_client",
    "clickhouse_error_class",
]
