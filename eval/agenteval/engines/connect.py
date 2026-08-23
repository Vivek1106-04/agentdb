"""Building a live client, without making the driver a hard dependency.

Same shape as the model adapters: the SDK is reached through an injected
importer, so the harness imports and its unit tests run on a machine with no
database driver and no server. Only ``make bench`` needs either.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import ModuleType
from typing import Any
from urllib.parse import urlsplit

from agenteval.engines.clickhouse import ClickHouseClient
from agenteval.engines.databricks import DatabricksClient

Importer = Callable[[str], ModuleType]

DRIVER = "clickhouse_connect"
DBX_SDK = "databricks.sdk"
DBX_PARAMETER_MODULE = "databricks.sdk.service.sql"
"""Where ``StatementParameterListItem`` lives; reached through the same injected
importer, so unit tests still need no SDK."""

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


DBX_WAIT_SECONDS = (5, 50)
"""The Statement Execution API's documented synchronous wait range."""


@dataclass(frozen=True, slots=True)
class DatabricksTarget:
    """Which warehouse to run on, and as whom.

    No credential has a default: a benchmark that silently reached a workspace
    the operator did not choose would be worse than one that refused to start.
    """

    host: str
    warehouse_id: str
    token: str = ""
    client_id: str = ""
    client_secret: str = ""
    catalog: str = "samples"
    schema: str = "tpch"

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> DatabricksTarget:
        """Read ``AGENTEVAL_DBX_*``. Credentials come from the environment only."""
        source = os.environ if env is None else env
        host = source.get("AGENTEVAL_DBX_HOST", "")
        warehouse = source.get("AGENTEVAL_DBX_WAREHOUSE_ID", "")
        missing = [
            name
            for name, value in (
                ("AGENTEVAL_DBX_HOST", host),
                ("AGENTEVAL_DBX_WAREHOUSE_ID", warehouse),
            )
            if not value
        ]
        if missing:
            raise EngineConnectionError(
                f"Databricks configuration is incomplete: {', '.join(missing)} is unset. "
                "Export the host and warehouse id, plus either AGENTEVAL_DBX_TOKEN or an "
                "OAuth client id and secret."
            )
        return cls(
            host=normalize_host(host),
            warehouse_id=warehouse,
            token=source.get("AGENTEVAL_DBX_TOKEN", ""),
            client_id=source.get("AGENTEVAL_DBX_CLIENT_ID", ""),
            client_secret=source.get("AGENTEVAL_DBX_CLIENT_SECRET", ""),
            catalog=source.get("AGENTEVAL_DBX_CATALOG", "samples"),
            schema=source.get("AGENTEVAL_DBX_SCHEMA", "tpch"),
        )


def normalize_host(raw: str) -> str:
    """Reduce a pasted workspace URL to the scheme and host the SDK wants.

    Operators copy the address bar, which carries a path and the ``?o=`` account
    parameter. The SDK appends its API path to whatever it is given, so those
    extra parts turn every statement into a 404 that reads as
    ``NotFound: Not Found`` — a message that looks like a missing table rather
    than a malformed host. Observed on the first live Free Edition run.
    """
    text = raw.strip().rstrip("/")
    if not text:
        return text
    parsed = urlsplit(text if "//" in text else f"https://{text}")
    return f"{parsed.scheme or 'https'}://{parsed.netloc}"


@dataclass(frozen=True, slots=True)
class StatementResponse:
    """One statement's outcome, in the shape the executor reads."""

    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    statement_id: str | None = None
    rows_read: int | None = None
    bytes_read: int | None = None


_INTEGER_TYPES = frozenset({"BYTE", "SHORT", "INT", "LONG"})
_REAL_TYPES = frozenset({"FLOAT", "DOUBLE", "DECIMAL"})


def _type_name(raw: object) -> str:
    """The column's declared type as a bare name.

    The SDK hands back a ``ColumnInfoTypeName`` enum whose ``str()`` is
    ``'ColumnInfoTypeName.LONG'``, not ``'LONG'`` — so reading it as a string
    matches nothing and every cell silently stays text, which is the failure this
    coercion exists to prevent. The enum's ``value`` is the bare name.
    """
    if raw is None:
        return ""
    return str(getattr(raw, "value", raw))


def _coerce(value: object, type_name: str) -> object:
    """Turn one API cell into the value its column says it is.

    The Statement Execution API returns **every** cell as text, including
    numbers. The grader normalizes int, float and Decimal to float but leaves a
    string a string, so without this the Databricks half of a run compares
    numbers by their spelling: `33199131663.478` and `33199131663.4780` are the
    same revenue and different strings, and the second is graded wrong.

    Typed from the response manifest rather than guessed from the text, so a
    genuinely textual column that happens to hold digits — an order id, a phone
    number — stays text and keeps comparing as one.

    A cell the declared type cannot parse is passed through untouched. A
    benchmark that crashed on one odd value would lose the whole run to it.
    """
    if value is None or not isinstance(value, str):
        return value
    name = type_name.upper()
    try:
        if name in _INTEGER_TYPES:
            return int(value)
        if name in _REAL_TYPES:
            return float(value)
        if name == "BOOLEAN":
            return value.strip().lower() == "true"
    except ValueError:
        return value
    return value


@dataclass(frozen=True, slots=True)
class StatementExecutionClient:
    """The Statement Execution API behind the executor's client protocol.

    Preferred over the DB-API connector because it returns a ``statement_id``
    synchronously: attribution by primary key rather than by string matching.
    """

    api: Any
    warehouse_id: str
    catalog: str
    schema: str
    parameter: Callable[..., Any]
    """Builds one API parameter object by keyword. Injected rather than
    defaulted: the API takes a typed ``StatementParameterListItem`` and a plain
    dict fails inside the SDK with ``'dict' object has no attribute 'as_dict'``,
    which is a failure worth catching in a unit test rather than in a bench run."""

    history: Any = None
    """The query-history API, the only source of measured pruning on Databricks.
    ``None`` on a client built for execution alone."""

    query_filter: Callable[..., Any] = dict
    """Builds the history filter by keyword, for the same reason as
    :attr:`parameter`: it is a typed SDK object, not a dict."""

    async def query_info(self, statement_id: str) -> Mapping[str, Any] | None:
        """The warehouse's record of one execution, looked up by primary key.

        Not ``system.query.history``: that table was measured 1,514 to 23,290
        seconds behind the warehouse clock on a Free Edition workspace, so a
        benchmark joining to it would attribute nothing. The history *API*
        answered by statement id immediately on every live probe.
        """
        if not statement_id or self.history is None:
            return None
        response = self.history.list(
            filter_by=self.query_filter(statement_ids=[statement_id]),
            include_metrics=True,
            max_results=1,
        )
        for entry in getattr(response, "res", None) or ():
            payload = entry.as_dict() if hasattr(entry, "as_dict") else None
            if isinstance(payload, Mapping):
                return payload
        return None

    async def statement(
        self,
        sql: str,
        *,
        parameters: Mapping[str, Any],
        row_limit: int | None = None,
        timeout_s: int | None = None,
        catalog: str | None = None,
        schema: str | None = None,
    ) -> StatementResponse:
        """Run one statement, passing values as markers rather than interpolating them.

        ``catalog`` and ``schema`` override the client's own for this statement,
        so a run can cross schemas without a second client.
        """
        low, high = DBX_WAIT_SECONDS
        response = self.api.execute_statement(
            statement=sql,
            warehouse_id=self.warehouse_id,
            catalog=catalog or self.catalog,
            schema=schema or self.schema,
            parameters=[
                self.parameter(name=name, value=_text(value)) for name, value in parameters.items()
            ],
            row_limit=row_limit,
            wait_timeout=f"{high if timeout_s is None else min(max(timeout_s, low), high)}s",
        )
        status = getattr(response, "status", None)
        error = getattr(status, "error", None) if status is not None else None
        if error is not None:
            raise EngineConnectionError(str(getattr(error, "message", error)))

        manifest = getattr(response, "manifest", None)
        schema_info = getattr(manifest, "schema", None) if manifest is not None else None
        result = getattr(response, "result", None)
        schema_columns = tuple(getattr(schema_info, "columns", None) or ())
        types = tuple(_type_name(getattr(column, "type_name", None)) for column in schema_columns)
        return StatementResponse(
            columns=tuple(str(column.name) for column in schema_columns),
            rows=tuple(
                tuple(
                    _coerce(cell, types[index] if index < len(types) else "")
                    for index, cell in enumerate(row)
                )
                for row in (getattr(result, "data_array", None) or ())
            ),
            statement_id=getattr(response, "statement_id", None),
            rows_read=getattr(manifest, "total_row_count", None) if manifest is not None else None,
            bytes_read=getattr(manifest, "total_byte_count", None)
            if manifest is not None
            else None,
        )


def _text(value: object) -> str:
    """The API takes parameter values as strings, typed by the statement itself."""
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


async def build_databricks_client(
    target: DatabricksTarget, *, importer: Importer = importlib.import_module
) -> DatabricksClient:
    """Connect, failing with a message that says what to install or configure."""
    try:
        module = importer(DBX_SDK)
    except ImportError as exc:
        raise EngineConnectionError(
            "the 'databricks-sdk' package is not installed; install the optional "
            "extra with: uv sync --extra databricks"
        ) from exc

    try:
        workspace = module.WorkspaceClient(
            host=target.host,
            token=target.token or None,
            client_id=target.client_id or None,
            client_secret=target.client_secret or None,
        )
    except Exception as exc:
        raise EngineConnectionError(
            f"cannot reach the Databricks workspace at {target.host}: {exc}"
        ) from exc

    service = importer(DBX_PARAMETER_MODULE)
    client: DatabricksClient = StatementExecutionClient(
        api=workspace.statement_execution,
        warehouse_id=target.warehouse_id,
        catalog=target.catalog,
        schema=target.schema,
        parameter=service.StatementParameterListItem,
        history=workspace.query_history,
        query_filter=service.QueryFilter,
    )
    return client
