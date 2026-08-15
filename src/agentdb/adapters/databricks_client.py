"""Building a live Databricks client without making the SDK a hard dependency.

Importing agentdb, and running its unit tests, must work on a machine with no
warehouse and no SDK. Only the server process and a benchmark run need either, so
the SDK is reached through an injected importer and named in an optional extra —
the same shape as the ClickHouse client factory.

The transport is the **Statement Execution API** rather than the DB-API
connector, for one reason that matters to this project: it returns a
``statement_id`` on submission, which joins to ``system.query.history`` by primary
key. Attribution without string matching is the difference between an audit trail
and a hopeful ``LIKE`` (SPEC §8.2).
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import ModuleType
from typing import Any
from urllib.parse import urlsplit

from agentdb.adapters.base import EngineConnectionError
from agentdb.adapters.databricks import DatabricksClient, StatementResult

Importer = Callable[[str], ModuleType]

SDK = "databricks.sdk"
PARAMETER_MODULE = "databricks.sdk.service.sql"
"""Where ``StatementParameterListItem`` lives; imported through the same
injected importer so unit tests still need no SDK."""

MIN_WAIT_SECONDS = 5
MAX_WAIT_SECONDS = 50
"""The Statement Execution API's documented synchronous wait range."""

DEFAULT_WAIT_TIMEOUT = "50s"
"""Synchronous wait before the API hands back a pending statement id.

50s is the API's documented maximum for the synchronous path; past it the call
returns and the statement keeps running, which the adapter treats as a timeout
rather than as an empty result."""


@dataclass(frozen=True, slots=True)
class DatabricksTarget:
    """Which warehouse to run on, and as whom.

    No credential has a default. A benchmark that silently reached a workspace
    the operator did not choose would be worse than one that failed to start.
    """

    host: str
    warehouse_id: str
    token: str = ""
    client_id: str = ""
    client_secret: str = ""
    catalog: str = "samples"
    schema: str = "tpch"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> DatabricksTarget:
        """Read ``AGENTDB_DBX_*``. Credentials come from the environment only (SPEC §13.3)."""
        source = os.environ if env is None else env
        host = source.get("AGENTDB_DBX_HOST", "")
        warehouse = source.get("AGENTDB_DBX_WAREHOUSE_ID", "")
        missing = [
            name
            for name, value in (
                ("AGENTDB_DBX_HOST", host),
                ("AGENTDB_DBX_WAREHOUSE_ID", warehouse),
            )
            if not value
        ]
        if missing:
            raise EngineConnectionError(
                f"Databricks configuration is incomplete: {', '.join(missing)} is unset",
                suggestion=(
                    "export AGENTDB_DBX_HOST and AGENTDB_DBX_WAREHOUSE_ID, plus either "
                    "AGENTDB_DBX_TOKEN or an OAuth client id and secret"
                ),
            )
        return cls(
            host=normalize_host(host),
            warehouse_id=warehouse,
            token=source.get("AGENTDB_DBX_TOKEN", ""),
            client_id=source.get("AGENTDB_DBX_CLIENT_ID", ""),
            client_secret=source.get("AGENTDB_DBX_CLIENT_SECRET", ""),
            catalog=source.get("AGENTDB_DBX_CATALOG", "samples"),
            schema=source.get("AGENTDB_DBX_SCHEMA", "tpch"),
        )


def normalize_host(raw: str) -> str:
    """Reduce a pasted workspace URL to the scheme and host the SDK wants.

    Operators copy the address bar, which carries a path and the ``?o=`` account
    parameter: ``https://dbc-....cloud.databricks.com/?o=1234``. The SDK appends
    its API path to whatever it is given, so the extra parts turn every statement
    into a 404 that reads as ``NotFound: Not Found`` and looks like a missing
    table rather than a malformed host. Observed on a Free Edition workspace.
    """
    text = raw.strip().rstrip("/")
    if not text:
        return text
    parsed = urlsplit(text if "//" in text else f"https://{text}")
    return f"{parsed.scheme or 'https'}://{parsed.netloc}"


@dataclass(frozen=True, slots=True)
class ApiStatementResult:
    """One statement's response, in the shape the adapter reads."""

    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    statement_id: str | None = None
    truncated: bool = False
    rows_read: int | None = None
    bytes_read: int | None = None
    duration_ms: int | None = None


@dataclass(frozen=True, slots=True)
class StatementExecutionClient:
    """The Statement Execution API behind the adapter's client protocol.

    ``execute_statement`` is called through the SDK rather than over raw HTTP so
    that token refresh, OAuth M2M and workspace host normalization stay the SDK's
    problem. Everything this class adds is the translation into
    :class:`ApiStatementResult` — and the refusal to invent a value the response
    did not carry.
    """

    api: Any
    warehouse_id: str
    catalog: str
    schema: str
    parameter: Callable[..., Any]
    """Builds one API parameter object, by keyword: ``parameter(name=…, value=…)``.

    Deliberately injected rather than defaulted. The API takes a typed
    ``StatementParameterListItem``, not a dict, and a plain dict fails inside the
    SDK with ``'dict' object has no attribute 'as_dict'`` — observed on the first
    live run. Requiring it here means a caller cannot forget it and discover the
    problem against a real warehouse."""

    wait_timeout: str = DEFAULT_WAIT_TIMEOUT

    async def statement(
        self,
        sql: str,
        *,
        parameters: Mapping[str, Any],
        row_limit: int | None = None,
        byte_limit: int | None = None,
        timeout_s: int | None = None,
    ) -> ApiStatementResult:
        """Run one statement and return its result.

        Named parameters are passed through as the API's parameter list, so a
        value never reaches the statement text by interpolation. ``timeout_s``
        becomes the API's synchronous wait: past it the statement keeps running
        and the call returns, which the adapter surfaces rather than waiting on a
        warehouse that may be cold (SPEC §8.2 footgun 6).
        """
        response = self.api.execute_statement(
            statement=sql,
            warehouse_id=self.warehouse_id,
            catalog=self.catalog,
            schema=self.schema,
            parameters=[
                self.parameter(name=name, value=_api_value(value))
                for name, value in parameters.items()
            ],
            row_limit=row_limit,
            byte_limit=byte_limit,
            wait_timeout=self._wait_timeout(timeout_s),
        )
        return _to_result(response)

    def _wait_timeout(self, timeout_s: int | None) -> str:
        """``timeout_s`` as an API wait string, clamped to the documented range."""
        if timeout_s is None:
            return self.wait_timeout
        return f"{min(max(timeout_s, MIN_WAIT_SECONDS), MAX_WAIT_SECONDS)}s"


def _api_value(value: object) -> str:
    """The API takes parameter values as strings, typed by the statement itself."""
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _to_result(response: Any) -> ApiStatementResult:
    """Read a statement response, tolerating absent sections.

    A statement that returned no rows has no ``result`` at all, and a warehouse
    that did not report metrics reports nothing rather than zero — the adapter
    passes both through as ``None`` so that "unmeasured" never renders as "free".
    """
    manifest = getattr(response, "manifest", None)
    schema = getattr(manifest, "schema", None) if manifest is not None else None
    columns = tuple(str(column.name) for column in getattr(schema, "columns", None) or ())

    result = getattr(response, "result", None)
    rows = tuple(tuple(row) for row in (getattr(result, "data_array", None) or ()))

    status = getattr(response, "status", None)
    error = getattr(status, "error", None) if status is not None else None
    if error is not None:
        raise EngineConnectionError(str(getattr(error, "message", error)))

    return ApiStatementResult(
        columns=columns,
        rows=rows,
        statement_id=getattr(response, "statement_id", None),
        truncated=bool(getattr(manifest, "truncated", False)) if manifest is not None else False,
        rows_read=getattr(manifest, "total_row_count", None) if manifest is not None else None,
        bytes_read=getattr(manifest, "total_byte_count", None) if manifest is not None else None,
    )


async def build_client(
    target: DatabricksTarget, *, importer: Importer = importlib.import_module
) -> DatabricksClient:
    """Connect, failing with a message that says what to install or configure."""
    try:
        module = importer(SDK)
    except ImportError as exc:
        raise EngineConnectionError(
            "the 'databricks-sdk' package is not installed",
            suggestion="install the optional extra with: uv sync --extra databricks",
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
            f"cannot reach the Databricks workspace at {target.host}: {exc}",
            suggestion="check AGENTDB_DBX_HOST and the credentials in the environment",
        ) from exc

    client: DatabricksClient = StatementExecutionClient(
        api=workspace.statement_execution,
        warehouse_id=target.warehouse_id,
        catalog=target.catalog,
        schema=target.schema,
        parameter=importer(PARAMETER_MODULE).StatementParameterListItem,
    )
    return client


_RESULT_PROTOCOL_CHECK: StatementResult = ApiStatementResult(columns=(), rows=())
"""Import-time proof that the API result satisfies the protocol the adapter reads."""
