"""The ClickHouse write channel shadow validation runs through (SPEC §9.1.B, §13.3).

Deliberately a separate object from :class:`~agentdb.adapters.clickhouse.ClickHouseAdapter`
and, in any sane deployment, a separate *connection*. The adapter is served by a
role whose profile carries ``readonly = 1``; that is the boundary that makes the
whole server safe to hand an agent, and validation cannot borrow it.

So an operator who wants measured recommendations configures a second, writable
principal and points this at it. One that does not gets estimates, which is the
correct default: nothing here runs unless ``AGENTDB_ALLOW_SHADOW`` is set as well.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from agentdb.adapters.clickhouse_sql import EXPLAIN_SETTINGS
from agentdb.adapters.errors import clickhouse_error
from agentdb.adapters.models import Engine, ExplainMode, RawPlan

LIST_TABLES = """
SELECT name
FROM system.tables
WHERE database = {namespace:String}
ORDER BY name
"""


class ShadowClient(Protocol):
    """The one driver method this channel needs."""

    async def query(
        self,
        query: str,
        *,
        parameters: Mapping[str, Any],
        settings: Mapping[str, Any],
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class ClickHouseShadowRunner:
    """Runs the DDL shadow validation needs, and nothing else it does not."""

    client: ShadowClient
    engine: Engine = "clickhouse"

    async def run(self, sql: str) -> None:
        """Execute one statement, translating a driver failure into an adapter error."""
        await self._query(sql)

    async def explain(self, sql: str, mode: ExplainMode) -> RawPlan:
        """Plan ``sql``, with the same settings the read-only adapter uses.

        The settings matter more here than anywhere else: a query-condition cache
        left on would answer the second plan read from the first, and a
        validation that measured its own cache would report every candidate as a
        triumph.
        """
        statement = (
            f"EXPLAIN indexes = 1, projections = 1, json = 1\n{sql}\nSETTINGS {EXPLAIN_SETTINGS}"
        )
        result = await self._query(statement)
        payload = "".join(str(row[0]) for row in result.result_rows)
        return RawPlan(engine=self.engine, mode=mode, sql=sql, payload=payload)

    async def list_tables(self, namespace: str) -> Sequence[str]:
        """Every table in ``namespace``, for the reaper to sift for its marker."""
        result = await self._query(LIST_TABLES, {"namespace": namespace})
        return tuple(str(row[0]) for row in result.result_rows)

    async def _query(self, sql: str, parameters: Mapping[str, Any] | None = None) -> Any:
        try:
            return await self.client.query(sql, parameters=parameters or {}, settings={})
        except Exception as exc:
            raise clickhouse_error(str(exc)) from exc
