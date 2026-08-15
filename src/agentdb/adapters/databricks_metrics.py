"""Reading measured metrics out of a Databricks query-history entry (SPEC §8.2).

Databricks ``EXPLAIN`` is estimate-only, and its Photon plans print no file
counts whatsoever — so unlike ClickHouse, whose plan carries granule counts
before the query runs, **the Databricks pruning evidence only exists after
execution**. This module is where it is read.

Two sources exist and only one of them is usable inside a benchmark run:

* the **Query History API**, keyed by ``statement_id``, which answered with
  ``is_final=True`` at t+0s on every probe against a Free Edition workspace;
* ``system.query.history``, the system table spelling of the same data, measured
  **1,514 to 23,290 seconds behind the warehouse clock** across two runs on the
  same workspace. Good for mining last week's workload, useless for attributing
  the statement that just ran.

Pure translation, no I/O: the transport fetches the entry, this decides what it
means.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from agentdb.adapters.models import QueryMetrics

METRICS_KEY: Final = "metrics"
"""Where the metrics live on a ``QueryInfo``. Absent on a statement the warehouse
reported no metrics for, which is not the same as one that read nothing.

The field names read below are pinned to the payload observed live on DBSQL
2026.20 — ``read_files_count``, ``pruned_files_count``, ``read_files_bytes`` — and
none of them is guessable from the ``system.query.history`` column names, which
spell the same quantities ``read_files``, ``pruned_files``, ``read_files_bytes``.
Two APIs, two vocabularies, one warehouse."""


def from_query_info(entry: Mapping[str, Any] | None) -> QueryMetrics | None:
    """One query-history entry as :class:`QueryMetrics`, or ``None``.

    ``None`` means the history had nothing to say — no entry, or an entry with no
    ``metrics`` section. A caller must not read that as "the query read nothing":
    the distinction between *unmeasured* and *measured to be zero* is the whole
    reason this returns an optional instead of a zeroed record.
    """
    if not entry:
        return None
    statement_id = entry.get("query_id") or entry.get("statement_id")
    if not statement_id:
        return None
    metrics = entry.get(METRICS_KEY)
    if not isinstance(metrics, Mapping):
        return None

    return QueryMetrics(
        statement_id=str(statement_id),
        engine="databricks",
        source="query_history_api",
        from_result_cache=_optional_bool(metrics.get("result_from_cache")),
        files_read=_optional_int(metrics.get("read_files_count")),
        files_pruned=_optional_int(metrics.get("pruned_files_count")),
        partitions_read=_optional_int(metrics.get("read_partitions_count")),
        rows_read=_optional_int(metrics.get("rows_read_count")),
        rows_produced=_optional_int(metrics.get("rows_produced_count")),
        bytes_read=_optional_int(metrics.get("read_bytes")),
        bytes_in_files_read=_optional_int(metrics.get("read_files_bytes")),
        bytes_pruned=_optional_int(metrics.get("pruned_bytes")),
        spill_bytes=_optional_int(metrics.get("spill_to_disk_bytes")),
        photon_time_ms=_optional_float(metrics.get("photon_total_time_ms")),
        execution_time_ms=_optional_float(metrics.get("execution_time_ms")),
        compilation_time_ms=_optional_float(metrics.get("compilation_time_ms")),
        total_time_ms=_optional_float(metrics.get("total_time_ms")),
    )


def is_final(entry: Mapping[str, Any] | None) -> bool:
    """Whether the warehouse considers this entry complete.

    A running statement appears in the history immediately with partial metrics,
    so a caller that polls must know when to stop. Absent, the entry is treated
    as final: an API that stops reporting the flag should not turn every lookup
    into an infinite wait.
    """
    if not entry:
        return False
    value = entry.get("is_final")
    return True if value is None else bool(value)


def _optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return int(value)
    text = str(value).strip()
    return int(text) if text.lstrip("-").isdigit() else None


def _optional_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(str(value))
    except ValueError:
        return None


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    return None
