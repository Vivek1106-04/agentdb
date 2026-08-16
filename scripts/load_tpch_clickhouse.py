"""Load TPC-H into ClickHouse at the scale factor Databricks `samples.tpch` ships.

SPEC §11.2 requires the cross-engine suite to run on both engines over *the same
data*: "matching the scale factor is not optional; a cross-engine accuracy
comparison at different scale factors is not a comparison." Databricks ships
`samples.tpch` pre-loaded and unchangeable, so it is the fixed side and
ClickHouse is loaded to match it.

The generator is DuckDB's `tpch` extension, which is a port of the reference
`dbgen`. That is not a convenience choice — it is the reason the comparison is
sound. Verified on 2026-08-16, `dbgen(sf=5)` reproduces `samples.tpch` row for
row across all eight tables, lineitem included at 29,999,795 rows. Anything that
merely *approximated* the data would leave every cross-engine delta
indistinguishable from a data difference.

Usage::

    make load-tpch                 # SF5, matching samples.tpch
    make load-tpch TPCH_SCALE=1    # smaller, for a laptop that cannot spare 6GB

Re-running is safe: a table already holding the expected row count is skipped,
and any other count is truncated and reloaded rather than appended to.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

TABLES: tuple[str, ...] = (
    "region",
    "nation",
    "supplier",
    "customer",
    "part",
    "partsupp",
    "orders",
    "lineitem",
)
"""Smallest first, so a misconfiguration fails on 5 rows rather than on 30M."""

SF5_ROWS: dict[str, int] = {
    "region": 5,
    "nation": 25,
    "supplier": 50_000,
    "customer": 750_000,
    "part": 1_000_000,
    "partsupp": 4_000_000,
    "orders": 7_500_000,
    "lineitem": 29_999_795,
}
"""Counted on Databricks `samples.tpch` on 2026-08-16, not taken from the TPC-H
specification. At SF5 this is the cross-engine contract: if the generator ever
stops reproducing these numbers, the suite is comparing two different datasets
and must fail rather than report."""

DEFAULT_SCALE = 5
ROW_GROUP_SIZE = 1_000_000
"""Parquet row-group size. Bounds ClickHouse's insert-side memory on lineitem;
the default would hand it the whole file as one group."""

HTTP_TIMEOUT_S = 3600
"""Loading 30M rows over one request legitimately takes minutes."""


@dataclass(frozen=True, slots=True)
class ClickHouse:
    """Where to load, and as whom.

    The seeding account is the read-write one. The harness's `agentdb_ro` cannot
    write by construction (SPEC §13.3), which is the point of it.
    """

    host: str
    port: int
    user: str
    password: str
    database: str

    @classmethod
    def from_env(cls) -> ClickHouse:
        return cls(
            host=os.environ.get("AGENTEVAL_CLICKHOUSE_HOST", "localhost"),
            port=int(os.environ.get("AGENTEVAL_CLICKHOUSE_PORT", "58123")),
            user=os.environ.get("AGENTDB_SEED_USER", "agentdb"),
            password=os.environ.get("AGENTDB_SEED_PASSWORD", "agentdb"),
            database=os.environ.get("AGENTDB_TPCH_DATABASE", "tpch"),
        )

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def _request(self, query: str, body: object = None) -> urllib.request.Request:
        request = urllib.request.Request(
            self.url + "?" + urllib.parse.urlencode({"query": query}),
            data=body,  # type: ignore[arg-type]
            method="POST",
        )
        request.add_header("X-ClickHouse-User", self.user)
        request.add_header("X-ClickHouse-Key", self.password)
        return request

    def command(self, query: str) -> str:
        """Run a statement and return its response body as text."""
        try:
            with urllib.request.urlopen(self._request(query), timeout=HTTP_TIMEOUT_S) as response:
                return str(response.read().decode("utf-8")).strip()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace").strip()
            raise SystemExit(f"ClickHouse rejected `{query[:60]}...`:\n{detail}") from exc
        except urllib.error.URLError as exc:
            raise SystemExit(
                f"cannot reach ClickHouse at {self.url}: {exc.reason}\nstart it with: make up"
            ) from exc

    def count(self, table: str) -> int:
        return int(self.command(f"SELECT count() FROM {self.database}.{table}"))

    def insert_parquet(self, table: str, path: Path) -> None:
        """Stream one Parquet file into ``table``.

        The file object is handed to urllib rather than read into memory:
        http.client sends it in blocks, so lineitem does not have to fit in RAM.
        """
        query = f"INSERT INTO {self.database}.{table} FORMAT Parquet"
        with path.open("rb") as body:
            request = self._request(query, body)
            request.add_header("Content-Length", str(path.stat().st_size))
            try:
                with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S):
                    return
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace").strip()
                raise SystemExit(f"loading {table} failed:\n{detail}") from exc


def expected_rows(scale: int, generated: dict[str, int]) -> dict[str, int]:
    """The row counts to hold the load to.

    At SF5 these are the counts measured on `samples.tpch`, and disagreeing with
    them is fatal: it means the local generator no longer reproduces the data the
    Databricks half runs on, and the cross-engine comparison silently stops being
    one. At any other scale there is nothing to cross-check against, so the
    generator's own counts stand and the caller has already been told the suite
    is not cross-engine at that scale.
    """
    if scale != DEFAULT_SCALE:
        return generated
    mismatched = {
        table: (generated[table], SF5_ROWS[table])
        for table in TABLES
        if generated[table] != SF5_ROWS[table]
    }
    if mismatched:
        lines = "\n".join(
            f"  {table}: generator produced {got:,}, samples.tpch holds {want:,}"
            for table, (got, want) in sorted(mismatched.items())
        )
        raise SystemExit(
            "the TPC-H generator no longer reproduces Databricks samples.tpch:\n"
            f"{lines}\n"
            "The cross-engine suite requires identical data on both engines "
            "(SPEC 11.2). Refusing to load."
        )
    return SF5_ROWS


def generate(workdir: Path, scale: int) -> dict[str, int]:
    """Generate TPC-H at ``scale`` and write one Parquet file per table."""
    try:
        import duckdb
    except ImportError:
        raise SystemExit(
            "duckdb is not installed; it is the TPC-H generator.\n"
            "install the seeding extra with: uv sync --extra seed"
        ) from None

    connection = duckdb.connect(str(workdir / "tpch.duckdb"))
    connection.execute("INSTALL tpch; LOAD tpch;")
    print(f"generating TPC-H at scale factor {scale}...", flush=True)
    started = time.monotonic()
    connection.execute(f"CALL dbgen(sf={scale})")
    print(f"  generated in {time.monotonic() - started:.0f}s", flush=True)

    counts: dict[str, int] = {}
    for table in TABLES:
        target = workdir / f"{table}.parquet"
        connection.execute(
            f"COPY {table} TO '{target}' (FORMAT PARQUET, ROW_GROUP_SIZE {ROW_GROUP_SIZE})"
        )
        counted = connection.execute(f"SELECT count(*) FROM {table}").fetchone()
        if counted is None:
            raise SystemExit(f"the generator produced no count for {table}")
        counts[table] = int(counted[0])
    connection.close()
    return counts


def load(clickhouse: ClickHouse, workdir: Path, expected: dict[str, int]) -> None:
    """Load each table, skipping any that already holds the expected rows."""
    for table in TABLES:
        want = expected[table]
        if clickhouse.count(table) == want:
            print(f"  {table}: already holds {want:,} rows, skipping", flush=True)
            continue

        clickhouse.command(f"TRUNCATE TABLE {clickhouse.database}.{table}")
        started = time.monotonic()
        clickhouse.insert_parquet(table, workdir / f"{table}.parquet")
        loaded = clickhouse.count(table)
        if loaded != want:
            raise SystemExit(
                f"{table}: loaded {loaded:,} rows but expected {want:,}. "
                "The table is left as loaded so the difference can be inspected."
            )
        print(f"  {table}: {loaded:,} rows in {time.monotonic() - started:.0f}s", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scale",
        type=int,
        default=int(os.environ.get("TPCH_SCALE", DEFAULT_SCALE)),
        help=f"TPC-H scale factor (default {DEFAULT_SCALE}, which matches samples.tpch)",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="where to generate; a temporary directory is used and removed by default",
    )
    arguments = parser.parse_args(argv)

    if arguments.scale != DEFAULT_SCALE:
        print(
            f"WARNING: scale factor {arguments.scale} does not match Databricks "
            f"samples.tpch (SF{DEFAULT_SCALE}). tpch_nl results from this data are "
            "NOT cross-engine comparable.",
            file=sys.stderr,
            flush=True,
        )

    clickhouse = ClickHouse.from_env()
    clickhouse.command("SELECT 1")

    owned = arguments.workdir is None
    workdir = Path(tempfile.mkdtemp(prefix="agentdb-tpch-")) if owned else arguments.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        expected = expected_rows(arguments.scale, generate(workdir, arguments.scale))
        print(f"loading into {clickhouse.url}{clickhouse.database}...", flush=True)
        load(clickhouse, workdir, expected)
    finally:
        if owned:
            shutil.rmtree(workdir, ignore_errors=True)

    print(f"TPC-H SF{arguments.scale} loaded into ClickHouse database {clickhouse.database!r}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
