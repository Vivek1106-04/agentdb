"""Literal-parameterized SQL, the dedup key of the exemplar store (SPEC §10.2).

The store's job is to remember *shapes*, not constants. Two questions about two
different years must collapse to one exemplar, and a top-10 must not collapse
into a top-1000 — a limit is part of the shape, not a parameter of it.
"""

from __future__ import annotations

from agentdb.core.memory import normalize_sql
from agentdb.core.memory.normalize import PLACEHOLDER


def test_two_queries_differing_only_in_their_constants_normalize_to_one_key() -> None:
    y1994 = "SELECT sum(l_extendedprice) FROM lineitem WHERE l_shipdate >= '1994-01-01'"
    y1995 = "SELECT sum(l_extendedprice) FROM lineitem WHERE l_shipdate >= '1995-06-30'"

    assert normalize_sql(y1994, "databricks") == normalize_sql(y1995, "databricks")
    assert PLACEHOLDER in normalize_sql(y1994, "databricks")


def test_a_limit_is_part_of_the_shape_and_survives_normalization() -> None:
    top_10 = normalize_sql("SELECT URL FROM hits ORDER BY EventDate LIMIT 10", "clickhouse")
    top_1000 = normalize_sql("SELECT URL FROM hits ORDER BY EventDate LIMIT 1000", "clickhouse")

    assert top_10 != top_1000
    assert top_10.endswith("LIMIT 10")


def test_an_offset_and_a_positional_group_by_are_shape_too() -> None:
    normalized = normalize_sql(
        "SELECT CounterID, count() FROM hits GROUP BY 1 ORDER BY 2 DESC LIMIT 5 OFFSET 20",
        "clickhouse",
    )

    assert "GROUP BY 1" in normalized
    assert "ORDER BY 2 DESC" in normalized
    assert "OFFSET 20" in normalized


def test_identifier_case_is_preserved_because_clickhouse_is_case_sensitive() -> None:
    normalized = normalize_sql("select CounterID from hits where URL = 'x'", "clickhouse")

    assert "CounterID" in normalized
    assert "counterid" not in normalized


def test_formatting_differences_collapse() -> None:
    spaced = normalize_sql("select   a,\n  b  from t where a = 1", "clickhouse")
    tight = normalize_sql("SELECT a, b FROM t WHERE a = 1", "clickhouse")

    assert spaced == tight


def test_the_placeholder_does_not_change_shape_between_engines() -> None:
    """sqlglot's own placeholder renders ``{?: }`` on ClickHouse and ``?`` on
    Databricks; one query would then dedup as two the moment the other engine
    was asked the same question."""
    query = "SELECT a FROM t WHERE d = 'x'"

    assert normalize_sql(query, "clickhouse") == normalize_sql(query, "databricks")
    assert normalize_sql(query, "clickhouse").endswith("= ?")


def test_unparseable_sql_still_yields_a_key() -> None:
    """The failures worth remembering are exactly the ones that may not parse."""
    assert normalize_sql("SELECT * FROM (((", "clickhouse") == "select * from ((("


def test_an_unknown_engine_falls_back_to_the_default_dialect() -> None:
    assert normalize_sql("SELECT a FROM t WHERE a = 1", "duckdb") == "SELECT a FROM t WHERE a = ?"
