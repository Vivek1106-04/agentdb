"""Reading the structural facts out of a query (SPEC §7).

Half the warnings are claims about the text, so a misparse becomes a confident
wrong warning. The rule the tests enforce: when the parser cannot read a query,
the shape says so and every text-derived rule goes quiet.
"""

from __future__ import annotations

from agentdb.core.query_shape import analyze


def test_a_grouped_aggregate_is_read_in_full() -> None:
    shape = analyze(
        "SELECT SearchEngineID, count() FROM hits "
        "WHERE UserID = 42 AND EventDate > '2013-07-01' "
        "GROUP BY SearchEngineID ORDER BY count() DESC LIMIT 10",
        "clickhouse",
    )

    assert shape.parsed is True
    assert shape.tables == ("hits",)
    assert shape.is_single_relation is True
    assert shape.filter_columns == frozenset({"UserID", "EventDate"})
    assert shape.group_by_columns == ("SearchEngineID",)
    assert shape.has_limit is True
    assert shape.has_aggregate is True
    assert shape.selects_star is False


def test_a_prewhere_filter_counts_the_same_as_a_where_filter() -> None:
    shape = analyze("SELECT URL FROM hits PREWHERE CounterID = 42", "clickhouse")

    assert shape.filter_columns == frozenset({"CounterID"})


def test_joins_are_read_left_to_right_because_the_build_side_is_the_right_one() -> None:
    shape = analyze(
        "SELECT count() FROM visits JOIN hits ON visits.UserID = hits.UserID", "clickhouse"
    )

    assert shape.tables == ("visits", "hits")
    assert shape.joined_tables == ("hits",)
    assert shape.is_single_relation is False


def test_an_ordering_expression_is_read_down_to_its_columns() -> None:
    shape = analyze("SELECT UserID FROM hits ORDER BY toDate(EventTime) DESC", "clickhouse")

    assert shape.order_by_columns == ("EventTime",)


def test_a_query_nobody_can_parse_says_so_instead_of_guessing() -> None:
    shape = analyze("SELEC coun() FRM hits WHERE", "clickhouse")

    assert shape.parsed is False
    assert shape.tables == ()
    assert shape.filter_columns == frozenset()


def test_databricks_is_parsed_in_its_own_dialect() -> None:
    shape = analyze('SELECT "user" FROM events WHERE ts > now() LIMIT 5', "databricks")

    assert shape.parsed is True
    assert shape.tables == ("events",)
    assert shape.has_limit is True
