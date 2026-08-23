"""The deterministic rewrites, and the one that deliberately refuses to rewrite.

Each of these is either exactly right or absent. The test that matters most is
the last group: a rewrite that changed what a query returns would be worse than
no advice at all, so ``SELECT *`` is costed and explained but never rewritten.
"""

from __future__ import annotations

from agentdb.adapters import (
    ColumnDef,
    DialectRules,
    PhysicalLayout,
    RelationDetail,
    RelationRef,
)
from agentdb.core.advisor.base import Confidence, Recommendation
from agentdb.core.advisor.rewrites import rewrites, rewritten_query

LINEITEM = RelationRef(catalog="samples", namespace="tpch", name="lineitem")
HITS = RelationRef(namespace="agentdb", name="hits")

DBX = DialectRules(
    engine="databricks",
    version="16.1",
    identifier_quote="`",
    reserved_words=frozenset({"order", "select"}),
)
CH = DialectRules(
    engine="clickhouse",
    version="25.9",
    identifier_quote='"',
    reserved_words=frozenset({"array"}),
)


def wide_hits(columns: int = 105) -> RelationDetail:
    return RelationDetail(
        ref=HITS,
        columns=(
            ColumnDef(name="CounterID", data_type="UInt32", is_nullable=False, compressed_bytes=10),
            ColumnDef(name="EventDate", data_type="Date", is_nullable=False, compressed_bytes=10),
            *(
                ColumnDef(
                    name=f"Filler{index}",
                    data_type="String",
                    is_nullable=False,
                    compressed_bytes=10,
                )
                for index in range(columns - 2)
            ),
        ),
        create_statement="CREATE TABLE agentdb.hits (...)",
    )


def only(found: tuple[Recommendation, ...], fragment: str) -> Recommendation:
    matching = [item for item in found if fragment in item.rationale]
    assert len(matching) == 1, [item.rationale for item in found]
    return matching[0]


# --------------------------------------------------------------------------
# qualification — a correctness fix, not a performance one
# --------------------------------------------------------------------------


def test_a_bare_table_name_is_qualified_in_full() -> None:
    found = rewrites(
        sql="SELECT count(*) FROM lineitem WHERE l_orderkey = 1", ref=LINEITEM, rules=DBX
    )

    fix = only(found, "without its catalog")
    assert fix.rewritten_sql is not None
    assert "samples.tpch.lineitem" in fix.rewritten_sql
    assert fix.confidence is Confidence.MEASURED
    assert "correctness fix" in fix.expected_effect.method


def test_an_already_qualified_query_needs_no_qualification() -> None:
    found = rewrites(sql="SELECT count(*) FROM samples.tpch.lineitem", ref=LINEITEM, rules=DBX)

    assert [item for item in found if "without its catalog" in item.rationale] == []


def test_an_engine_with_no_catalog_level_never_proposes_a_three_part_name() -> None:
    """ClickHouse namespaces are one level deep; a catalog here would be invented."""
    found = rewrites(sql="SELECT count() FROM hits", ref=HITS, rules=CH)

    assert [item for item in found if "without its catalog" in item.rationale] == []


# --------------------------------------------------------------------------
# function-wrapped temporal predicates
# --------------------------------------------------------------------------


def test_a_year_equality_becomes_a_half_open_range() -> None:
    found = rewrites(
        sql="SELECT count(*) FROM samples.tpch.lineitem WHERE year(l_shipdate) = 1995",
        ref=LINEITEM,
        rules=DBX,
    )

    fix = only(found, "wraps l_shipdate in a function")
    assert fix.rewritten_sql is not None
    assert "l_shipdate >= '1995-01-01'" in fix.rewritten_sql
    assert "l_shipdate < '1996-01-01'" in fix.rewritten_sql


def test_the_same_rule_fires_on_clickhouses_own_spelling() -> None:
    found = rewrites(
        sql="SELECT count() FROM hits WHERE toYear(EventDate) = 2013", ref=HITS, rules=CH
    )

    fix = only(found, "wraps EventDate in a function")
    assert fix.rewritten_sql is not None
    assert "EventDate >= '2013-01-01'" in fix.rewritten_sql
    assert fix.expected_effect.metric == "granules_read"


def test_wrapping_a_key_column_is_reported_as_measured_rather_than_guessed() -> None:
    """That the column carries the pruning is a fact about the layout, not a hunch."""
    layout = PhysicalLayout(
        engine="databricks",
        ref=LINEITEM,
        create_statement="",
        clustering_columns=("l_shipdate",),
    )

    found = rewrites(
        sql="SELECT count(*) FROM samples.tpch.lineitem WHERE year(l_shipdate) = 1995",
        ref=LINEITEM,
        rules=DBX,
        layout=layout,
    )

    fix = only(found, "wraps l_shipdate in a function")
    assert fix.confidence is Confidence.MEASURED
    assert "carries this table's pruning" in fix.rationale


def test_without_a_layout_the_same_rewrite_stays_heuristic() -> None:
    found = rewrites(
        sql="SELECT count(*) FROM samples.tpch.lineitem WHERE year(l_shipdate) = 1995",
        ref=LINEITEM,
        rules=DBX,
    )

    assert only(found, "wraps l_shipdate").confidence is Confidence.HEURISTIC


def test_a_function_that_is_not_a_year_extraction_is_left_alone() -> None:
    found = rewrites(
        sql="SELECT count(*) FROM samples.tpch.lineitem WHERE upper(l_returnflag) = 'R'",
        ref=LINEITEM,
        rules=DBX,
    )

    assert [item for item in found if "wraps" in item.rationale] == []


def test_a_bare_column_comparison_is_already_prunable() -> None:
    found = rewrites(
        sql="SELECT count(*) FROM samples.tpch.lineitem WHERE l_shipdate = '1995-01-01'",
        ref=LINEITEM,
        rules=DBX,
    )

    assert [item for item in found if "wraps" in item.rationale] == []


# --------------------------------------------------------------------------
# reserved words, from the engine's own list
# --------------------------------------------------------------------------


def test_a_reserved_column_name_is_reported_with_this_engines_quote() -> None:
    detail = RelationDetail(
        ref=LINEITEM,
        columns=(ColumnDef(name="order", data_type="bigint", is_nullable=False),),
        create_statement="",
    )

    found = rewrites(
        sql="SELECT count(*) FROM samples.tpch.lineitem WHERE `order` = 1",
        ref=LINEITEM,
        rules=DBX,
        detail=detail,
    )

    fix = only(found, "reserved on databricks")
    assert "`order`" in fix.rationale


def test_a_column_that_is_reserved_on_the_other_engine_only_is_left_alone() -> None:
    """Driven by dialect_rules, never by one engine's hardcoded list."""
    detail = RelationDetail(
        ref=HITS,
        columns=(ColumnDef(name="order", data_type="UInt32", is_nullable=False),),
        create_statement="",
    )

    found = rewrites(
        sql="SELECT count() FROM hits WHERE order = 1", ref=HITS, rules=CH, detail=detail
    )

    assert [item for item in found if "reserved on" in item.rationale] == []


def test_a_reserved_word_the_query_never_touches_is_not_worth_a_recommendation() -> None:
    detail = RelationDetail(
        ref=LINEITEM,
        columns=(
            ColumnDef(name="order", data_type="bigint", is_nullable=False),
            ColumnDef(name="l_orderkey", data_type="bigint", is_nullable=False),
        ),
        create_statement="",
    )

    found = rewrites(
        sql="SELECT count(*) FROM samples.tpch.lineitem WHERE l_orderkey = 1",
        ref=LINEITEM,
        rules=DBX,
        detail=detail,
    )

    assert [item for item in found if "reserved on" in item.rationale] == []


def test_without_a_schema_there_is_no_column_list_to_check() -> None:
    found = rewrites(
        sql="SELECT count(*) FROM samples.tpch.lineitem WHERE `order` = 1",
        ref=LINEITEM,
        rules=DBX,
    )

    assert [item for item in found if "reserved on" in item.rationale] == []


# --------------------------------------------------------------------------
# SELECT * — costed, explained, and deliberately not rewritten
# --------------------------------------------------------------------------


def test_select_star_on_a_wide_table_is_costed_from_the_catalogue() -> None:
    found = rewrites(
        sql="SELECT * FROM hits WHERE CounterID = 62", ref=HITS, rules=CH, detail=wide_hits()
    )

    fix = only(found, "SELECT * reads all 105 columns")
    assert fix.expected_effect.after is not None
    assert fix.expected_effect.after < 0.02, "one referenced column out of 105"
    assert fix.confidence is Confidence.ESTIMATED


def test_select_star_is_never_rewritten_automatically() -> None:
    """Only the author knows which columns the answer needs."""
    fix = only(
        rewrites(
            sql="SELECT * FROM hits WHERE CounterID = 62", ref=HITS, rules=CH, detail=wide_hits()
        ),
        "SELECT * reads all",
    )

    assert fix.rewritten_sql is None
    assert any("only the author knows" in note for note in fix.risk_notes)


def test_a_table_with_no_size_information_still_earns_the_warning_without_a_number() -> None:
    sizeless = RelationDetail(
        ref=HITS,
        columns=(ColumnDef(name="CounterID", data_type="UInt32", is_nullable=False),),
        create_statement="",
    )

    fix = only(
        rewrites(sql="SELECT * FROM hits", ref=HITS, rules=CH, detail=sizeless),
        "SELECT * reads all 1 columns",
    )

    assert fix.expected_effect.after is None
    assert fix.confidence is Confidence.HEURISTIC


def test_a_query_that_names_its_columns_needs_no_projection_advice() -> None:
    found = rewrites(sql="SELECT CounterID FROM hits", ref=HITS, rules=CH, detail=wide_hits())

    assert [item for item in found if "SELECT *" in item.rationale] == []


# --------------------------------------------------------------------------
# the composed form
# --------------------------------------------------------------------------


def test_every_mechanical_fix_composes_into_one_runnable_query() -> None:
    """An agent about to re-run wants the result, not a stack of alternatives."""
    composed = rewritten_query(
        sql="SELECT count(*) FROM lineitem WHERE year(l_shipdate) = 1995",
        ref=LINEITEM,
        rules=DBX,
    )

    assert composed is not None
    assert "samples.tpch.lineitem" in composed
    assert "l_shipdate >= '1995-01-01'" in composed


def test_the_composed_form_works_on_an_engine_with_no_catalog_level() -> None:
    composed = rewritten_query(
        sql="SELECT count() FROM hits WHERE toYear(EventDate) = 2013", ref=HITS, rules=CH
    )

    assert composed is not None
    assert "EventDate >= '2013-01-01'" in composed
    assert "agentdb.hits" not in composed, "ClickHouse names stay one level deep"


def test_other_predicates_beside_a_year_filter_are_left_exactly_as_written() -> None:
    composed = rewritten_query(
        sql=(
            "SELECT count(*) FROM samples.tpch.lineitem "
            "WHERE year(l_shipdate) = 1995 AND l_returnflag = 'R'"
        ),
        ref=LINEITEM,
        rules=DBX,
    )

    assert composed is not None
    assert "l_returnflag = 'R'" in composed


def test_an_unknown_function_wrapping_a_column_is_not_assumed_to_be_a_year() -> None:
    found = rewrites(
        sql="SELECT count(*) FROM samples.tpch.lineitem WHERE fiscal_year(l_shipdate) = 1995",
        ref=LINEITEM,
        rules=DBX,
    )

    assert [item for item in found if "wraps" in item.rationale] == []


def test_a_year_compared_to_something_other_than_a_number_is_left_alone() -> None:
    """``year(a) = year(b)`` has no range to rewrite it into."""
    found = rewrites(
        sql=(
            "SELECT count(*) FROM samples.tpch.lineitem WHERE year(l_shipdate) = year(l_commitdate)"
        ),
        ref=LINEITEM,
        rules=DBX,
    )

    assert [item for item in found if "wraps" in item.rationale] == []


def test_a_query_with_nothing_to_fix_composes_to_nothing() -> None:
    assert (
        rewritten_query(sql="SELECT count(*) FROM samples.tpch.lineitem", ref=LINEITEM, rules=DBX)
        is None
    )


def test_unparseable_sql_yields_no_rewrites_at_all() -> None:
    assert rewrites(sql="SELECT * FROM (((", ref=HITS, rules=CH) == ()
    assert rewritten_query(sql="SELECT * FROM (((", ref=HITS, rules=CH) is None
