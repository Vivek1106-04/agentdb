"""The warning rules (SPEC §7).

Each test states the physical fact that makes the warning true, because that is
the claim the report will rest on. The negative tests matter as much: a rule that
fires without evidence teaches an agent to ignore every rule.
"""

from __future__ import annotations

from agentdb.adapters import ColumnProfile, PhysicalLayout, Projection, RelationRef
from agentdb.config import Config
from agentdb.core.plan_ir import PlanNode, PlanOp, PlanSummary, Severity, WarningCode
from agentdb.core.plan_rules import RelationFacts, evaluate
from agentdb.core.query_shape import UNPARSED, analyze

REF = RelationRef(namespace="agentdb", name="hits")
CONFIG = Config()

LAYOUT = PhysicalLayout(
    engine="clickhouse",
    ref=REF,
    create_statement="CREATE TABLE hits (...)",
    table_engine="MergeTree",
    order_by=("CounterID", "EventDate", "UserID"),
    partition_by=("toYYYYMM(EventDate)",),
    approx_rows=99_997_497,
)


def _summary(
    *,
    granules_total: int | None = 1_000,
    granules_selected: int | None = 10,
    relation: str | None = "hits",
    projection: str | None = None,
    sql: str = "SELECT 1",
) -> PlanSummary:
    scan = PlanNode(
        op=PlanOp.SCAN,
        node_type="ReadFromMergeTree",
        relation=relation,
        granules_total=granules_total,
        granules_selected=granules_selected,
        projection_used=projection,
    )
    ratio = (
        None
        if not granules_total or granules_selected is None
        else granules_selected / granules_total
    )
    return PlanSummary(
        root=scan,
        engine="clickhouse",
        sql=sql,
        pruning_ratio=ratio,
        pruning_unit="granule" if ratio is not None else None,
    )


def _facts(layout: PhysicalLayout = LAYOUT, **kwargs: object) -> dict[str, RelationFacts]:
    return {"hits": RelationFacts(layout=layout, **kwargs)}  # type: ignore[arg-type]


def _codes(summary: PlanSummary) -> set[WarningCode]:
    return {warning.code for warning in summary.warnings}


# --------------------------------------------------------------------------
# pruning
# --------------------------------------------------------------------------


def test_a_scan_that_pruned_nothing_on_a_large_relation_is_a_full_scan() -> None:
    sql = "SELECT count() FROM hits WHERE URL LIKE '%agentdb%'"

    result = evaluate(
        _summary(granules_selected=1_000, sql=sql), analyze(sql, "clickhouse"), _facts(), CONFIG
    )

    warning = next(w for w in result.warnings if w.code is WarningCode.FULL_SCAN)
    assert warning.relation == "hits"
    assert "100% of granules" in warning.human_message
    assert warning.suggested_rewrite is not None
    assert "CounterID" in warning.suggested_rewrite


def test_a_small_relation_does_not_earn_a_full_scan_warning() -> None:
    sql = "SELECT count() FROM hits WHERE URL LIKE '%x%'"
    small = PhysicalLayout(
        engine="clickhouse", ref=REF, create_statement="CREATE TABLE hits (...)", approx_rows=1_000
    )

    result = evaluate(
        _summary(granules_selected=1_000, sql=sql),
        analyze(sql, "clickhouse"),
        _facts(small),
        CONFIG,
    )

    assert WarningCode.FULL_SCAN not in _codes(result)


def test_a_well_pruned_scan_says_nothing_about_scanning() -> None:
    sql = "SELECT count() FROM hits WHERE CounterID = 42"

    result = evaluate(_summary(sql=sql), analyze(sql, "clickhouse"), _facts(), CONFIG)

    assert WarningCode.FULL_SCAN not in _codes(result)
    assert WarningCode.SORT_KEY_UNUSED not in _codes(result)


# --------------------------------------------------------------------------
# the sort key
# --------------------------------------------------------------------------


def test_filtering_only_on_columns_outside_the_sort_key_is_reported() -> None:
    sql = "SELECT count() FROM hits WHERE SearchEngineID = 2"

    result = evaluate(_summary(sql=sql), analyze(sql, "clickhouse"), _facts(), CONFIG)

    warning = next(w for w in result.warnings if w.code is WarningCode.SORT_KEY_UNUSED)
    assert warning.columns == ("SearchEngineID",)
    assert "CounterID, EventDate, UserID" in warning.human_message


def test_skipping_the_leading_sort_key_column_is_the_critical_case() -> None:
    sql = "SELECT count() FROM hits WHERE EventDate > '2013-07-01'"

    result = evaluate(_summary(sql=sql), analyze(sql, "clickhouse"), _facts(), CONFIG)

    warning = next(w for w in result.warnings if w.code is WarningCode.SORT_KEY_PREFIX_SKIPPED)
    assert warning.severity is Severity.CRITICAL
    assert warning.columns == ("EventDate",)
    assert WarningCode.SORT_KEY_UNUSED not in _codes(result)


def test_a_filter_inside_a_function_still_counts_as_reaching_the_key() -> None:
    sql = "SELECT count() FROM hits WHERE toStartOfMonth(EventDate) = '2013-07-01'"

    result = evaluate(_summary(sql=sql), analyze(sql, "clickhouse"), _facts(), CONFIG)

    assert WarningCode.SORT_KEY_PREFIX_SKIPPED in _codes(result)
    assert WarningCode.MISSING_PARTITION_PREDICATE not in _codes(result)


def test_a_query_with_no_filter_at_all_says_nothing_about_the_sort_key() -> None:
    sql = "SELECT count() FROM hits"

    result = evaluate(_summary(sql=sql), analyze(sql, "clickhouse"), _facts(), CONFIG)

    assert WarningCode.SORT_KEY_UNUSED not in _codes(result)
    assert WarningCode.MISSING_PARTITION_PREDICATE not in _codes(result)


def test_a_table_with_no_sort_key_earns_no_sort_key_warnings() -> None:
    sql = "SELECT count() FROM hits WHERE SearchEngineID = 2"
    keyless = PhysicalLayout(
        engine="clickhouse",
        ref=REF,
        create_statement="CREATE TABLE hits (...)",
        approx_rows=99_997_497,
    )

    result = evaluate(_summary(sql=sql), analyze(sql, "clickhouse"), _facts(keyless), CONFIG)

    assert not _codes(result) & {WarningCode.SORT_KEY_UNUSED, WarningCode.SORT_KEY_PREFIX_SKIPPED}


def test_a_full_scan_of_a_keyless_table_suggests_no_rewrite_it_cannot_justify() -> None:
    sql = "SELECT count() FROM hits WHERE URL LIKE '%x%'"
    keyless = PhysicalLayout(
        engine="clickhouse",
        ref=REF,
        create_statement="CREATE TABLE hits (...)",
        approx_rows=99_997_497,
    )

    result = evaluate(
        _summary(granules_selected=1_000, sql=sql),
        analyze(sql, "clickhouse"),
        _facts(keyless),
        CONFIG,
    )

    warning = next(w for w in result.warnings if w.code is WarningCode.FULL_SCAN)
    assert warning.suggested_rewrite is None


# --------------------------------------------------------------------------
# partitions and projections
# --------------------------------------------------------------------------


def test_a_partitioned_table_queried_without_a_partition_predicate_is_reported() -> None:
    sql = "SELECT count() FROM hits WHERE CounterID = 42"

    result = evaluate(_summary(sql=sql), analyze(sql, "clickhouse"), _facts(), CONFIG)

    warning = next(w for w in result.warnings if w.code is WarningCode.MISSING_PARTITION_PREDICATE)
    assert warning.columns == ("toYYYYMM(EventDate)",)


def test_a_projection_that_would_have_served_the_query_is_pointed_out() -> None:
    sql = "SELECT UserID, count() FROM hits WHERE CounterID = 1 GROUP BY UserID"
    layout = PhysicalLayout(
        engine="clickhouse",
        ref=REF,
        create_statement="CREATE TABLE hits (...)",
        order_by=("CounterID",),
        projections=(Projection(name="by_user", query="SELECT UserID, count() GROUP BY UserID"),),
        approx_rows=99_997_497,
    )

    result = evaluate(_summary(sql=sql), analyze(sql, "clickhouse"), _facts(layout), CONFIG)

    warning = next(w for w in result.warnings if w.code is WarningCode.PROJECTION_AVAILABLE_UNUSED)
    assert warning.severity is Severity.INFO
    assert "by_user" in warning.human_message


def test_a_projection_that_did_serve_the_query_is_not_complained_about() -> None:
    sql = "SELECT UserID, count() FROM hits GROUP BY UserID"
    layout = PhysicalLayout(
        engine="clickhouse",
        ref=REF,
        create_statement="CREATE TABLE hits (...)",
        projections=(Projection(name="by_user", query="SELECT UserID, count() GROUP BY UserID"),),
        approx_rows=99_997_497,
    )

    result = evaluate(
        _summary(sql=sql, projection="by_user"), analyze(sql, "clickhouse"), _facts(layout), CONFIG
    )

    assert WarningCode.PROJECTION_AVAILABLE_UNUSED not in _codes(result)


def test_a_projection_covering_other_columns_is_not_offered() -> None:
    sql = "SELECT RegionID, count() FROM hits GROUP BY RegionID"
    layout = PhysicalLayout(
        engine="clickhouse",
        ref=REF,
        create_statement="CREATE TABLE hits (...)",
        projections=(Projection(name="by_user", query="SELECT UserID, count() GROUP BY UserID"),),
        approx_rows=99_997_497,
    )

    result = evaluate(_summary(sql=sql), analyze(sql, "clickhouse"), _facts(layout), CONFIG)

    assert WarningCode.PROJECTION_AVAILABLE_UNUSED not in _codes(result)


# --------------------------------------------------------------------------
# what the statistics say
# --------------------------------------------------------------------------


def test_grouping_by_a_very_high_cardinality_column_is_reported() -> None:
    sql = "SELECT UserID, count() FROM hits WHERE CounterID = 1 GROUP BY UserID"
    profile = ColumnProfile(
        name="UserID",
        data_type="UInt64",
        sample_method="sample",
        sampled_rows=999_974,
        approx_distinct=17_630_976,
    )

    result = evaluate(
        _summary(sql=sql),
        analyze(sql, "clickhouse"),
        _facts(profiles={"UserID": profile}),
        CONFIG,
    )

    warning = next(w for w in result.warnings if w.code is WarningCode.HIGH_CARD_GROUP_BY)
    assert "17,630,976 groups" in warning.human_message
    assert warning.suggested_rewrite is not None


def test_grouping_by_a_nullable_column_says_where_the_nulls_go() -> None:
    sql = "SELECT SearchPhrase, count() FROM hits GROUP BY SearchPhrase"
    profile = ColumnProfile(
        name="SearchPhrase",
        data_type="Nullable(String)",
        sample_method="sample",
        sampled_rows=1_000,
        approx_distinct=500,
        null_ratio=0.3,
    )

    result = evaluate(
        _summary(sql=sql),
        analyze(sql, "clickhouse"),
        _facts(profiles={"SearchPhrase": profile}),
        CONFIG,
    )

    warning = next(w for w in result.warnings if w.code is WarningCode.NULLABLE_IN_KEY)
    assert "30% of sampled rows" in warning.human_message
    assert WarningCode.HIGH_CARD_GROUP_BY not in _codes(result)


def test_a_group_by_column_with_no_profile_produces_no_statistical_claim() -> None:
    sql = "SELECT UserID, count() FROM hits GROUP BY UserID"

    result = evaluate(_summary(sql=sql), analyze(sql, "clickhouse"), _facts(), CONFIG)

    assert not _codes(result) & {WarningCode.HIGH_CARD_GROUP_BY, WarningCode.NULLABLE_IN_KEY}


# --------------------------------------------------------------------------
# query shape
# --------------------------------------------------------------------------


def test_select_star_on_a_wide_table_is_reported_with_the_column_count() -> None:
    sql = "SELECT * FROM hits WHERE CounterID = 1 LIMIT 10"

    result = evaluate(
        _summary(sql=sql), analyze(sql, "clickhouse"), _facts(column_count=105), CONFIG
    )

    warning = next(w for w in result.warnings if w.code is WarningCode.SELECT_STAR_WIDE)
    assert "all 105 columns" in warning.human_message


def test_select_star_on_a_narrow_table_is_not_worth_saying() -> None:
    sql = "SELECT * FROM hits LIMIT 10"

    result = evaluate(_summary(sql=sql), analyze(sql, "clickhouse"), _facts(column_count=5), CONFIG)

    assert WarningCode.SELECT_STAR_WIDE not in _codes(result)


def test_an_unbounded_row_returning_query_is_reported_with_a_rough_projection() -> None:
    sql = "SELECT UserID FROM hits WHERE CounterID = 42"

    result = evaluate(
        _summary(granules_total=1_000, granules_selected=500, sql=sql),
        analyze(sql, "clickhouse"),
        _facts(),
        CONFIG,
    )

    warning = next(w for w in result.warnings if w.code is WarningCode.NO_LIMIT_UNBOUNDED)
    assert "roughly 49,998,748 rows" in warning.human_message


def test_an_aggregate_or_a_limit_makes_the_unbounded_rule_silent() -> None:
    aggregated = "SELECT count() FROM hits"
    limited = "SELECT UserID FROM hits LIMIT 10"

    for sql in (aggregated, limited):
        result = evaluate(_summary(sql=sql), analyze(sql, "clickhouse"), _facts(), CONFIG)
        assert WarningCode.NO_LIMIT_UNBOUNDED not in _codes(result)


def test_a_join_with_the_larger_relation_on_the_build_side_is_reported() -> None:
    sql = "SELECT count() FROM visits JOIN hits ON visits.UserID = hits.UserID"
    visits = PhysicalLayout(
        engine="clickhouse",
        ref=RelationRef(namespace="agentdb", name="visits"),
        create_statement="CREATE TABLE visits (...)",
        approx_rows=10_000,
    )
    facts = {"hits": RelationFacts(layout=LAYOUT), "visits": RelationFacts(layout=visits)}

    result = evaluate(_summary(sql=sql, relation=None), analyze(sql, "clickhouse"), facts, CONFIG)

    warning = next(w for w in result.warnings if w.code is WarningCode.JOIN_ORDER_SUSPECT)
    assert warning.relation == "hits"
    assert warning.suggested_rewrite == "put visits on the right of the join instead"


def test_a_join_with_the_smaller_relation_on_the_build_side_is_fine() -> None:
    sql = "SELECT count() FROM hits JOIN visits ON visits.UserID = hits.UserID"
    visits = PhysicalLayout(
        engine="clickhouse",
        ref=RelationRef(namespace="agentdb", name="visits"),
        create_statement="CREATE TABLE visits (...)",
        approx_rows=10_000,
    )
    facts = {"hits": RelationFacts(layout=LAYOUT), "visits": RelationFacts(layout=visits)}

    result = evaluate(_summary(sql=sql, relation=None), analyze(sql, "clickhouse"), facts, CONFIG)

    assert WarningCode.JOIN_ORDER_SUSPECT not in _codes(result)


def test_a_join_against_a_relation_of_unknown_size_makes_no_claim() -> None:
    sql = "SELECT count() FROM hits JOIN unknown ON hits.UserID = unknown.UserID"

    result = evaluate(
        _summary(sql=sql, relation=None), analyze(sql, "clickhouse"), _facts(), CONFIG
    )

    assert WarningCode.JOIN_ORDER_SUSPECT not in _codes(result)


# --------------------------------------------------------------------------
# missing evidence
# --------------------------------------------------------------------------


def test_a_query_nobody_could_parse_produces_no_text_derived_warnings() -> None:
    result = evaluate(_summary(granules_selected=1_000), UNPARSED, _facts(column_count=105), CONFIG)

    assert _codes(result) == {WarningCode.FULL_SCAN}


def test_a_relation_the_catalogue_does_not_know_is_skipped() -> None:
    sql = "SELECT count() FROM somewhere_else WHERE x = 1"

    result = evaluate(
        _summary(relation="somewhere_else", sql=sql), analyze(sql, "clickhouse"), {}, CONFIG
    )

    assert result.warnings == ()


def test_a_plan_scan_qualified_by_database_still_matches_the_catalogue() -> None:
    sql = "SELECT count() FROM hits WHERE SearchEngineID = 2"

    result = evaluate(
        _summary(relation="agentdb.hits", sql=sql), analyze(sql, "clickhouse"), _facts(), CONFIG
    )

    assert WarningCode.SORT_KEY_UNUSED in _codes(result)


def test_the_rendered_summary_carries_the_warnings_an_agent_should_read() -> None:
    sql = "SELECT count() FROM hits WHERE EventDate > '2013-07-01'"

    result = evaluate(_summary(sql=sql), analyze(sql, "clickhouse"), _facts(), CONFIG)
    rendered = result.render()

    assert "granules read after pruning: 1.0%" in rendered
    assert "[critical] SORT_KEY_PREFIX_SKIPPED" in rendered
    assert "try: add a predicate on CounterID" in rendered


def test_a_small_unbounded_result_is_not_worth_a_warning() -> None:
    sql = "SELECT UserID FROM hits WHERE CounterID = 42"
    small = PhysicalLayout(
        engine="clickhouse",
        ref=REF,
        create_statement="CREATE TABLE hits (...)",
        order_by=("CounterID",),
        approx_rows=1_000,
    )

    result = evaluate(_summary(sql=sql), analyze(sql, "clickhouse"), _facts(small), CONFIG)

    assert WarningCode.NO_LIMIT_UNBOUNDED not in _codes(result)


def test_a_partition_key_written_with_spaces_is_still_matched_to_its_column() -> None:
    sql = "SELECT count() FROM hits WHERE EventDate > '2013-07-01'"
    spaced = PhysicalLayout(
        engine="clickhouse",
        ref=REF,
        create_statement="CREATE TABLE hits (...)",
        order_by=("CounterID",),
        partition_by=("toYYYYMM( EventDate )",),
        approx_rows=99_997_497,
    )

    result = evaluate(_summary(sql=sql), analyze(sql, "clickhouse"), _facts(spaced), CONFIG)

    assert WarningCode.MISSING_PARTITION_PREDICATE not in _codes(result)
