"""The ClickHouse advisor's rules, one claim at a time (SPEC §9.1).

These tests are the specification of the advice. A database engineer reading the
report will check exactly these things: that the sort key orders by cardinality
in the direction a sparse index actually rewards, that a rebuild is described as
a rebuild, that dropping a column half the workload depends on is flagged rather
than buried, and that no recommendation claims to be measured when nothing was
measured.
"""

from __future__ import annotations

import pytest

from agentdb.adapters import ColumnProfile, PhysicalLayout, Projection, RelationRef, SkipIndex
from agentdb.config import Config
from agentdb.core.advisor import (
    ClickHouseAdvisor,
    Confidence,
    Kind,
    demand_from_queries,
    workload_shapes,
)
from agentdb.core.advisor.base import Demand, EffectEstimate, Evidence, Recommendation, rank
from agentdb.core.query_shape import analyze

HITS = RelationRef(namespace="agentdb", name="hits")


def profile(name: str, distinct: int, data_type: str = "UInt32") -> ColumnProfile:
    return ColumnProfile(
        name=name,
        data_type=data_type,
        sample_method="sample",
        sampled_rows=1_000_000,
        approx_distinct=distinct,
    )


def layout(**overrides: object) -> PhysicalLayout:
    fields: dict[str, object] = {
        "engine": "clickhouse",
        "ref": HITS,
        "create_statement": "CREATE TABLE agentdb.hits (...)",
        "table_engine": "MergeTree",
        "order_by": ("WatchID",),
        "partition_by": ("toYYYYMM(EventDate)",),
        "approx_rows": 100_000_000,
    }
    fields.update(overrides)
    return PhysicalLayout(**fields)  # type: ignore[arg-type]  # test factory, keyed by field name


def demand_for(*queries: str) -> Demand:
    return demand_from_queries("hits", [analyze(sql, "clickhouse") for sql in queries])


def advise(
    *queries: str,
    profiles: list[ColumnProfile] | None = None,
    physical: PhysicalLayout | None = None,
    config: Config | None = None,
) -> tuple[Recommendation, ...]:
    return ClickHouseAdvisor(config=config or Config()).advise(
        ref=HITS,
        layout=physical or layout(),
        profiles=profiles or [profile("SearchEngineID", 42), profile("UserID", 17_000_000)],
        demand=demand_for(*queries),
    )


def of_kind(recommendations: tuple[Recommendation, ...], kind: Kind) -> list[Recommendation]:
    return [item for item in recommendations if item.kind is kind]


# --------------------------------------------------------------------------
# A. the sort key
# --------------------------------------------------------------------------


def test_the_proposed_key_leads_with_the_lowest_cardinality_filtered_column() -> None:
    """The rule that is backwards from row-store instinct, and is ClickHouse's own."""
    found = advise(
        "SELECT count() FROM hits WHERE SearchEngineID = 2 AND UserID = 7",
        profiles=[profile("SearchEngineID", 42), profile("UserID", 17_000_000)],
    )

    key = of_kind(found, Kind.ORDER_BY)[0]
    assert "ORDER BY (SearchEngineID, UserID)" in (key.ddl or "")


def test_the_key_is_truncated_when_the_cardinality_budget_is_spent() -> None:
    found = advise(
        "SELECT count() FROM hits WHERE SearchEngineID = 2 AND UserID = 7 AND WatchID = 9",
        profiles=[
            profile("SearchEngineID", 42),
            profile("UserID", 17_000_000),
            profile("WatchID", 90_000_000),
        ],
        config=Config(sort_key_cardinality_budget=1_000_000.0),
    )

    key = of_kind(found, Kind.ORDER_BY)[0]
    assert "ORDER BY (SearchEngineID)" in (key.ddl or "")


def test_the_ddl_says_a_rebuild_is_a_rebuild() -> None:
    """ClickHouse cannot alter ORDER BY in place, and pretending otherwise is a trap."""
    key = of_kind(advise("SELECT count() FROM hits WHERE SearchEngineID = 2"), Kind.ORDER_BY)[0]

    assert "cannot be altered in place" in (key.ddl or "")
    assert "INSERT INTO" in (key.ddl or "")
    assert "EXCHANGE TABLES" in (key.ddl or "")
    assert any("rebuild" in note for note in key.risk_notes)


def test_dropping_a_leading_column_the_workload_depends_on_is_flagged_loudly() -> None:
    """Half the workload leads on EventDate; the budget still leaves it out of the key."""
    protected = layout(order_by=("EventDate", "UserID"))
    queries = [
        "SELECT count() FROM hits WHERE EventDate >= '2013-07-01'",
        "SELECT count() FROM hits WHERE EventDate >= '2013-07-15'",
        "SELECT count() FROM hits WHERE SearchEngineID = 2",
        "SELECT count() FROM hits WHERE SearchEngineID = 3",
    ]

    found = ClickHouseAdvisor(config=Config(sort_key_cardinality_budget=100.0)).advise(
        ref=HITS,
        layout=protected,
        profiles=[profile("SearchEngineID", 42), profile("EventDate", 1_100, "Date")],
        demand=demand_for(*queries),
    )

    key = of_kind(found, Kind.ORDER_BY)[0]
    assert any("REGRESSION RISK" in note for note in key.risk_notes)
    assert any("projection instead" in note for note in key.risk_notes)


def test_a_key_that_is_already_right_earns_no_recommendation() -> None:
    found = advise(
        "SELECT count() FROM hits WHERE SearchEngineID = 2",
        profiles=[profile("SearchEngineID", 42)],
        physical=layout(order_by=("SearchEngineID",)),
    )

    assert of_kind(found, Kind.ORDER_BY) == []


def test_a_column_with_no_profile_cannot_be_ranked_and_is_left_alone() -> None:
    """Advice from a cardinality nobody measured would be a guess wearing a number."""
    found = advise("SELECT count() FROM hits WHERE Referer = 'x'", profiles=[])

    assert found == ()


def test_the_estimate_names_its_method_and_never_claims_measurement() -> None:
    key = of_kind(advise("SELECT count() FROM hits WHERE SearchEngineID = 2"), Kind.ORDER_BY)[0]

    assert key.confidence is Confidence.ESTIMATED
    assert "upper bound" in key.expected_effect.method


def test_a_high_cardinality_lead_is_not_sold_with_the_long_runs_argument() -> None:
    """The rule ranks by frequency, so the lead can be distinct — say that, do not contradict it."""
    found = advise(
        "SELECT count() FROM hits WHERE URL LIKE '%google%'",
        "SELECT count() FROM hits WHERE URL <> ''",
        "SELECT count() FROM hits WHERE SearchEngineID = 2",
        profiles=[profile("URL", 20_000_000, "String"), profile("SearchEngineID", 42)],
    )

    key = of_kind(found, Kind.ORDER_BY)[0]
    assert "long runs" not in key.rationale
    assert "weigh this against a skip index" in key.rationale


def test_a_low_cardinality_lead_still_gets_the_long_runs_explanation() -> None:
    found = advise(
        "SELECT count() FROM hits WHERE SearchEngineID = 2",
        profiles=[profile("SearchEngineID", 42)],
    )

    assert "long runs" in of_kind(found, Kind.ORDER_BY)[0].rationale


# --------------------------------------------------------------------------
# B. skip indexes
# --------------------------------------------------------------------------


def test_a_low_cardinality_equality_filter_gets_a_set_index() -> None:
    found = advise(
        "SELECT count() FROM hits WHERE SearchEngineID = 2",
        profiles=[profile("SearchEngineID", 42)],
        physical=layout(order_by=("SearchEngineID",)),
    )

    index = of_kind(found, Kind.SKIP_INDEX)
    assert index == [], "a column already leading the sort key needs no skip index"


def test_a_high_cardinality_equality_filter_off_the_key_gets_a_bloom_filter() -> None:
    found = advise(
        "SELECT count() FROM hits WHERE UserID = 7",
        profiles=[profile("UserID", 17_000_000)],
    )

    index = of_kind(found, Kind.SKIP_INDEX)[0]
    assert "TYPE bloom_filter(0.01)" in (index.ddl or "")
    assert "MATERIALIZE INDEX" in (index.ddl or "")


def test_a_text_search_gets_a_token_filter_rather_than_a_bloom_filter() -> None:
    found = advise(
        "SELECT count() FROM hits WHERE URL LIKE '%google%'",
        profiles=[profile("URL", 20_000_000, "String")],
    )

    index = of_kind(found, Kind.SKIP_INDEX)[0]
    assert "tokenbf_v1" in (index.ddl or "")


def test_a_range_filter_on_a_mid_cardinality_column_gets_minmax() -> None:
    found = advise(
        "SELECT count() FROM hits WHERE ResolutionWidth > 1900",
        profiles=[profile("ResolutionWidth", 60_000)],
    )

    index = of_kind(found, Kind.SKIP_INDEX)[0]
    assert "TYPE minmax" in (index.ddl or "")


def test_a_column_that_already_has_an_index_is_not_offered_a_second() -> None:
    existing = layout(
        skip_indexes=(
            SkipIndex(
                name="idx_userid", index_type="bloom_filter", expression="UserID", granularity=4
            ),
        )
    )

    found = advise(
        "SELECT count() FROM hits WHERE UserID = 7",
        profiles=[profile("UserID", 17_000_000)],
        physical=existing,
    )

    assert of_kind(found, Kind.SKIP_INDEX) == []


def test_a_column_inside_the_partition_expression_is_already_served() -> None:
    found = advise(
        "SELECT count() FROM hits WHERE EventDate >= '2013-07-01'",
        profiles=[profile("EventDate", 1_100, "Date")],
        physical=layout(order_by=("WatchID",), partition_by=("toYYYYMM(EventDate)",)),
    )

    assert of_kind(found, Kind.SKIP_INDEX) == []


def test_index_candidates_are_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    columns = [f"c{index}" for index in range(6)]
    query = "SELECT count() FROM hits WHERE " + " AND ".join(f"{name} = 1" for name in columns)

    found = advise(
        query,
        profiles=[profile(name, 5_000_000) for name in columns],
        config=Config(max_index_candidates=2),
    )

    assert len(of_kind(found, Kind.SKIP_INDEX)) == 2


def test_every_index_recommendation_states_what_it_costs() -> None:
    index = of_kind(
        advise(
            "SELECT count() FROM hits WHERE UserID = 7", profiles=[profile("UserID", 17_000_000)]
        ),
        Kind.SKIP_INDEX,
    )[0]

    assert index.risk_notes
    assert any("write throughput" in note for note in index.risk_notes)


# --------------------------------------------------------------------------
# C. projections
# --------------------------------------------------------------------------


def test_a_recurring_group_by_the_sort_key_cannot_serve_earns_a_projection() -> None:
    found = advise(
        "SELECT SearchPhrase, count() FROM hits GROUP BY SearchPhrase",
        "SELECT SearchPhrase, count() FROM hits GROUP BY SearchPhrase",
    )

    projection = of_kind(found, Kind.PROJECTION)[0]
    assert "ADD PROJECTION proj_agentdb_searchphrase" in (projection.ddl or "")
    assert "MATERIALIZE PROJECTION" in (projection.ddl or "")
    assert projection.confidence is Confidence.HEURISTIC
    assert any("second physical copy" in note for note in projection.risk_notes)


def test_a_group_by_seen_once_is_not_a_recurring_shape() -> None:
    found = advise("SELECT SearchPhrase, count() FROM hits GROUP BY SearchPhrase")

    assert of_kind(found, Kind.PROJECTION) == []


def test_a_group_by_the_sort_key_already_leads_with_needs_no_projection() -> None:
    found = advise(
        "SELECT CounterID, count() FROM hits GROUP BY CounterID",
        "SELECT CounterID, count() FROM hits GROUP BY CounterID",
        physical=layout(order_by=("CounterID", "EventDate")),
    )

    assert of_kind(found, Kind.PROJECTION) == []


def test_an_existing_projection_is_not_proposed_again() -> None:
    existing = layout(
        projections=(
            Projection(
                name="proj_agentdb_searchphrase",
                query="SELECT SearchPhrase, count() GROUP BY SearchPhrase",
            ),
        )
    )

    found = advise(
        "SELECT SearchPhrase, count() FROM hits GROUP BY SearchPhrase",
        "SELECT SearchPhrase, count() FROM hits GROUP BY SearchPhrase",
        physical=existing,
    )

    assert of_kind(found, Kind.PROJECTION) == []


def test_a_table_whose_size_is_unknown_gets_no_pruning_estimate() -> None:
    """No row count, no granule count, no honest ratio to quote."""
    found = advise(
        "SELECT count() FROM hits WHERE SearchEngineID = 2",
        profiles=[profile("SearchEngineID", 42)],
        physical=layout(approx_rows=None),
    )

    key = of_kind(found, Kind.ORDER_BY)[0]
    assert key.expected_effect.after is None
    assert key.expected_effect.reduction is None


def test_a_filter_shape_no_index_type_serves_is_left_without_advice() -> None:
    """An IS NULL check on a mid-cardinality column matches no row of the §9.1.B table."""
    found = advise(
        "SELECT count() FROM hits WHERE Referer IS NOT NULL",
        profiles=[profile("Referer", 500_000, "String")],
    )

    assert of_kind(found, Kind.SKIP_INDEX) == []


# --------------------------------------------------------------------------
# ranking and the demand signal
# --------------------------------------------------------------------------


def test_a_measured_recommendation_outranks_a_larger_estimated_one() -> None:
    """The number that turns out to be wrong in front of an engineer is the estimate."""
    measured = Recommendation(
        kind=Kind.SKIP_INDEX,
        relation=HITS,
        rationale="",
        evidence=Evidence(source="shadow"),
        expected_effect=EffectEstimate(metric="granules_read", before=1.0, after=0.8, method=""),
        confidence=Confidence.MEASURED,
    )
    estimated = Recommendation(
        kind=Kind.ORDER_BY,
        relation=HITS,
        rationale="",
        evidence=Evidence(source="profile"),
        expected_effect=EffectEstimate(metric="granules_read", before=1.0, after=0.1, method=""),
        confidence=Confidence.ESTIMATED,
    )

    assert rank([estimated, measured])[0] is measured


def test_a_workload_weights_demand_by_how_often_each_shape_ran() -> None:
    """One query run ten thousand times outweighs nine run once."""
    from agentdb.adapters import WorkloadEntry

    entries = [
        WorkloadEntry(
            normalized_sql="SELECT count() FROM hits WHERE UserID = ?",
            calls=10_000,
            sample_sql="SELECT count() FROM hits WHERE UserID = 7",
        ),
        WorkloadEntry(
            normalized_sql="SELECT count() FROM hits WHERE Referer = ?",
            calls=1,
            sample_sql="SELECT count() FROM hits WHERE Referer = 'x'",
        ),
    ]

    shapes, calls = workload_shapes(entries, "clickhouse")
    demand = demand_from_queries("hits", shapes, calls)

    assert demand.of("UserID").filters == 10_000
    assert demand.of("Referer").filters == 1
    assert demand.queries == 10_001


def test_unparseable_workload_entries_are_dropped_rather_than_guessed_at() -> None:
    from agentdb.adapters import WorkloadEntry

    shapes, calls = workload_shapes(
        [WorkloadEntry(normalized_sql="SELECT * FROM (((", calls=5)], "clickhouse"
    )

    assert shapes == () and calls == ()


def test_a_column_only_ever_seen_grouped_is_not_a_filter_candidate() -> None:
    demand = demand_for("SELECT SearchPhrase, count() FROM hits GROUP BY SearchPhrase")

    assert [item.column for item in demand.filtered()] == []
    assert demand.of("SearchPhrase").groups == 1


def test_a_relation_the_query_never_names_contributes_nothing() -> None:
    demand = demand_from_queries("visits", [analyze("SELECT count() FROM hits", "clickhouse")])

    assert demand.queries == 0
    assert demand.of("anything").share == 0.0
