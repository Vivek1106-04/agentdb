"""How advice reads to the agent that has to act on it (SPEC §11.3, arm A6).

The rendering is what A6 adds over A5, so what it leaves out is under test as
much as what it says: no DDL, because an agent answering a question cannot run
ALTER TABLE, and every finding labelled with how confident it is, because an
agent that reads "may help" as "will help" writes a worse query.
"""

from __future__ import annotations

from agentdb.adapters import RelationRef
from agentdb.core.advisor.base import (
    Confidence,
    EffectEstimate,
    Evidence,
    Kind,
    Recommendation,
)
from agentdb.core.advisor.render import HEADER, render_recommendations

HITS = RelationRef(namespace="agentdb", name="hits")


def recommendation(
    *,
    kind: Kind = Kind.SKIP_INDEX,
    rationale: str = "UserID is filtered by the workload and is not in the sort key",
    confidence: Confidence = Confidence.ESTIMATED,
    before: float | None = None,
    after: float | None = None,
    rewritten_sql: str | None = None,
    ddl: str | None = None,
) -> Recommendation:
    return Recommendation(
        kind=kind,
        relation=HITS,
        rationale=rationale,
        evidence=Evidence(source="profile"),
        expected_effect=EffectEstimate(
            metric="granules_read", before=before, after=after, method="stated"
        ),
        confidence=confidence,
        ddl=ddl,
        rewritten_sql=rewritten_sql,
    )


def test_nothing_found_renders_nothing() -> None:
    assert render_recommendations([]) == ""


def test_a_design_finding_is_labelled_with_its_confidence() -> None:
    rendered = render_recommendations([recommendation(confidence=Confidence.HEURISTIC)])

    assert HEADER in rendered
    assert "- [rule of thumb] UserID is filtered" in rendered


def test_an_estimated_effect_is_quoted_as_a_share_of_the_metric() -> None:
    rendered = render_recommendations([recommendation(before=1.0, after=0.25)])

    assert "remove up to 75% of the granules read" in rendered


def test_an_arithmetically_perfect_estimate_is_not_quoted_as_a_promise() -> None:
    """1/167,955 selectivity rounds to 100%, and no index removes 100% of granules."""
    rendered = render_recommendations([recommendation(before=1.0, after=0.000006)])

    assert "100%" not in rendered
    assert "up to 95%" in rendered


def test_an_effect_nobody_could_estimate_is_simply_not_quoted() -> None:
    rendered = render_recommendations([recommendation(before=1.0, after=None)])

    assert "remove about" not in rendered


def test_a_rewrite_carries_the_corrected_sql_on_one_line() -> None:
    rendered = render_recommendations(
        [
            recommendation(
                kind=Kind.REWRITE,
                rationale="the filter wraps EventDate in a function",
                rewritten_sql="SELECT count()\n  FROM hits\n  WHERE EventDate >= '2013-01-01'",
            )
        ]
    )

    assert "Write it as: SELECT count() FROM hits WHERE EventDate >= '2013-01-01'" in rendered


def test_a_rewrite_with_no_mechanical_fix_states_the_problem_alone() -> None:
    rendered = render_recommendations(
        [recommendation(kind=Kind.REWRITE, rationale="SELECT * reads all 105 columns")]
    )

    assert "SELECT * reads all 105 columns" in rendered
    assert "Write it as" not in rendered


def test_rewrites_lead_because_they_change_what_to_write_next() -> None:
    rendered = render_recommendations(
        [
            recommendation(kind=Kind.ORDER_BY, rationale="the sort key leads with WatchID"),
            recommendation(kind=Kind.REWRITE, rationale="qualify the table name"),
        ]
    )

    assert rendered.index("qualify the table name") < rendered.index("the sort key leads")


def test_the_migration_never_reaches_the_agents_context() -> None:
    rendered = render_recommendations(
        [recommendation(ddl="ALTER TABLE agentdb.hits ADD INDEX idx_agentdb_userid ...")]
    )

    assert "ALTER TABLE" not in rendered
