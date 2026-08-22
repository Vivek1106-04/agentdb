"""The hybrid ranking, term by term (SPEC §10.4).

Each test zeroes every weight but one, which is the same operation the ablation
arms perform. If a term cannot be isolated here, its row in the published
weight-ablation table means nothing.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from agentdb.config import RetrievalWeights
from agentdb.core.memory import Exemplar, Outcome, Provenance, rank
from agentdb.core.memory.ranking import cosine, jaccard, recency

NOW = datetime(2026, 8, 22, 12, 0, tzinfo=UTC)
TAU = 30.0


def exemplar(
    id: int,  # noqa: A002 — mirrors the column name
    *,
    relations: tuple[str, ...] = ("hits",),
    outcome: Outcome = Outcome.SUCCESS,
    embedding: tuple[float, ...] = (),
    bytes_read: int | None = None,
    learned_days_ago: float = 0.0,
) -> Exemplar:
    return Exemplar(
        id=id,
        engine="clickhouse",
        namespace="agentdb",
        question=f"question {id}",
        sql="SELECT 1",
        normalized_sql="SELECT ?",
        relations=relations,
        columns=("CounterID",),
        schema_version_id=1,
        outcome=outcome,
        provenance=Provenance.AGENT,
        valid_from=NOW,
        tx_from=NOW - timedelta(days=learned_days_ago),
        embedding=embedding,
        bytes_read=bytes_read,
        error_class=None if outcome is Outcome.SUCCESS else "semantic",
    )


def only(term: str) -> RetrievalWeights:
    """Weights with a single term active — one ablation arm."""
    return RetrievalWeights(
        **{name: (1.0 if name == term else 0.0) for name in RetrievalWeights().as_mapping()}
    )


def ranked(
    candidates: list[Exemplar],
    *,
    weights: RetrievalWeights,
    question_embedding: tuple[float, ...] = (),
    relations: tuple[str, ...] = ("hits",),
    limit: int | None = None,
) -> tuple[int, ...]:
    scored = rank(
        candidates,
        question_embedding=question_embedding,
        relations=relations,
        now=NOW,
        weights=weights,
        tau_days=TAU,
        limit=limit,
    )
    return tuple(item.exemplar.id for item in scored)


# --------------------------------------------------------------------------
# one term at a time
# --------------------------------------------------------------------------


def test_the_semantic_term_orders_by_cosine_against_the_question() -> None:
    order = ranked(
        [exemplar(1, embedding=(0.0, 1.0)), exemplar(2, embedding=(1.0, 0.0))],
        weights=only("sem"),
        question_embedding=(1.0, 0.0),
    )

    assert order == (2, 1)


def test_the_relation_term_orders_by_overlap_with_the_relations_asked_about() -> None:
    order = ranked(
        [exemplar(1, relations=("visits",)), exemplar(2, relations=("hits",))],
        weights=only("rel"),
    )

    assert order == (2, 1)


def test_the_success_term_puts_working_queries_above_failed_ones() -> None:
    order = ranked(
        [exemplar(1, outcome=Outcome.ERROR), exemplar(2)],
        weights=only("success"),
    )

    assert order == (2, 1)


def test_the_recency_term_prefers_what_was_learned_most_recently() -> None:
    order = ranked(
        [exemplar(1, learned_days_ago=90.0), exemplar(2, learned_days_ago=1.0)],
        weights=only("recency"),
    )

    assert order == (2, 1)


def test_the_cost_term_penalizes_the_most_expensive_candidate_in_the_pool() -> None:
    order = ranked(
        [exemplar(1, bytes_read=10_000_000_000), exemplar(2, bytes_read=1_000)],
        weights=only("cost"),
    )

    assert order == (2, 1)


def test_an_unmeasured_query_is_not_treated_as_an_expensive_one() -> None:
    scored = rank(
        [exemplar(1, bytes_read=None), exemplar(2, bytes_read=10_000_000_000)],
        question_embedding=(),
        relations=("hits",),
        now=NOW,
        weights=only("cost"),
        tau_days=TAU,
    )

    assert scored[0].exemplar.id == 1
    assert scored[0].components["cost"] == 0.0


# --------------------------------------------------------------------------
# the contract the harness depends on
# --------------------------------------------------------------------------


def test_every_component_is_reported_beside_the_total() -> None:
    scored = rank(
        [exemplar(1, embedding=(1.0, 0.0), bytes_read=5)],
        question_embedding=(1.0, 0.0),
        relations=("hits",),
        now=NOW,
        weights=RetrievalWeights(),
        tau_days=TAU,
    )

    components = scored[0].components
    assert set(components) == {"sem", "rel", "success", "recency", "cost"}
    expected = 0.40 * 1.0 + 0.30 * 1.0 + 0.15 * 1.0 + 0.10 * components["recency"] - 0.05 * 1.0
    assert scored[0].score == expected


def test_ties_break_deterministically_on_the_newest_exemplar() -> None:
    order = ranked([exemplar(3), exemplar(7), exemplar(5)], weights=only("success"))

    assert order == (7, 5, 3)


def test_the_limit_truncates_after_ranking_not_before() -> None:
    order = ranked(
        [exemplar(1, relations=("visits",)), exemplar(2, relations=("hits",))],
        weights=only("rel"),
        limit=1,
    )

    assert order == (2,)


def test_an_empty_pool_ranks_to_nothing() -> None:
    assert ranked([], weights=RetrievalWeights()) == ()


# --------------------------------------------------------------------------
# the primitives
# --------------------------------------------------------------------------


def test_cosine_is_zero_where_a_vector_is_missing_mismatched_or_degenerate() -> None:
    assert cosine((), ()) == 0.0
    assert cosine((1.0, 0.0), (1.0,)) == 0.0
    assert cosine((0.0, 0.0), (1.0, 0.0)) == 0.0
    assert cosine((1.0, 0.0), (0.0, 0.0)) == 0.0


def test_cosine_matches_the_definition() -> None:
    assert math.isclose(cosine((1.0, 1.0), (1.0, 0.0)), math.sqrt(2) / 2)


def test_jaccard_of_two_empty_sets_is_zero_rather_than_undefined() -> None:
    assert jaccard(frozenset(), frozenset()) == 0.0
    assert jaccard(frozenset({"a", "b"}), frozenset({"b"})) == 0.5


def test_recency_decays_by_tau_and_never_exceeds_one() -> None:
    assert recency(NOW, NOW, TAU) == 1.0
    assert recency(NOW - timedelta(days=TAU), NOW, TAU) == math.exp(-1.0)


def test_an_exemplar_learned_in_the_future_scores_as_brand_new() -> None:
    """Clock skew between the store and the caller must not inflate a score."""
    assert recency(NOW + timedelta(days=5), NOW, TAU) == 1.0
