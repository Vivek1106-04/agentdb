"""The hybrid exemplar ranking of SPEC §10.4.

::

    score = w_sem     · cosine(embedding, q_embedding)
          + w_rel     · jaccard(exemplar.relations, candidate_relations)
          + w_success · [outcome = 'success']
          + w_recency · exp(-age_days / TAU)
          - w_cost    · normalized(bytes_read)

Kept pure, and kept out of SQL, for one reason: **every weight is an ablation
arm**. The report publishes what happens when each term is zeroed, which means
the ranking has to be re-runnable over a fixed candidate set without a database
and without a model. pgvector picks the pool; this picks the answer.

The cost term is normalized against the pool rather than an absolute byte
budget, because "expensive" only means anything relative to the alternatives
being ranked — a single 40GB scan is the cheapest option if it is the only one.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime

from agentdb.config import RetrievalWeights
from agentdb.core.memory.models import Exemplar, Outcome, ScoredExemplar

TERMS = ("sem", "rel", "success", "recency", "cost")
"""Score component names, in the order SPEC §10.4 writes them."""


def rank(
    candidates: Sequence[Exemplar],
    *,
    question_embedding: Sequence[float],
    relations: Sequence[str],
    now: datetime,
    weights: RetrievalWeights,
    tau_days: float,
    limit: int | None = None,
) -> tuple[ScoredExemplar, ...]:
    """Score and order ``candidates``, best first.

    Ties break on exemplar id, descending — the more recently recorded of two
    equally-scored exemplars wins. Without a deterministic tiebreak the same
    arm could return different context on two runs of the same seed, which
    would make a paired significance test compare noise.
    """
    max_bytes = max((c.bytes_read or 0 for c in candidates), default=0)
    wanted = frozenset(relations)
    scored = tuple(
        _score(
            candidate,
            question_embedding=question_embedding,
            wanted=wanted,
            now=now,
            weights=weights,
            tau_days=tau_days,
            max_bytes=max_bytes,
        )
        for candidate in candidates
    )
    ordered = sorted(scored, key=lambda s: (-s.score, -s.exemplar.id))
    return tuple(ordered if limit is None else ordered[:limit])


def _score(
    exemplar: Exemplar,
    *,
    question_embedding: Sequence[float],
    wanted: frozenset[str],
    now: datetime,
    weights: RetrievalWeights,
    tau_days: float,
    max_bytes: int,
) -> ScoredExemplar:
    components = {
        "sem": cosine(question_embedding, exemplar.embedding),
        "rel": jaccard(frozenset(exemplar.relations), wanted),
        "success": 1.0 if exemplar.outcome is Outcome.SUCCESS else 0.0,
        "recency": recency(exemplar.tx_from, now, tau_days),
        "cost": _normalized_cost(exemplar.bytes_read, max_bytes),
    }
    weighted = weights.as_mapping()
    total = sum(weighted[term] * components[term] for term in TERMS if term != "cost")
    total -= weighted["cost"] * components["cost"]
    return ScoredExemplar(exemplar=exemplar, score=total, components=components)


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity, or ``0.0`` where either side is absent or degenerate.

    A missing embedding scores zero rather than raising: an exemplar recorded
    before an embedder was configured is still rankable on its other four terms,
    and dropping it would quietly shrink the pool an ablation is measuring.
    """
    if len(left) != len(right) or not left:
        return 0.0
    norm_left = math.sqrt(sum(value * value for value in left))
    norm_right = math.sqrt(sum(value * value for value in right))
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    return dot / (norm_left * norm_right)


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    """Set overlap of two relation sets; ``0.0`` when both are empty."""
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def recency(learned_at: datetime, now: datetime, tau_days: float) -> float:
    """Exponential decay on transaction time — how recently agentdb *learned* this.

    Transaction time, not valid time, because valid time answers a different
    question: whether the exemplar is true at all. An exemplar that is still
    valid but was learned a year ago should rank below one learned yesterday,
    and both rank above one that is no longer valid — which retrieval excludes
    before ranking ever sees it.
    """
    age_days = max((now - learned_at).total_seconds(), 0.0) / 86_400.0
    return math.exp(-age_days / tau_days)


def _normalized_cost(bytes_read: int | None, max_bytes: int) -> float:
    """Cost relative to the most expensive candidate in the pool.

    An exemplar with no recorded cost scores ``0.0``: an unmeasured query is not
    evidence of an expensive one, and penalizing it would bias retrieval toward
    exemplars that happen to predate cost accounting.
    """
    if not bytes_read or max_bytes <= 0:
        return 0.0
    return bytes_read / max_bytes
