"""Recommendations as text an agent reads before it writes (SPEC §11.3, arm A6).

What A6 adds over A5 is *this rendering*, so what it leaves out matters as much
as what it includes:

* **The DDL is not here.** An agent answering a question cannot run ``ALTER
  TABLE``, and a migration script in its context is tokens spent on something it
  will never do. What it needs is the *fact* the recommendation rests on — this
  column has no statistics, that filter cannot prune — because that fact changes
  the query it writes. The DDL is for the human reading ``advise_*`` output.
* **The confidence is here, in words.** A heuristic and a measured finding read
  differently on purpose: an agent that treats "may help" as "will help" writes
  a worse query than one that was told which is which.
* **Rewrites lead.** A rewrite changes what the agent should write next; a
  physical-design recommendation only changes what it should expect.

Deterministic, like every other measured payload: two runs of one arm must
render byte-identically or the token columns compare noise.
"""

from __future__ import annotations

from collections.abc import Sequence

from agentdb.core.advisor.base import Confidence, Kind, Recommendation

HEADER = "What this table's physical design means for the query you are about to write:"

MAX_QUOTED_REDUCTION = 0.95
"""Ceiling on a quoted reduction.

Every estimate in the advisor is an upper bound derived from selectivity, and a
selective enough column drives that bound to 1.0. "Removes 100% of the granules"
is a promise no index keeps — pruning happens at granule granularity, not at row
granularity — so the quoted figure stops short of the arithmetic.
"""

CONFIDENCE_WORDS = {
    Confidence.MEASURED: "measured",
    Confidence.ESTIMATED: "estimated",
    Confidence.HEURISTIC: "rule of thumb",
}

DESIGN_KINDS = frozenset(
    {
        Kind.ORDER_BY,
        Kind.SKIP_INDEX,
        Kind.PROJECTION,
        Kind.CLUSTER_BY,
        Kind.STATS_COLUMNS,
        Kind.COMPACTION,
        Kind.BROADCAST_HINT,
    }
)


def render_recommendations(recommendations: Sequence[Recommendation]) -> str:
    """The advisor's half of an ``A6_full`` payload, or ``""`` when it has nothing.

    An empty string rather than an empty header: a section that says "no advice"
    charges the arm tokens for the absence of the thing under measurement.
    """
    if not recommendations:
        return ""

    rewrites = [item for item in recommendations if item.kind is Kind.REWRITE]
    design = [item for item in recommendations if item.kind in DESIGN_KINDS]

    lines = [HEADER]
    for item in rewrites:
        lines.append(_rewrite_line(item))
    for item in design:
        lines.append(_design_line(item))
    return "\n".join(lines)


def _rewrite_line(item: Recommendation) -> str:
    """A rewrite, with the corrected SQL where one exists mechanically."""
    head = f"- {item.rationale}"
    if item.rewritten_sql is None:
        return head
    return f"{head}\n  Write it as: {_flatten(item.rewritten_sql)}"


def _design_line(item: Recommendation) -> str:
    """A physical-design finding, as the fact rather than as the migration."""
    effect = ""
    reduction = item.expected_effect.reduction
    if reduction is not None:
        # An upper bound, rendered as one. Selectivity of 1/167,955 rounds to
        # "100% of the granules", which is a promise no index keeps: pruning
        # happens at granule granularity, so the realised figure is always lower.
        effect = (
            f" Fixing the layout could remove up to {min(reduction, MAX_QUOTED_REDUCTION):.0%} "
            f"of the {item.expected_effect.metric.replace('_', ' ')}"
        )
    return f"- [{CONFIDENCE_WORDS[item.confidence]}] {item.rationale}{effect}"


def _flatten(sql: str) -> str:
    return " ".join(sql.split())
