"""Retrieved exemplars, as text an agent reads before it writes (SPEC §10.4).

Two blocks, never merged. Positives are examples to follow; negatives are the
queries that failed here, with the error class that killed them, offered as
explicit "do not do this" context. Nobody else ships the second block, which is
why arm ``A5_negmemory`` exists to say whether it is worth its tokens.

Rendering is deterministic, like every other payload the benchmark measures: two
runs of one arm must produce byte-identical context, or the token columns and the
paired significance tests are comparing noise rather than grounding.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from agentdb.core.memory.models import ScoredExemplar

POSITIVE_HEADER = "Queries that answered questions about these tables before:"
NEGATIVE_HEADER = "Queries that failed against these tables before — do not repeat them:"


def render_exemplars(
    positive: Sequence[ScoredExemplar] = (),
    negative: Sequence[ScoredExemplar] = (),
) -> str:
    """The exemplar half of a grounded payload, or ``""`` when there is nothing to show.

    An empty string rather than an empty header: a block that says "no exemplars"
    teaches an agent to skim the section, and it charges the arm tokens for the
    absence of the thing being measured.
    """
    blocks = []
    if positive:
        blocks.append(_block(POSITIVE_HEADER, positive, _positive_line))
    if negative:
        blocks.append(_block(NEGATIVE_HEADER, negative, _negative_line))
    return "\n\n".join(blocks)


def _block(
    header: str,
    items: Sequence[ScoredExemplar],
    line: Callable[[int, ScoredExemplar], str],
) -> str:
    return "\n".join([header, *(line(index, item) for index, item in enumerate(items, start=1))])


def _positive_line(index: int, item: ScoredExemplar) -> str:
    exemplar = item.exemplar
    facts = []
    if exemplar.rows_returned is not None:
        facts.append(f"{exemplar.rows_returned:,} rows")
    if exemplar.bytes_read is not None:
        facts.append(f"{exemplar.bytes_read / 1024 / 1024:.1f} MiB read")
    suffix = f"  [{'; '.join(facts)}]" if facts else ""
    return f"{index}. Q: {exemplar.question}\n   SQL: {_flatten(exemplar.sql)}{suffix}"


def _negative_line(index: int, item: ScoredExemplar) -> str:
    """A failure is only useful with the reason attached.

    The error class is the actionable half — a semantic error means the names
    were wrong, a plan rejection means the shape was — so it leads the line and
    the engine's own message follows it.
    """
    exemplar = item.exemplar
    reason = exemplar.error_class or exemplar.outcome.value
    detail = f": {_flatten(exemplar.error_text)}" if exemplar.error_text else ""
    return f"{index}. Q: {exemplar.question}\n   SQL: {_flatten(exemplar.sql)}\n   {reason}{detail}"


def _flatten(text: str | None) -> str:
    """Collapse to one line: a payload whose whitespace varies is not deterministic."""
    return " ".join((text or "").split())
