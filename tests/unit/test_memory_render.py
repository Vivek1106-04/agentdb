"""The exemplar block an agent reads before it writes (SPEC §10.4).

Determinism and separation are the two properties under test. A payload whose
whitespace or ordering varied between runs would make the token columns and the
paired tests compare noise; a block that mixed successes with failures would
teach a model to copy a query that did not work.
"""

from __future__ import annotations

from datetime import UTC, datetime

from agentdb.core.memory.models import Exemplar, Outcome, Provenance, ScoredExemplar
from agentdb.core.memory.render import NEGATIVE_HEADER, POSITIVE_HEADER, render_exemplars

NOW = datetime(2026, 8, 22, 12, 0, tzinfo=UTC)


def scored(
    *,
    question: str = "how many hits per counter?",
    sql: str = "SELECT CounterID, count()\n  FROM hits\n  GROUP BY CounterID",
    outcome: Outcome = Outcome.SUCCESS,
    rows_returned: int | None = None,
    bytes_read: int | None = None,
    error_class: str | None = None,
    error_text: str | None = None,
) -> ScoredExemplar:
    exemplar = Exemplar(
        id=1,
        engine="clickhouse",
        namespace="agentdb",
        question=question,
        sql=sql,
        normalized_sql="SELECT ?",
        relations=("hits",),
        columns=("CounterID",),
        schema_version_id=1,
        outcome=outcome,
        provenance=Provenance.AGENT,
        valid_from=NOW,
        tx_from=NOW,
        rows_returned=rows_returned,
        bytes_read=bytes_read,
        error_class=error_class,
        error_text=error_text,
    )
    return ScoredExemplar(exemplar=exemplar, score=1.0, components={})


def test_nothing_retrieved_renders_nothing_at_all() -> None:
    """An empty header would charge the arm tokens for the absence of the thing measured."""
    assert render_exemplars() == ""


def test_a_positive_exemplar_carries_its_question_its_sql_and_its_cost() -> None:
    block = render_exemplars([scored(rows_returned=12_345, bytes_read=4 * 1024 * 1024)])

    assert POSITIVE_HEADER in block
    assert "1. Q: how many hits per counter?" in block
    assert "SQL: SELECT CounterID, count() FROM hits GROUP BY CounterID" in block
    assert "[12,345 rows; 4.0 MiB read]" in block


def test_an_unmeasured_exemplar_renders_no_empty_brackets() -> None:
    block = render_exemplars([scored()])

    assert "[" not in block


def test_a_negative_exemplar_leads_with_the_error_class() -> None:
    block = render_exemplars(
        negative=[
            scored(
                outcome=Outcome.ERROR,
                error_class="semantic",
                error_text="Code: 47.  UNKNOWN_IDENTIFIER\n  UserID",
            )
        ]
    )

    assert NEGATIVE_HEADER in block
    assert "semantic: Code: 47. UNKNOWN_IDENTIFIER UserID" in block


def test_a_failure_with_no_error_class_still_names_what_happened() -> None:
    block = render_exemplars(negative=[scored(outcome=Outcome.REJECTED)])

    assert "rejected" in block


def test_the_two_blocks_are_separate_and_ordered() -> None:
    block = render_exemplars([scored()], [scored(outcome=Outcome.ERROR, error_class="timeout")])

    assert block.index(POSITIVE_HEADER) < block.index(NEGATIVE_HEADER)


def test_rendering_is_byte_identical_across_calls() -> None:
    first = render_exemplars([scored(bytes_read=10)], [scored(error_class="syntax")])
    second = render_exemplars([scored(bytes_read=10)], [scored(error_class="syntax")])

    assert first == second
