"""What the exemplar store's value objects refuse to represent (SPEC §10.2).

The invariants here are the ones that would otherwise surface as a silent
scoring bug: a "successful" exemplar carrying an error class ranks with the
positive exemplars while describing a failure, and an exemplar naming no
relation can never be invalidated by a schema change because nothing about it
is checkable.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from agentdb.core.memory import (
    Exemplar,
    ExemplarDraft,
    Outcome,
    Provenance,
    SchemaVersion,
)

NOW = datetime(2026, 8, 22, 12, 0, tzinfo=UTC)


def draft(**overrides: object) -> ExemplarDraft:
    fields: dict[str, object] = {
        "engine": "clickhouse",
        "namespace": "agentdb",
        "question": "how many hits per counter?",
        "sql": "SELECT CounterID, count() FROM hits GROUP BY CounterID",
        "normalized_sql": "SELECT CounterID, count() FROM hits GROUP BY CounterID",
        "relations": ("hits",),
        "columns": ("CounterID",),
        "outcome": Outcome.SUCCESS,
    }
    fields.update(overrides)
    return ExemplarDraft(**fields)  # type: ignore[arg-type]  # test factory, keyed by field name


def recorded(**overrides: object) -> Exemplar:
    fields: dict[str, object] = {
        "id": 1,
        "engine": "clickhouse",
        "namespace": "agentdb",
        "question": "how many hits per counter?",
        "sql": "SELECT CounterID, count() FROM hits GROUP BY CounterID",
        "normalized_sql": "SELECT CounterID, count() FROM hits GROUP BY CounterID",
        "relations": ("hits",),
        "columns": ("CounterID",),
        "schema_version_id": 7,
        "outcome": Outcome.SUCCESS,
        "provenance": Provenance.AGENT,
        "valid_from": NOW,
        "tx_from": NOW,
    }
    fields.update(overrides)
    return Exemplar(**fields)  # type: ignore[arg-type]  # test factory, keyed by field name


def test_a_draft_must_name_a_relation_it_can_later_be_invalidated_against() -> None:
    with pytest.raises(ValueError, match="at least one relation"):
        draft(relations=())


def test_a_successful_draft_may_not_carry_an_error_class() -> None:
    with pytest.raises(ValueError, match="cannot carry an error_class"):
        draft(error_class="semantic")


def test_a_failed_draft_must_carry_an_error_class() -> None:
    with pytest.raises(ValueError, match="must carry an error_class"):
        draft(outcome=Outcome.ERROR)


def test_a_rejected_plan_is_recordable_and_is_a_negative_exemplar() -> None:
    rejected = draft(outcome=Outcome.REJECTED, error_class="plan_rejection")

    assert rejected.outcome is Outcome.REJECTED
    assert rejected.provenance is Provenance.AGENT


def test_time_axes_read_independently() -> None:
    live = recorded()
    invalidated = recorded(valid_to=NOW + timedelta(days=3))
    corrected = recorded(tx_to=NOW + timedelta(days=1))

    assert live.is_valid and live.is_current
    assert not invalidated.is_valid and invalidated.is_current
    assert corrected.is_valid and not corrected.is_current


def test_a_non_success_outcome_is_what_makes_an_exemplar_negative() -> None:
    assert not recorded().is_negative
    assert recorded(outcome=Outcome.ERROR, error_class="semantic").is_negative
    assert recorded(outcome=Outcome.REJECTED, error_class="plan_rejection").is_negative


def test_a_schema_version_is_current_until_it_is_superseded() -> None:
    version = SchemaVersion(
        id=7,
        engine="clickhouse",
        namespace="agentdb",
        fingerprint="abc",
        layout_json={"relations": []},
        observed_at=NOW,
    )

    assert version.is_current
    assert not replace(version, superseded_at=NOW).is_current
