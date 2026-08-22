"""What the exemplar store holds, as Python (SPEC §10.2).

One row of ``agentdb_exemplar`` is one :class:`Exemplar`; one row of
``agentdb_schema_version`` is one :class:`SchemaVersion`. Both carry their
identity from the database, so a value that has never been written has
``id=None`` and is an :class:`ExemplarDraft` instead — the type distinguishes
"what the agent proposes to remember" from "what the store has recorded", which
is the difference between a value that can be ranked and one that cannot.

The two time axes are here in full rather than collapsed into one timestamp,
because the collapse is the bug: ``valid_to`` says the schema moved out from
under this query, ``tx_to`` says agentdb replaced its record of it, and only
keeping both can answer *when did this stop working, and what changed*.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from agentdb.adapters import Engine


class Outcome(StrEnum):
    """What happened when the exemplar's SQL was last run.

    ``REJECTED`` is not an engine error: it is agentdb refusing to run a plan,
    and it is worth remembering separately because the query may be perfectly
    valid SQL that simply must not be issued against this layout.
    """

    SUCCESS = "success"
    ERROR = "error"
    REJECTED = "rejected"


class Provenance(StrEnum):
    """Where an exemplar came from.

    Provenance is reported alongside every retrieved exemplar because the three
    sources carry different authority: a curated exemplar was written by a human,
    a mined one was observed in a real workload, and an agent-written one is only
    as good as the outcome recorded beside it.
    """

    AGENT = "agent"
    WORKLOAD_MINED = "workload_mined"
    CURATED = "curated"


@dataclass(frozen=True, slots=True)
class SchemaVersion:
    """One observed state of a namespace's schema and physical layout.

    ``superseded_at`` is ``None`` for the current version. Superseding never
    deletes: the old row is what an invalidated exemplar points back to when it
    explains which layout it was true of.
    """

    id: int
    engine: Engine
    namespace: str
    fingerprint: str
    layout_json: Mapping[str, object]
    observed_at: datetime
    superseded_at: datetime | None = None

    @property
    def is_current(self) -> bool:
        return self.superseded_at is None


@dataclass(frozen=True, slots=True)
class ExemplarDraft:
    """An exemplar proposed for recording, before the store has assigned it one.

    ``relations`` and ``columns`` are what makes cheap re-validation possible:
    on a fingerprint change the store asks only whether these names still exist
    with compatible types, which is a set comparison rather than a re-execution.
    """

    engine: Engine
    namespace: str
    question: str
    sql: str
    normalized_sql: str
    relations: tuple[str, ...]
    columns: tuple[str, ...]
    outcome: Outcome
    provenance: Provenance = Provenance.AGENT
    rows_returned: int | None = None
    bytes_read: int | None = None
    duration_ms: int | None = None
    error_class: str | None = None
    error_text: str | None = None

    def __post_init__(self) -> None:
        if not self.relations:
            raise ValueError("an exemplar must name at least one relation")
        if self.outcome is Outcome.SUCCESS and self.error_class is not None:
            raise ValueError("a successful exemplar cannot carry an error_class")
        if self.outcome is not Outcome.SUCCESS and self.error_class is None:
            raise ValueError(f"a {self.outcome} exemplar must carry an error_class")


@dataclass(frozen=True, slots=True)
class Exemplar:
    """A recorded exemplar, on both time axes (SPEC §10.2)."""

    id: int
    engine: Engine
    namespace: str
    question: str
    sql: str
    normalized_sql: str
    relations: tuple[str, ...]
    columns: tuple[str, ...]
    schema_version_id: int
    outcome: Outcome
    provenance: Provenance
    valid_from: datetime
    tx_from: datetime
    embedding: tuple[float, ...] = ()
    rows_returned: int | None = None
    bytes_read: int | None = None
    duration_ms: int | None = None
    error_class: str | None = None
    error_text: str | None = None
    valid_to: datetime | None = None
    tx_to: datetime | None = None

    @property
    def is_valid(self) -> bool:
        """Whether the exemplar is still true of the current schema."""
        return self.valid_to is None

    @property
    def is_current(self) -> bool:
        """Whether this is the live record rather than a superseded revision."""
        return self.tx_to is None

    @property
    def is_negative(self) -> bool:
        """Whether this is a "do not do this" exemplar (SPEC §10.4, arm A5)."""
        return self.outcome is not Outcome.SUCCESS


@dataclass(frozen=True, slots=True)
class ScoredExemplar:
    """One retrieved exemplar with the ranking that selected it (SPEC §10.4).

    ``components`` is kept beside the total because every weight in the score is
    an ablation arm: a report that says retrieval helped has to be able to say
    which term did the work.
    """

    exemplar: Exemplar
    score: float
    components: Mapping[str, float]
