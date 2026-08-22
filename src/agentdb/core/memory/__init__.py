"""Bi-temporal exemplar memory (SPEC §10).

Query memory, not agent memory. What is stored is a validated
question→SQL→outcome triple bound to a *schema version*, on two independent
time axes: valid time says when the exemplar was true of the schema,
transaction time says when agentdb learned it. Corrections append; nothing is
mutated and nothing is deleted.

The point is a failure a single-timestamp store cannot avoid: an agent reuses a
query that was correct three weeks ago against a column that has since been
renamed or retyped. Here the invalidation signal is mechanical — a schema
fingerprint changed — rather than inferred, which is the whole reason this is
worth building on schema-bound SQL and not on conversation.

Postgres is agentdb's private state store, never a system under test. It speaks
plain SQL through this package rather than through the :mod:`agentdb.adapters`
protocol, and the import-linter contract of SPEC §12 keeps
``agentdb.adapters.*`` from importing it — that is what stops the memory store
from reappearing as a third engine.
"""

from __future__ import annotations

from agentdb.core.memory.embedding import (
    EMBEDDING_DIMENSIONS,
    Embedder,
    HashingEmbedder,
)
from agentdb.core.memory.fingerprint import (
    NamespaceSnapshot,
    RelationSnapshot,
    fingerprint,
    invalidation_reason,
    snapshot,
    snapshot_from_json,
    snapshot_to_json,
)
from agentdb.core.memory.models import (
    Exemplar,
    ExemplarDraft,
    Outcome,
    Provenance,
    SchemaVersion,
    ScoredExemplar,
)
from agentdb.core.memory.normalize import normalize_sql
from agentdb.core.memory.ranking import rank

__all__ = [
    "EMBEDDING_DIMENSIONS",
    "Embedder",
    "Exemplar",
    "ExemplarDraft",
    "HashingEmbedder",
    "NamespaceSnapshot",
    "Outcome",
    "Provenance",
    "RelationSnapshot",
    "SchemaVersion",
    "ScoredExemplar",
    "fingerprint",
    "invalidation_reason",
    "normalize_sql",
    "rank",
    "snapshot",
    "snapshot_from_json",
    "snapshot_to_json",
]
