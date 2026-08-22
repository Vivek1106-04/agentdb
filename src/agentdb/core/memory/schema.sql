-- The bi-temporal exemplar store (SPEC §10.2).
--
-- Applied by ExemplarStore.ensure_schema(), idempotently, so a fresh clone needs
-- no migration step beyond `docker compose up`. The pgvector extension itself is
-- created by docker/seed/postgres/00-extensions.sql on first container start.
--
-- Nothing here is ever UPDATEd except to close a time window. Rows are not
-- deleted: "when did this query stop working, and what changed" is answerable
-- only because the superseded rows are still present.

CREATE TABLE IF NOT EXISTS agentdb_schema_version (
    id              BIGSERIAL PRIMARY KEY,
    engine          TEXT NOT NULL,
    namespace       TEXT NOT NULL,
    fingerprint     TEXT NOT NULL,          -- sha256 over sorted (relation, column, type, physical layout)
    layout_json     JSONB NOT NULL,
    observed_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    superseded_at   TIMESTAMPTZ,            -- NULL = current
    UNIQUE (engine, namespace, fingerprint)
);

CREATE TABLE IF NOT EXISTS agentdb_exemplar (
    id                  BIGSERIAL PRIMARY KEY,
    engine              TEXT NOT NULL,
    namespace           TEXT NOT NULL,
    question            TEXT NOT NULL,
    sql                 TEXT NOT NULL,
    normalized_sql      TEXT NOT NULL,      -- literals parameterized, for dedup
    relations           TEXT[] NOT NULL,
    columns             TEXT[] NOT NULL,
    schema_version_id   BIGINT NOT NULL REFERENCES agentdb_schema_version(id),
    outcome             TEXT NOT NULL CHECK (outcome IN ('success','error','rejected')),
    rows_returned       BIGINT,
    bytes_read          BIGINT,
    duration_ms         INTEGER,
    error_class         TEXT,               -- syntax | semantic | plan_rejection | timeout | permission
    error_text          TEXT,
    embedding           VECTOR(1536),
    -- bi-temporal axes
    valid_from          TIMESTAMPTZ NOT NULL,
    valid_to            TIMESTAMPTZ,        -- NULL = still valid
    tx_from             TIMESTAMPTZ NOT NULL DEFAULT now(),
    tx_to               TIMESTAMPTZ,        -- NULL = current record
    provenance          TEXT NOT NULL       -- 'agent' | 'workload_mined' | 'curated'
);

-- Approximate-nearest-neighbour over the semantic term. The hybrid ranking of
-- SPEC §10.4 runs in Python over the pool this index selects, because every
-- weight in that ranking is an ablation arm and an arm has to be re-runnable
-- against a fixed candidate set.
CREATE INDEX IF NOT EXISTS agentdb_exemplar_embedding_idx
    ON agentdb_exemplar USING hnsw (embedding vector_cosine_ops);

-- The retrieval predicate, verbatim: current records of one namespace that are
-- still true of the schema.
CREATE INDEX IF NOT EXISTS agentdb_exemplar_live_idx
    ON agentdb_exemplar (engine, namespace, outcome)
    WHERE valid_to IS NULL AND tx_to IS NULL;

-- Re-validation walks exemplars by the relations they name.
CREATE INDEX IF NOT EXISTS agentdb_exemplar_relations_idx
    ON agentdb_exemplar USING gin (relations);

-- Dedup on write, and the transaction-time chain a correction appends to.
CREATE INDEX IF NOT EXISTS agentdb_exemplar_normalized_idx
    ON agentdb_exemplar (engine, namespace, normalized_sql)
    WHERE tx_to IS NULL;
