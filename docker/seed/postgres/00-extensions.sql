-- The exemplar store's one requirement. Runs once, on first container start.
--
-- Postgres holds the bi-temporal exemplar store of SPEC §10 and nothing else.
-- No agent connects here, so there is no read-only benchmark role: the
-- read-only boundary that matters lives on the measured engines (ClickHouse's
-- agentdb_ro user, and the Databricks principal's SELECT-only grant).
CREATE EXTENSION IF NOT EXISTS vector;
