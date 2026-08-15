# Postgres exists in this stack for exactly one job: the bi-temporal exemplar
# store of SPEC §10. It is infrastructure, never a system under test — spec 1.2
# replaced it with Databricks as the second measured engine.
#
#   pgvector — exemplar embeddings (SPEC §10.2)
#
# hypopg and pg_stat_statements were here when Postgres was an engine under
# measurement. Hypothetical-index validation is gone entirely (spec 1.2 item 3:
# shadow validation is the single validation story for both engines), and
# workload mining reads system.query_log / system.query.history, never
# pg_stat_statements. Both are removed rather than left installed: an unused
# extension in a published image is a claim about what this project does.
FROM pgvector/pgvector:pg17
