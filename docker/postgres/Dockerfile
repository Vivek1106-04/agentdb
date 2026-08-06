# Postgres with the three extensions the advisor and the memory store depend on:
#   pgvector           — exemplar embeddings (SPEC §10.2)
#   hypopg             — hypothetical-index validation (SPEC §9.2)
#   pg_stat_statements — workload mining (SPEC §8.2), enabled via command flags
FROM pgvector/pgvector:pg17

RUN apt-get update \
    && apt-get install -y --no-install-recommends postgresql-17-hypopg \
    && rm -rf /var/lib/apt/lists/*
