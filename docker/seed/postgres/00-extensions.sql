-- Extensions agentdb requires. Runs once, on first container start.
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS hypopg;
CREATE EXTENSION IF NOT EXISTS vector;

-- A read-only role is the enforcement boundary for SPEC §13.3: read-only is a
-- connection property, never a SQL-string inspection.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'agentdb_ro') THEN
        CREATE ROLE agentdb_ro LOGIN PASSWORD 'agentdb_ro';
    END IF;
END
$$;

GRANT CONNECT ON DATABASE agentdb TO agentdb_ro;
GRANT USAGE ON SCHEMA public TO agentdb_ro;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO agentdb_ro;
