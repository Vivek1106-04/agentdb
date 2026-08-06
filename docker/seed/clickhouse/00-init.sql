-- Runs once, on first container start.
CREATE DATABASE IF NOT EXISTS agentdb;

-- Read-only profile. SPEC §13.3: agents connect through a role that cannot
-- write, rather than through a server that inspects SQL strings.
CREATE ROLE IF NOT EXISTS agentdb_ro
    SETTINGS readonly = 1,
             max_execution_time = 30,
             max_result_rows = 10000,
             max_rows_to_read = 500000000;

GRANT SELECT ON agentdb.* TO agentdb_ro;
GRANT SELECT ON system.* TO agentdb_ro;
