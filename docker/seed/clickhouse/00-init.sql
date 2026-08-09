-- Runs once, on first container start.
CREATE DATABASE IF NOT EXISTS agentdb;

-- Read-only profile. SPEC §13.3: agents connect through a role that cannot
-- write, rather than through a server that inspects SQL strings.
-- readonly = 1 also forbids changing settings, which would block the two things
-- the harness must do per query: tag it for query_log attribution (SPEC §8.4)
-- and lower its own ceilings. CHANGEABLE_IN_READONLY opens exactly those, and
-- the MAX constraints keep them ceilings rather than suggestions.
CREATE ROLE IF NOT EXISTS agentdb_ro
    SETTINGS readonly = 1,
             max_execution_time = 30 MAX 30 CHANGEABLE_IN_READONLY,
             max_result_rows = 10000 MAX 10000 CHANGEABLE_IN_READONLY,
             max_rows_to_read = 500000000,
             log_comment = '' CHANGEABLE_IN_READONLY;

GRANT SELECT ON agentdb.* TO agentdb_ro;
GRANT SELECT ON system.* TO agentdb_ro;
