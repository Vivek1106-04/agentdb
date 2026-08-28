# Security posture

What this project does to make an LLM-driven query path safe, what it
deliberately does **not** do, and where the boundary actually is.

The short version: **an agent cannot write, because the account it connects as
cannot write.** Nothing here inspects a model's SQL and decides whether it looks
dangerous.

---

## 1. String-based SQL filtering is not a security boundary

It has to be said plainly, because it is the most common design in this space
and it does not work.

A layer that scans generated SQL for `DROP`, `INSERT`, or `;` is defeated by
comment injection, by dialect-specific syntax the parser does not model, by
encodings, by nesting, and — most reliably — by a query shape nobody thought to
add to the list. Worse, it *feels* like protection, so the real boundary gets
skipped.

So agentdb's read-only property is enforced where authorization actually lives:

**ClickHouse** — the harness and the server authenticate as `agentdb_ro`, a user
created with `readonly = 1` and granted `SELECT` and nothing more
(`docker/seed/clickhouse/00-init.sql`). A `DROP TABLE` from this account fails at
the server, whatever produced the string.

**Databricks** — a Unity Catalog service principal granted `USE CATALOG`,
`USE SCHEMA`, and `SELECT`. On this engine the boundary is stronger than on
ClickHouse, because UC grants *are* the authorization system. No SQL-parsing
"safety" layer is added on top, deliberately: it would add no protection and
invite false confidence.

The identifier quoting in `adapters/clickhouse_sql.py` is a **corruption check**,
not an injection defence, and its docstring says so — names reach the adapter
from core and from the engine's own system tables, never from a model.

---

## 2. Bounded everything

Read-only stops damage. It does nothing about a query that reads 40 GB and stalls
a shared warehouse. Every execution therefore carries ceilings, and on ClickHouse
those ceilings are set on the account so a caller can only ever lower them:

```sql
ALTER USER agentdb_ro
    SETTINGS readonly = 1,
             max_execution_time  = 30          MAX 30          CHANGEABLE_IN_READONLY,
             max_result_rows     = 10000       MAX 10000       CHANGEABLE_IN_READONLY,
             max_rows_to_read    = 500000000   MAX 500000000   CHANGEABLE_IN_READONLY,
             max_bytes_to_read   = 100000000000 MAX 100000000000 CHANGEABLE_IN_READONLY,
             result_overflow_mode = 'break'                    CHANGEABLE_IN_READONLY,
             log_comment          = ''                         CHANGEABLE_IN_READONLY;
```

`MAX` makes each one a ceiling rather than a default. `CHANGEABLE_IN_READONLY` is
required because `readonly = 1` otherwise forbids setting *any* session setting,
including the two the harness must set per query: its own tighter limits, and the
`log_comment` that makes an execution attributable. A ceiling the caller cannot
lower is not useful; a ceiling the caller can raise is not a ceiling.

On Databricks the equivalent bounds ride on the Statement Execution API call —
`row_limit`, `wait_timeout` — and the warehouse's own limits apply.

Two settings in that list are not limits at all. `use_query_condition_cache` and
`use_skip_indexes_on_data_read` must be switched **off** for `EXPLAIN` to report
honest index evidence on ClickHouse ≥ 25.9, and both only ever make the server do
*less* work-avoidance, never more. Locking them would break the plan tools
without protecting anything.

---

## 3. Write paths are separately privileged

Three things in agentdb write: `ANALYZE`, `OPTIMIZE`, and shadow-validation
tables. None of them may borrow the read-only connection — that is the whole
point of the read-only connection.

Shadow validation (`core/advisor/shadow.py`) is the sharpest case, and it is
built around four properties:

- **Opt-in.** Nothing runs unless `AGENTDB_ALLOW_SHADOW` is set. The default is
  `False` in `config.py`, not a comment in a README.
- **A separate, explicitly configured write channel**, into a scratch schema
  (`AGENTDB_DBX_SCRATCH_SCHEMA` on Databricks) that is never the catalog under
  measurement. Where no such channel is configured, validation does not happen
  and the recommendation stays labelled as an estimate.
- **Namespaced and capped.** Every table carries the `__agentdb_shadow` marker
  plus a per-run token, and the sample is bounded by `SHADOW_TABLE_MAX_ROWS`.
- **Cleaned up twice.** Dropped in a `finally`, including when the plan read
  fails — and reaped on startup anyway, because a `finally` does not run when the
  process is killed. That is what the marker is for. A chaos test kills the
  process mid-validation and asserts the orphan is found.

DDL is **never** executed on the user's behalf. Advisor output is DDL *text*.
Applying it requires an MCP elicitation the user accepts.

---

## 4. Secrets

- **Environment only.** No credential is read from a config file in this
  repository, and none is written to a trace.
- Server configs name the variables they need; a trace records the *name* and the
  literal `<from-env>` in place of any value (`mcp/config.py`).
- `.env` is gitignored. `.env.example` carries names and no values.
- Missing credentials fail **at the moment an arm that needs them is built**, not
  at import: a Family A run must not fail on a workspace token no arm in it uses.
- No credential has a default. A benchmark that silently reached a workspace the
  operator did not choose would be worse than one that refuses to start.

---

## 5. Auditability

Every execution is tagged with `agentdb:{client_context_id}:{turn}` — via
`log_comment` on ClickHouse, and via the returned `statement_id` on Databricks,
which is why the Statement Execution API is preferred over the DB-API connector:
attribution by primary key rather than by string-matching a comment.

Every benchmark cell is committed to `results/raw/*.jsonl` with the prompt, every
query emitted, the engine's reply, timings, and tokens. A reader auditing a
published claim does not have to trust the summary.

---

## 6. What is not built yet

Stated here rather than implied by silence.

- **Only stdio is wired.** Streamable HTTP is designed but not implemented
  (`server/transports.py` says so at the top). Consequently the OAuth 2.1
  resource-server tier, JWKS verification, and DNS-rebinding protection of SPEC
  §13.3 — all of which are HTTP-transport concerns — are **not implemented**.
  Do not deploy this as a remote server today.
- **The local ClickHouse account has no password.** `no_password` is deliberate
  for a compose file that binds to `127.0.0.1` and holds public benchmark data.
  Any deployment beyond a laptop must give the account a password from the
  environment.
- **Rate limiting** is per-run concurrency bounding, not a general-purpose
  limiter.

## Reporting a vulnerability

Open a GitHub issue for anything non-sensitive. For something that should not be
public first, email the address in the git history and expect a reply before it
goes anywhere else. Findings are credited unless the reporter asks otherwise.
