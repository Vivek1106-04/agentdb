# agentdb

[![CI](https://github.com/Vivek1106-04/agentdb/actions/workflows/ci.yml/badge.svg)](https://github.com/Vivek1106-04/agentdb/actions/workflows/ci.yml)
![coverage 100%](https://img.shields.io/badge/coverage-100%25-brightgreen)
![license Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)

**Agentic analytics shipped in 2026 on every engine. Nobody has measured it
across two of them.** This repository is building that measurement — ClickHouse
and Databricks, the same natural-language questions, several agent stacks, one
blind grader, every trace committed and every number regenerable on your own
hardware.

> ### Read this before the table below
>
> **The leaderboard has one row.** The harness, the suites, the grader, the
> arms and the report generator are built and green; the matrix has not been
> run, because running it needs API keys and a warehouse. What follows is
> honest about which cells exist and which do not. A benchmark that showed you
> a full-looking table today would be showing you numbers nobody measured.

---

## What has actually been measured

| arm | suite / engine | EX (95% CI) | cells | measured |
|---|---|---|---|---|
| `S5_claude_code` (claude-cli/sonnet) | `tpch_nl` / Databricks | 100.0% [100.0%, 100.0%] | 36 | 2026-08-15 |

That is a finding about the *seed set*, not about the system. It was taken on a
12-question set, before `tpch_nl` reached its full 60, and an arm that cannot
make a mistake leaves no room to show a difference between arms — which is the
whole point of the project. The suite was rewritten against the failure modes
that run could not provoke. Full detail, including the 19,462 tokens of product
context that arm carried per call: [`results/REPORT.md`](results/REPORT.md).

**What is built and waiting on a run:**

| Family | Arms | State |
|---|---|---|
| A — context ablations | `A0_baseline`, `A1_stats`, `A2_layout`, `A3_plan`, `A4_memory`, `A5_negmemory`, `A6_full`, `A7_oracle` | code green, unmeasured |
| S — local stacks | `S1_mcp_clickhouse`, `S2_mcp_agentdb`, `S5_agentdb`, `S5_claude_code` | one row measured, above |
| S — managed products | `S3_clickhouse_agents`, `S4a_genie_minimal`, `S4b_genie_curated` | code green; needs vendor access and their beta terms read first |

Suites: `clickbench_nl` (100 tasks, 37 ClickBench-derived and 63 authored) and
`tpch_nl` (60 tasks, both engines). Gold results are hashed per engine and
committed. Task authoring was finished and git-timestamped on 2026-08-16,
before any Family S run — which is what makes the contamination check mean
something.

`tpch_nl` asks natural-language questions over TPC-H-derived data (Databricks'
`samples.tpch`, and the same scale factor loaded locally). It measures
NL→SQL accuracy, not database performance: nothing here is an official TPC
benchmark result and none of it is comparable to one. `clickbench_nl` is
likewise a natural-language layer over the ClickBench `hits` dataset, not a
ClickBench performance run.

---

## Reproduce something in five minutes

**The fast one — Databricks, no local infrastructure at all:**

```bash
export AGENTEVAL_DBX_HOST=... AGENTEVAL_DBX_WAREHOUSE_ID=... AGENTEVAL_DBX_TOKEN=...
export ANTHROPIC_API_KEY=...
make bench-quick-dbx                 # five tasks, one seed, against samples.tpch
make report                          # regenerates REPORT.md + charts from the traces
```

`samples.tpch` ships pre-loaded in every Databricks workspace, **Free Edition
included** — [sign up](https://docs.databricks.com/aws/en/getting-started/free-edition),
no data loading, no container, no cost.

**The local one — ClickHouse in Docker:**

```bash
docker compose -f docker/docker-compose.yml up -d --wait
make load-clickbench CLICKBENCH_PARTS=100    # the full ~100M rows; this is the slow part
export ANTHROPIC_API_KEY=...
make bench-quick
```

`CLICKBENCH_PARTS=100` is not optional here, and the reason is the point:
`clickbench_nl`'s gold results are hashed against the full table, so a partial
load stops the run on gold drift rather than quietly scoring against different
data. That check is doing its job — but it means the ClickHouse path costs a
download, which is why the Databricks one is listed first.

`make report` calls no model and touches no engine. It is a pure function of
`results/raw/*.jsonl`, so anyone can regenerate every published number from the
committed evidence and get the same file back.

---

## What grounding actually changes

### ClickHouse — measured just now, on 99,997,497 rows

The question: *how many hits did counter 62 record?* Both queries return
**738172**.

<table>
<tr><th>Written from a schema dump</th><th>Written knowing the physical layout</th></tr>
<tr><td>

```sql
SELECT count() FROM agentdb.hits
WHERE toString(CounterID) = '62'
```

**255 KB read**, 65,362 rows

</td><td>

```sql
SELECT count() FROM agentdb.hits
WHERE CounterID = 62
```

**64.0 KB read**, 16,385 rows

</td></tr>
</table>

**4.0x**, for one function call. Nothing in a schema dump says `CounterID` is
the leading sort-key column, so an agent treats it as an ordinary attribute and
compares it as text. The answer is correct either way, and the cost is invisible
from where the agent is standing.

The plans, verbatim, from `EXPLAIN indexes = 1`:

```
                    toString(CounterID) = '62'          CounterID = 62
Condition:          (toString(CounterID) in ['62','62'])  (CounterID in [62, 62])
Parts:              4/4                                   1/4
Granules:           108/12366                             92/12366
Search Algorithm:   generic exclusion search              binary search
Ranges:             18                                    2
```

Note what this is *not*: the index does not stop working — ClickHouse falls back
to generic exclusion search and still prunes. It reads every part instead of
one, across 18 disjoint ranges instead of 2, and that is where the 4x lives.
Reproduce it yourself with `make demo`. (On a cold mark cache the first run
reads 575 KB; the number above is steady state, and the grounded side is stable
at 64.0 KB either way.)

### Databricks — where a predicate is spent

Delta skips files using per-file statistics, and it collects them for only the
first `delta.dataSkippingNumIndexedCols` columns — **32 by default**, counted in
schema order. A filter on column 41 of a wide table skips nothing at all. Not
"skips less". Nothing. The query succeeds, the plan says nothing about it, and
the schema handed to the agent looks identical either way.

Here is a real scan block from a Free Edition warehouse
([fixture](tests/fixtures/databricks/explain_formatted.txt)):

```
(1) PhotonScan parquet samples.tpch.lineitem
DictionaryFilters: [(l_shipdate#21951 >= 1995-01-01)]
RequiredDataFilters: [isnotnull(l_shipdate#21951), (l_shipdate#21951 >= 1995-01-01)]
```

There is no `PushedFilters` line at all — a `PhotonScan` prints
`RequiredDataFilters` instead, and a reader who knows only the documented
spelling concludes a well-pruned query is doing a full scan.

What agentdb hands the model *before* it writes SQL, when a filtered column
falls outside the statistics set:

> **`STATS_NOT_COLLECTED`** (critical) — `event_properties` has no per-file
> statistics on `main.analytics.events` (Delta indexes only the first 32 columns
> in schema order), so filtering on it cannot skip any file.
> *Suggested rewrite:* filter on an indexed column as well, or ask the table
> owner to set `delta.dataSkippingStatsColumns`.

The full mechanism, including why widening that property is **not retroactive**:
[`docs/databricks-grounding.md`](docs/databricks-grounding.md).

---

## Honest scope

Almost everything here already exists in some form. Databricks ships AI/BI
Genie, a good natural-language analytics product, and mature evaluation tooling
in MLflow — agentdb's Databricks adapter builds on their own primitives
(`system.query.history`, `DESCRIBE DETAIL`, `EXPLAIN FORMATTED`, Delta
data-skipping statistics), and `agenteval` is designed to complement MLflow
rather than replace it. ClickHouse ships a managed Claude-powered agent product
and engine-side automatic query optimization. What does **not** exist is a
*comparison*: no published execution-accuracy measurement puts agentic NL→SQL on
a columnar OLAP engine and on a lakehouse side by side, on identical questions,
under one blind grader, with every trace committed. agentdb's contribution is
that measurement and the apparatus that produces it — `clickbench_nl`, the first
natural-language benchmark over ClickBench, and `agenteval`, a harness that
scores any agent stack, including ones agentdb did not write. The context layer
exists to answer "does grounding the agent in physical design help, and by how
much, and does the answer differ by engine," with ablations. **If the delta
turns out to be small, that result gets published too.**

| Project | What it does | What it does not do |
|---|---|---|
| **ClickHouse Agents** (beta, May 2026) | Managed agentic analytics in ClickHouse Cloud, powered by Claude, MCP-native. | No published execution-accuracy number. **A system agentdb measures, not one it competes with.** |
| `ClickHouse/mcp-clickhouse` | Official OSS connector: `run_select_query`, listings, `chdb_query`. Read-only by default. | No plan introspection, no layout advice, no statistics, no accuracy evidence. |
| **Databricks AI/BI Genie** | Shipped NL analytics over Unity Catalog, curated spaces, public Conversation API. Genuinely good at the curated case. | Grounding is *authored* by a human, not derived from the engine. No plan introspection exposed to the caller. |
| **MLflow GenAI evaluation** | Real, mature eval tooling: judges, tracing, human labeling. | Judges *your* agent on *your* data. A framework for running an eval, not a published benchmark with fixed tasks and gold results. **Complementary.** |
| Photon / predictive optimization / automatic query optimization | Engine-side optimization, both vendors. | Improves *the query the agent already wrote*. Cannot fix an agent that omitted the partition predicate or defeated the sort key. Different layer. |
| `mem0`, `Zep`/`Graphiti` | Agent memory, bi-temporal in Graphiti's case. | Conversational, not query-shaped. No validated SQL exemplars keyed to schema versions. |
| ClickBench / CostBench | Excellent, adversarially checkable engine benchmarks. | Measure the *engine*. No natural-language layer, no agent. |

---

## Architecture

```
MCP client (Claude Code / Cline / Cursor)
        │  MCP 2026-07-28 · stdio
┌───────▼──────────────────────────────────────────┐
│ agentdb-server   typed outputSchema per tool     │
├──────────────────────────────────────────────────┤
│ agentdb-core                                     │
│   ContextBuilder · PlanAnalyzer · StatsProfiler  │
│   Advisor · WorkloadMiner · MemoryStore          │
├────────────────────┬─────────────────────────────┤
│ ClickHouseAdapter  │ DatabricksAdapter           │
│ EXPLAIN indexes=1  │ EXPLAIN FORMATTED           │
│ system.query_log   │ query history API           │
└────────────────────┴─────────────────────────────┘

┌──────────────────────────────────────────────────┐
│ agenteval — imports NOTHING from agentdb.        │
│ Enforced by an import-linter contract in CI.     │
└──────────────────────────────────────────────────┘
```

That last box is the load-bearing one. A benchmark whose harness shares code
with one of its subjects is a self-report. agentdb is measured through the same
`SystemUnderTest` interface as `mcp-clickhouse`, ClickHouse Agents and Genie —
and if it loses, that is the result.
[`docs/architecture.md`](docs/architecture.md).

---

## Install

```bash
uv sync                          # or: pip install -e .
```

**Claude Code**

```bash
claude mcp add agentdb -- uv run agentdb --engine clickhouse
```

**Cline / Cursor** — add to the MCP settings JSON:

```json
{
  "mcpServers": {
    "agentdb": {
      "command": "uv",
      "args": ["run", "agentdb", "--engine", "clickhouse"],
      "env": {
        "AGENTDB_CLICKHOUSE_HOST": "localhost",
        "AGENTDB_CLICKHOUSE_PORT": "58123",
        "AGENTDB_CLICKHOUSE_USER": "agentdb_ro",
        "AGENTDB_CLICKHOUSE_DATABASE": "agentdb"
      }
    }
  }
}
```

Only stdio is wired today. The Streamable HTTP transport and its OAuth
resource-server tier are designed and **not implemented** — do not deploy this
as a remote server yet. See [`docs/security.md`](docs/security.md).

The agent connects as an account that cannot write. Read-only is a property of
the connection, not of a layer that inspects SQL strings for dangerous-looking
words — that is not a security boundary and this project does not pretend
otherwise.

---

## Docs

- [**`docs/methodology.md`**](docs/methodology.md) — how a number is produced,
  and every threat to validity known to the author: contamination,
  gold-annotation error, single-instance timings, sampling, managed-service
  curation.
- [`docs/databricks-grounding.md`](docs/databricks-grounding.md) — the
  32-column limit, `PushedFilters` vs `DataFilters`, why `EXPLAIN COST` without
  `ANALYZE` is worse than no plan at all.
- [`docs/clickhouse-advisor.md`](docs/clickhouse-advisor.md) — sparse indexes
  and granule pruning, why sort keys are cardinality-ordered, projection versus
  skip index.
- [`docs/disclosure.md`](docs/disclosure.md) — the rules a measurement of
  another company's product is published under.
- [`docs/architecture.md`](docs/architecture.md) · [`docs/security.md`](docs/security.md)

Docstrings throughout the source cite `SPEC.md`, an unpublished design document
that fixed these decisions before they were built. Each citation is provenance
for a rule the docstring already states in full — nothing needed to read the
code lives only in that file.

## Development

```bash
make install   # uv sync
make check     # lint + mypy --strict + import contracts + tests at 100% coverage
make up        # ClickHouse + Postgres/pgvector (the exemplar store, not a system under test)
make demo      # the before/after above, measured live
```

Measurements of another company's product follow the disclosure rules in
[`docs/disclosure.md`](docs/disclosure.md): methodology and numbers go to the
vendor privately, with a stated response window, before anything is published.
Corrections are credited by name.

## License

Apache-2.0.
