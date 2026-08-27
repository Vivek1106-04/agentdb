# Architecture

Three things live in this repository, and the boundaries between them are
enforced by CI rather than by discipline.

```
┌──────────────────────────────────────────────────────────────────┐
│                         MCP Client                               │
│           (Claude Code / Cline / Cursor / custom agent)          │
└───────────────────────────────┬──────────────────────────────────┘
                                │  MCP 2026-07-28  (stdio | Streamable HTTP)
┌───────────────────────────────▼──────────────────────────────────┐
│                       agentdb-server                             │
│  ─ tool registry, typed outputSchema contracts                   │
│  ─ auth tier (OAuth 2.1 resource server, JWKS)   [HTTP only]     │
│  ─ egress bounding, permit-pool concurrency                      │
│  ─ client_context_id namespacing for query auditability          │
└───────────────────────────────┬──────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────┐
│                        agentdb-core                              │
│                                                                  │
│  ContextBuilder ─ assembles the grounded context payload         │
│  PlanAnalyzer   ─ normalizes engine plans into a common IR       │
│  StatsProfiler  ─ cardinality / null-ratio / distribution probes │
│  Advisor        ─ index / sort-key / projection recommendations  │
│  WorkloadMiner  ─ mines executed queries into exemplars          │
│  MemoryStore    ─ bi-temporal exemplar + schema-version store    │
│  SafetyGate     ─ read-only enforcement, cost ceilings, timeouts │
└──────────┬──────────────────────────────────┬────────────────────┘
           │                                  │
┌──────────▼───────────┐          ┌───────────▼──────────┐
│  DatabricksAdapter   │          │  ClickHouseAdapter   │
│  EXPLAIN FORMATTED   │          │  EXPLAIN indexes=1,  │
│  EXPLAIN COST        │          │    projections=1     │
│  query history API   │          │  system.query_log    │
│  DESCRIBE DETAIL     │          │  system.parts /      │
│  information_schema  │          │    columns / tables  │
└──────────────────────┘          └──────────────────────┘

  Postgres + pgvector appears in exactly one place: the exemplar
  store. It is agentdb's private state, never a measured engine.

           ┌───────────────────────────────────────┐
           │            agenteval                  │
           │  task suites · runner · scorer ·      │
           │  ablation matrix · report generator   │
           └───────────────────────────────────────┘
```

## The one rule that matters

**`agenteval` imports nothing from `agentdb`.** Not a model, not a constant, not
a type. The harness measures agentdb the same way it measures `mcp-clickhouse`,
ClickHouse Agents, or Genie: through the `SystemUnderTest` protocol, over the
same tasks, with the same blind grader.

This is not a style preference. A benchmark whose harness shares code with one of
its subjects is a self-report, and every number it produces is worth less for it.
So it is a CI-enforced import-linter contract, and it fails the build:

```toml
[[tool.importlinter.contracts]]
name = "agenteval is vendor-neutral: it must never import agentdb"
type = "forbidden"
source_modules = ["agenteval"]
forbidden_modules = ["agentdb"]
```

The cost is real and paid deliberately: `QueryExecutor`, `ErrorClass`, and the
engine literals are declared twice, once on each side of the wall. That
duplication is the price of the property.

Three further contracts keep engine knowledge where it belongs — core and the
server know the `Adapter` protocol and never a concrete engine; adapters never
reach into the memory store. `make arch` runs all four.

## Layers, and what each may know

| Layer | Knows | Must not know |
|---|---|---|
| `agentdb.config` | Environment, defaults | Anything above it |
| `agentdb.adapters` | One engine's SQL and system tables | Core, the server, the memory store |
| `agentdb.core` | The `Adapter` protocol, the plan IR | Any concrete adapter |
| `agentdb.server` | Core, tool schemas, transports | Any concrete adapter |
| `agentdb.bench` | All of the above, as a provider factory | — |
| `agenteval` | Systems under test, through protocols | **`agentdb`, entirely** |

Adapter parity is the design rule underneath that table: anything core asks for
must be expressible on both engines or explicitly declared unsupported through a
capability flag. Engine-specific facts live in the adapter and in the
engine-specific rule module beside it (`plan_rules.py`,
`plan_rules_databricks.py`), never in the shape of a core interface. The
asymmetries are real — ClickHouse prunes granules, Delta prunes files — and they
are represented as data (`pruning_unit`), not smoothed away.

## The plan IR

Both engines' plans normalize into one `PlanSummary` of `PlanNode`s, each
carrying what that engine could actually report:

- ClickHouse: `granules_total` / `granules_selected`, read from
  `EXPLAIN indexes = 1` **before** the query runs.
- Databricks: `files_total` / `files_selected`, available only **after**
  execution, and `files_total` comes from `DESCRIBE DETAIL` rather than from the
  plan at all.

A ratio is computed only when both numbers were genuinely measured. Missing
evidence stays `None` and every downstream rule falls silent rather than
guessing — a warning nobody can check is worse than no warning.

`PlanWarning` is the output: a code, a severity, the columns it fired on, a
human-readable message, and where possible a suggested rewrite. The full code
list is in `core/plan_ir.py`; the engine-specific reasoning is documented in
[`clickhouse-advisor.md`](clickhouse-advisor.md) and
[`databricks-grounding.md`](databricks-grounding.md).

## agenteval's shape

```
Task ──> SystemUnderTest.answer() ──> Attempt ──> .blind() ──> Score ──> trace
                                                    │
                                          system identity stripped here
```

Four properties are structural rather than promised:

1. **The scorer never sees which system it grades.** It takes a `BlindAttempt`;
   `Attempt.blind()` is the only way to make one.
2. **Gold is resolved once per task** and shared by every arm, so no two arms are
   graded against different truth.
3. **The run loop is sequential.** Running arms concurrently against one engine
   would let a busy arm inflate a quiet arm's latency and bytes read, and those
   are reported numbers.
4. **A crash becomes a recorded failure, not a lost cell.** A provider outage
   mid-run produces an attempt with no queries, a note naming the exception, and
   a `no_query` verdict — visible in the report rather than absent from it.

Systems under test fall into three families:

- **Family A** — agentdb's own context ablations, A0 through A7. Each is a
  provider named by dotted path in `eval/providers.yaml` and resolved at run
  time, per engine. The indirection is the point: anyone can add a row for their
  own grounding service and be scored on the same tasks.
- **Family S, local** — MCP servers launched from `eval/servers.yaml`, driven by
  a model the harness controls. The SQL a server emits is re-executed through the
  harness's own connection so grading is byte-identical across arms.
- **Family S, managed** — ClickHouse Agents and Genie, from `eval/managed.yaml`.
  They select their own models, report no tokens, and may decline; see
  [`methodology.md`](methodology.md).

## What runs where

| Component | Process | Needs |
|---|---|---|
| `agentdb serve` | stdio or HTTP | An engine connection |
| `agentdb demo` | one-shot CLI | ClickHouse with ClickBench loaded |
| `python -m agenteval bench` | one-shot CLI | An engine, a model key, arm configs |
| `python -m agenteval report` | one-shot CLI | **Nothing.** Pure function of committed traces |

That last row is the property that makes the traces worth committing: anyone can
regenerate every published number from `results/raw/*.jsonl` and get the same
file back, with no model call and no engine.
