# agentdb

**Agentic analytics shipped in 2026. Nobody published an accuracy number.**

This repository is building one — NL→SQL execution accuracy for ClickHouse and
Databricks, across several agent stacks, fully reproducible on your own hardware.
No published measurement puts agentic NL→SQL on a columnar OLAP engine and on a
lakehouse side by side, on identical questions, under one blind grader.

Three artifacts, in order of importance:

1. **`agenteval`** — an open harness that scores *any* agent stack (MCP servers,
   managed services, bare models) on identical tasks with a blind grader.
2. **`clickbench_nl`** — the first natural-language benchmark over ClickBench,
   and **`tpch_nl`**, the suite that crosses both engines.
3. **agentdb-server** — a plan-grounded MCP context layer, present as the
   reference implementation *under test*, not as the headline.

> Status: **M3.5.** The harness runs end to end against a live Databricks SQL
> warehouse, and `results/REPORT.md` holds the first measured numbers.
>
> **First result, stated plainly: `S5_claude_code` scores 100% (36/36) on
> `tpch_nl`.** That is a finding about the *suite*, not about the system. A seed
> set of twelve questions on which the first arm measured cannot make a mistake
> has no room to show a difference between arms, and the whole project turns on
> differences between arms. M4's job is now a suite that discriminates — which
> is why the next tasks are being written against the failure modes this run
> could not provoke, not against the ones it could.
>
> Whatever the ladder shows lands here — including a small or negative effect
> from grounding.

Every number is regenerable with `make bench`, and every claim traces to a
committed trace in `results/raw/`. Two things worth knowing before reading any
of them:

- The Databricks half needs **no local infrastructure**: `samples.tpch` ships
  pre-loaded in every workspace, Free Edition included.
- `S5_claude_code` measures **Claude Code the product**, reachable on a
  subscription rather than an API key. It is a Family S row, never the A0
  baseline: the product carries 16k-30k tokens of its own context and the
  operator's instruction files into every call, which is recorded on each
  attempt rather than argued about later.

## Docs

- [`docs/methodology.md`](docs/methodology.md) — how a number is produced, and
  every threat to validity known to the author.
- [`docs/databricks-grounding.md`](docs/databricks-grounding.md) — what an agent
  needs to know about Delta that the schema does not say: the 32-column data
  skipping limit, `PushedFilters` versus `DataFilters`, and why `EXPLAIN COST`
  without `ANALYZE` is worse than no plan at all.
- [`docs/clickhouse-advisor.md`](docs/clickhouse-advisor.md) — the same for
  ClickHouse: sparse indexes and granule pruning, why sort keys are
  cardinality-ordered rather than selectivity-ordered, projection versus skip
  index.
- [`docs/architecture.md`](docs/architecture.md) — the layers, and the one
  import CI forbids.
- [`docs/security.md`](docs/security.md) — read-only is enforced at the
  connection, not by inspecting SQL strings. Includes what is *not* built yet.

## Development

```bash
make install   # uv sync
make check     # lint + mypy --strict + import contracts + tests at 100% coverage
make up        # ClickHouse + Postgres/pgvector (the exemplar store, not a system under test)
```

## License

Apache-2.0.
