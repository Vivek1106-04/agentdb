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

Design notes and the benchmark methodology land in `docs/` as the harness comes
together — `docs/methodology.md` (including threats to validity) ships with the
first results.

## Development

```bash
make install   # uv sync
make check     # lint + mypy --strict + import contracts + tests at 100% coverage
make up        # ClickHouse + Postgres/pgvector (the exemplar store, not a system under test)
```

## License

Apache-2.0.
