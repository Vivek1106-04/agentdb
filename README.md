# agentdb

**Agentic analytics shipped in 2026. Nobody published an accuracy number.**

This repository is building one — NL→SQL execution accuracy for ClickHouse and
Databricks, across several agent stacks, fully reproducible on your own hardware.
No published measurement puts agentic NL→SQL on a columnar OLAP engine and on a
lakehouse side by side, on identical questions, under one blind grader.

Three artifacts, in order of importance:

1. **`agenteval`** — an open harness that scores *any* agent stack (MCP servers,
   managed services, bare models) on identical tasks with a blind grader.
2. **`clickbench_nl`** — the first natural-language benchmark over ClickBench.
3. **agentdb-server** — a plan-grounded MCP context layer, present as the
   reference implementation *under test*, not as the headline.

> Status: **pre-M1.** No results yet. When results exist they will appear here
> first, whatever they show — including a small or negative effect from
> grounding. Every number in this README will be regenerable with `make bench`.

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
