# `tpch_nl` — the cross-engine suite

Natural-language questions over the TPC-H schema, answerable on **both** measured
engines from the same question text, the same gold SQL and the same grader. That
sameness is the point: SPEC §18.6 asks that the same questions run on an engine
this project never touched and land on the same axis, and a suite that needed
per-engine rewrites would not prove it.

`clickbench_nl` stays ClickHouse-only — `hits` is ~100M rows over 105 columns and
will not fit a Databricks Free Edition workspace (spec 1.2). `tpch_nl` is the
suite that crosses.

## Status: seed set, not v1

This directory holds a **seed** of hand-authored tasks. SPEC §16 puts the full
~60-task `tpch_nl` in M4; what is here exists so the Databricks path can be run
end to end while M3.5 is verified. Every task here still gets the M4 treatment —
two-pass authoring, gold hashes frozen against trusted data, ids stable.

## Data

* **Databricks** — `samples.tpch`, pre-loaded in every workspace including Free
  Edition. Nothing to load, no cluster to size.
* **ClickHouse** — *not yet seeded.* `docker/seed/clickhouse` creates the
  database and the read-only user but no TPC-H tables, so tasks here declare
  `engine: [databricks]` until a loader lands. Adding `clickhouse` to that list
  is then a one-word change per task.

The scale factor must match across engines before a task runs on both: identical
questions over different data produce different gold results, which would turn a
cross-engine comparison into two unrelated experiments. `samples.tpch` is SF1
(6,001,215 `lineitem` rows) and the ClickHouse side must be loaded to match.

## Authoring rules

1. **Portable SQL.** Gold uses standard forms both engines accept — `DATE`
   literals, `EXTRACT`, plain aggregates. No engine-specific function survives
   review, because gold that only runs on one engine is not cross-engine gold.
2. **Fully qualified is *not* required in gold.** The executor pins catalog and
   schema per statement. Under-qualification is a property the benchmark
   *measures* (`UNQUALIFIED_RELATION`), not a style the gold enforces.
3. **The question never names a column.** A question that says
   `l_shipdate` is testing transcription, not schema grounding.
4. **`notes` says what the task probes.** Usually the piece of physical design or
   schema semantics that a raw schema dump cannot convey.
5. **No task is edited in response to how a system scored on it** (SPEC §11.5.1).
   Task authoring is git-timestamped before any Family S run.

## Provenance

Questions are authored for this project. Where a task's gold derives from a
numbered TPC-H query, the id and `tags` record it so the lineage is checkable
against the TPC-H specification.
