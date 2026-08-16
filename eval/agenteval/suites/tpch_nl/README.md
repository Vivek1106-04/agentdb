# `tpch_nl` — the cross-engine suite

Natural-language questions over the TPC-H schema, answerable on **both** measured
engines from the same question text, the same gold SQL and the same grader. That
sameness is the point: SPEC §18.6 asks that the same questions run on an engine
this project never touched and land on the same axis, and a suite that needed
per-engine rewrites would not prove it.

`clickbench_nl` stays ClickHouse-only — `hits` is ~100M rows over 105 columns and
will not fit a Databricks Free Edition workspace (spec 1.2). `tpch_nl` is the
suite that crosses.

## Status: 60 tasks, both engines, gold frozen on each

Every task declares `engine: [clickhouse, databricks]` and every one has a
committed gold hash **per engine**. Task ids `000`–`059` are contiguous and
stable.

| | |
|---|---|
| tasks | 60 |
| runnable on ClickHouse | 60 |
| runnable on Databricks | 60 |
| difficulty | 11 easy, 28 medium, 21 hard |

Authoring is git-timestamped before any Family S run, per SPEC §11.5.1.

## Data — identical on both engines, verified rather than assumed

* **Databricks** — `samples.tpch`, pre-loaded in every workspace including Free
  Edition. Nothing to load, no cluster to size.
* **ClickHouse** — `make load-tpch`, which generates TPC-H with DuckDB's `tpch`
  extension (a port of the reference `dbgen`) and loads it over the HTTP
  interface. About a minute end to end.

The scale factor is **SF5**, because that is what `samples.tpch` holds and
Databricks' copy is the side that cannot be changed. Matching it is not optional:
identical questions over different data are two unrelated experiments, not a
comparison.

Counted on both engines on 2026-08-16, `dbgen(sf=5)` reproduces `samples.tpch`
**row for row**:

| table | rows |
|---|---|
| region | 5 |
| nation | 25 |
| supplier | 50,000 |
| customer | 750,000 |
| part | 1,000,000 |
| partsupp | 4,000,000 |
| orders | 7,500,000 |
| lineitem | **29,999,795** |

`scripts/load_tpch_clickhouse.py` refuses to load if the generator ever stops
reproducing these counts, rather than loading data that would quietly make every
cross-engine delta meaningless.

Column names, order, and decimal precision mirror `samples.tpch` exactly —
including `DECIMAL(18,2)` rather than the TPC-H specification's `DECIMAL(15,2)`,
because both engines derive a product's scale from its operands' and a width
mismatch would make `SUM(l_extendedprice * (1 - l_discount))` round differently
on the two engines.

## Two engine differences the suite works around rather than hides

**Nullability.** Delta declares every column nullable; the ClickHouse DDL
declares none. TPC-H data contains no NULLs, so every result agrees — but the A0
arm's schema dump genuinely reads differently per engine, and that is recorded
here rather than papered over.

**Outer joins are not portable, so the suite has none.** ClickHouse fills
non-matching outer-join rows with column *defaults* where Spark uses NULL.
Measured on customer 3, who has no orders:

```sql
SELECT count(o_orderkey) FROM customer LEFT JOIN orders ON c_custkey = o_custkey
WHERE c_custkey = 3
```

ClickHouse returns **1** — the default-filled `0` is counted. Databricks returns
**0**. `join_use_nulls=1` would reconcile them, but the harness connects as
`agentdb_ro`, which is `readonly=1`, and the setting is rejected. SPEC §11.2 puts
tasks that cannot be expressed identically on both engines out of this suite, so
gold uses inner joins, `IN`, and scalar subqueries throughout. TPC-H Q13 is the
casualty and belongs in a single-engine suite.

## What the tasks are built to provoke

The first live run scored 36/36 on the twelve seed tasks — too easy to separate
one arm from another, which is the only thing this benchmark is for. The 48 tasks
added in M4 target what that run could not provoke, grouped by failure mode:

| file | ids | what it probes |
|---|---|---|
| `revenue.yaml` | 000–006 | the original seed questions |
| `joins.yaml` | 007–011 | join order decided by statistics, not schema |
| `pruning.yaml` | 012–023 | filters on columns that are not sort-key prefixes |
| `semantics.yaml` | 024–035 | values the schema does not show |
| `temporal.yaml` | 036–045 | date boundaries invisible in the result shape |
| `multihop.yaml` | 046–053 | deep joins, aliased dimensions, composite keys |
| `aggregation.yaml` | 054–059 | high-cardinality grouping and wide column reads |

Both engines store `lineitem` ordered by `(l_orderkey, l_linenumber)` and
`orders` by `o_orderkey`, while nearly every business question filters on a date.
No date column is a sort-key prefix on either engine, so the §7 failure mode is
present in the data by construction rather than simulated.

## Authoring rules

1. **Portable SQL.** Gold uses standard forms both engines accept — `DATE`
   literals, `EXTRACT`, `INTERVAL`, plain aggregates. No engine-specific function
   survives review, because gold that only runs on one engine is not
   cross-engine gold. No outer joins, for the reason above.
2. **Fully qualified is *not* required in gold.** The executor pins catalog and
   schema per statement. Under-qualification is a property the benchmark
   *measures* (`UNQUALIFIED_RELATION`), not a style the gold enforces.
3. **The question never names a column.** A question that says `l_shipdate` is
   testing transcription, not schema grounding.
4. **No gold may be degenerate.** An empty result, or a lone `0`, scores every
   model that returns nothing. Two tasks were caught by this rule during M4
   authoring: `tpch_nl_058`'s threshold was above the observed maximum, and
   `tpch_nl_021` asked for a gap TPC-H's generator never produces — receipt is
   always 1 to 30 days after shipment. Both thresholds are now measured values.
5. **Every constant in a `notes` field is measured**, not recalled. The counts
   quoted throughout were run against the loaded data.
6. **`notes` says what the task probes** — usually the piece of physical design
   or schema semantics that a raw schema dump cannot convey.
7. **No task is edited in response to how a system scored on it** (SPEC §11.5.1).

## Provenance

Questions are authored for this project. Where a task's gold derives from a
numbered TPC-H query, the id and `tags` record it (`tpch_q3_derived`,
`tpch_q9_derived`, and so on) so the lineage is checkable against the TPC-H
specification.
