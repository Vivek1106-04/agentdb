# What an agent needs to know about Delta that the schema does not say

An LLM writing SQL against a Databricks warehouse is given a schema. The schema
tells it the column names and their types. It does not tell it the one thing
that decides whether a query reads four files or four hundred thousand: **which
columns Delta can skip on, and whether the predicate it wrote can reach them.**

Neither does the error output, because there is no error. The query returns the
correct answer. It just reads the entire table to do it.

This document is what agentdb tells the model instead, why each fact is the one
that matters, and how to read it out of Databricks yourself. It assumes you know
what Delta and Photon are. Everything here is implemented in
`src/agentdb/adapters/databricks.py` and `src/agentdb/core/` and is exercised
against a live warehouse in `tests/e2e/test_databricks_live.py`.

---

## 1. Data skipping is per file, and it stops at column 32

Delta writes min/max (and null count) statistics into the transaction log for
each data file. A query with `WHERE l_shipdate >= '1995-01-01'` skips any file
whose recorded max `l_shipdate` is below that. This is the whole mechanism —
file-granularity, statistics-driven, and entirely dependent on the statistics
existing.

They do not always exist. `delta.dataSkippingNumIndexedCols` defaults to **32**,
and it counts columns in *schema order*, not in order of usefulness. Nested
fields count individually toward the limit. So on a table with 60 columns, a
filter on the 41st column skips nothing at all — not "skips less", *nothing*.
Every file is opened, every file is read, and the answer comes back correct.

The cost is silent in three places at once: the query succeeds, the plan does not
say "no statistics available", and the schema an agent was handed looks identical
whether the column is number 5 or number 45.

Two properties control it:

| Property | Effect |
|---|---|
| `delta.dataSkippingNumIndexedCols` | How many leading columns get statistics. Default 32. |
| `delta.dataSkippingStatsColumns` | An explicit column list. When set, **it wins entirely** — the ordinal rule no longer applies. |

agentdb reads both out of `information_schema` / table properties and applies
exactly that precedence (`PhysicalLayout.has_file_statistics`, in
`src/agentdb/adapters/models.py`):

```python
if self.stats_columns is not None:  # dataSkippingStatsColumns set
    return column in self.stats_columns
if self.stats_indexed_columns is not None:
    return ordinal <= self.stats_indexed_columns
```

When a filtered column fails that test, the model is told before it runs
anything, as a `STATS_NOT_COLLECTED` warning at CRITICAL severity:

> `event_properties` has no per-file statistics on `main.analytics.events`
> (Delta indexes only the first 32 columns in schema order), so filtering on it
> cannot skip any file.
>
> *Suggested rewrite:* filter on an indexed column as well, or ask the table
> owner to set `delta.dataSkippingStatsColumns`.

**The thing worth taking away even if you never use agentdb:** raising
`dataSkippingNumIndexedCols` or setting `dataSkippingStatsColumns` is **not
retroactive**. Files already written carry no statistics for the newly named
columns. Until those files are rewritten — `OPTIMIZE`, or any operation that
rewrites them — the property is a promise about future writes only. agentdb
ships that sentence verbatim in the recommendation it emits
(`NOT_RETROACTIVE`, in `core/advisor/databricks_advisor.py`), because a
recommendation that quietly does nothing for a week is worse than none.

---

## 2. `PushedFilters` versus `DataFilters` — where predicates actually get spent

This is the section to read.

`EXPLAIN FORMATTED` gives you a numbered tree and then one detail block per node.
The scan's detail block is where a predicate's fate is recorded, and the field
names distinguish three very different things:

- **`PartitionFilters`** — evaluated against partition *directories*. Whole
  partitions are eliminated before any file is considered. The cheapest place a
  predicate can land.
- **`PushedFilters`** / **`RequiredDataFilters`** — pushed into the read: matched
  against per-file statistics for skipping, and against Parquet row-group
  statistics inside a file. This is where data skipping happens.
- **`DataFilters`** — evaluated *after* the data is read, on the rows that came
  back. A predicate here prunes nothing. It only reduces what flows upward.

A predicate that appears only under `DataFilters` cost you the entire scan. That
is the fact the plan is telling you, and it is easy to miss because the query is
correct and the plan is enormous.

### The trap: two spellings for one idea

Here is a real scan block, captured from a Free Edition warehouse
(`tests/fixtures/databricks/explain_formatted.txt`):

```
(1) PhotonScan parquet samples.tpch.lineitem
Output [3]: [l_extendedprice#21946, l_returnflag#21949, l_shipdate#21951]
DictionaryFilters: [(l_shipdate#21951 >= 1995-01-01)]
Location: PreparedDeltaFileIndex [s3://.../tables/281c5907-...]
ReadSchema: struct<l_extendedprice:decimal(18,2),l_returnflag:string,l_shipdate:date>
RequiredDataFilters: [isnotnull(l_shipdate#21951), (l_shipdate#21951 >= 1995-01-01)]
```

There is **no `PushedFilters` line at all.** A non-Photon `FileScan` prints
`PushedFilters`; a `PhotonScan` prints `RequiredDataFilters` and nothing else. A
parser — or an engineer — that knows only the documented spelling reads every
Photon plan as having pushed no predicates, and concludes that a perfectly
well-pruned query is doing a full scan.

agentdb accepts both spellings, and says why in the code
(`core/plan_analyzer_databricks.py`):

```python
PUSHED_FILTER_KEYS = ("PushedFilters", "RequiredDataFilters")
DATA_FILTER_KEYS = ("DataFilters", "DictionaryFilters")
```

`DictionaryFilters` is the fourth thing, and it sits between the other two:
Photon evaluating a predicate against a Parquet **dictionary page** — narrower
than a statistics-based file skip, much cheaper than a row scan. Counting it as
a pushed filter overstates pruning; counting it as a post-read filter understates
it. It is classified with the data filters and named separately in the trace.

### What silently refuses to push down

A predicate wrapped in a function does not push. `WHERE year(l_shipdate) = 1995`
looks like a date filter, reads like a date filter to a model that learned SQL
from the internet, and prunes **nothing** — no partition elimination, no file
skipping. `WHERE l_shipdate >= '1995-01-01' AND l_shipdate < '1996-01-01'` is the
same question and prunes properly.

This is why agentdb's `MISSING_PARTITION_PREDICATE` rule for Databricks reads the
*plan* rather than the query text (`core/plan_rules_databricks.py`): the query
mentions the partition column either way. Only the plan knows whether anything
was pushed.

---

## 3. Liquid clustering is not a sort key, and the ClickHouse rule does not transfer

Both engines have a "physical ordering key". They work differently enough that
carrying advice across is actively harmful.

ClickHouse's primary index is **sparse**: one mark per granule (8192 rows by
default). Pruning works by binary-searching those marks, which is why the sort
key's *leading* column is decisive and why the ordering rule is **lowest
cardinality first** — long runs of equal values let whole granule ranges be
excluded at once. Lead with a high-cardinality column and matching rows scatter
across every granule.

Liquid clustering has no sparse-mark structure. It uses a Hilbert-curve
assignment of rows to files, so:

- There is no "leading column" whose absence disables everything. Clustering keys
  are **not** prefix-ordered the way a sort key is.
- Cardinality ordering is the wrong heuristic. What matters is which columns are
  actually filtered on, and how selective those filters are.
- It is incremental: `OPTIMIZE` clusters new data without rewriting the whole
  table, which is precisely what a sort key change cannot do.

agentdb's two advisors are deliberately asymmetric for this reason. The
ClickHouse advisor ranks sort-key candidates by cardinality; the Databricks
advisor ranks clustering candidates by **filter frequency first, then
selectivity**, mined from the query history. The refusal to port §9.1's rule is
written into the module docstring so nobody "fixes" the inconsistency later.

The corresponding warning is `CLUSTERING_KEY_UNUSED`: none of the filtered
columns are in the clustering key, so the scan opens every file the partition
predicate left behind.

---

## 4. `EXPLAIN COST` without `ANALYZE` is worse than no plan at all

`EXPLAIN COST` prints row-count and size estimates. On a table that has never
been analyzed, those estimates are derived from file sizes and are frequently
off by orders of magnitude — and they look exactly as authoritative as real ones.

Look at the bottom of the same captured plan:

```
== Optimizer Statistics (table names per statistics state) ==
  missing = lineitem
  partial =
  full    =
```

`samples.tpch.lineitem` — a table shipped by Databricks in every workspace — has
**no** optimizer statistics. Any cost number for it is a guess.

Feeding that guess to a model is worse than feeding it nothing, because a model
handed a number treats it as measured and reasons from it. So:

- The adapter gates `EXPLAIN COST` behind a capability flag
  (`Capability.COST_ANNOTATED_PLAN`) and does not offer cost numbers when the
  table's statistics state is `missing`.
- The default explain mode is `EXPLAIN FORMATTED`, which reports structure and
  pushed filters — facts, not estimates.
- `ANALYZE TABLE … COMPUTE STATISTICS` is never issued by agentdb. It writes, and
  the benchmark's principal holds `SELECT` and nothing more (SPEC §13.3). It is
  emitted as a *recommendation*, for a human to run.

One more sharp edge, learned live: **Databricks answers `EXPLAIN` over an invalid
query with success.** The statement returns HTTP 200 and the error is a string
inside the result set. Code that checks the statement status and not the payload
treats a syntax error as a valid empty plan.

---

## 5. Files read, files total, and why the ratio is often absent

The obvious efficiency metric is "fraction of files pruned". Getting it honest
took more care than expected.

The plan says how many files were **read**. It does not say how many exist. The
total comes from `DESCRIBE DETAIL`, which is a separate call, and the ratio is
computed only when *both* numbers were actually measured:

```python
measured = [scan for scan in scans if scan.files_total and scan.files_selected is not None]
```

The comment above that line records why it is written defensively: a Photon plan
frequently carries no file counts at all, and an unreported `files_selected`
treated as zero produces a summary reading **"0.0% of files read after
pruning"** — a plan that measured nothing, reporting perfect pruning. That was
observed on the first live run.

Compare with ClickHouse, where `EXPLAIN indexes=1` gives you granule counts
before the query runs. On Databricks the file counts are only available *after*
execution, from the query history API. That asymmetry is why the two engines'
`PlanSummary` objects carry different pruning units (`granule` versus `file`) and
why cross-engine efficiency comparisons in the report are made in bytes read,
which both engines measure the same way.

A note on the query history, since it cost a day: **do not join to
`system.query.history`.** On a Free Edition workspace it was measured 1,514 to
23,290 seconds behind the warehouse clock, so a benchmark attributing through it
attributes nothing. The history **API**, queried by `statement_id`, answered
immediately on every probe. The Statement Execution API returns that
`statement_id` synchronously, which is why agentdb prefers it over the DB-API
connector: attribution by primary key rather than by string-matching a comment.

---

## 6. Shadow validation: what "measured" means here

Neither engine has a hypothetical-index facility. There is no `hypopg` for Delta.
So a recommendation like "cluster by `customer_id`" is, by default, an argument —
not a measurement.

agentdb can upgrade it to a measurement by building the proposed layout on a
**bounded sample** of the table in a scratch schema, re-planning the query
against it, and reading the pruning out of the new plan. The estimate is then
labelled `measured` and carries the sample fraction it was taken at.

The guard rails matter more than the feature:

- **Opt-in.** Nothing runs unless `AGENTDB_ALLOW_SHADOW` is set. Shadow tables
  cost real money on Databricks in a way they do not in a local container.
- **Namespaced and capped.** Every created table carries an `__agentdb_shadow`
  marker plus a run token, bounded by `SHADOW_TABLE_MAX_ROWS`.
- **Dropped in a `finally`** — including when the plan read fails.
- **Reaped on startup anyway,** because a `finally` does not run when the process
  is killed. That is what the marker is for, and there is a chaos test that kills
  the process mid-validation and asserts the orphan is found.
- **A separate write channel.** Read-only is enforced at the connection level, so
  validation cannot borrow the read-only connection — it must be handed a channel
  the operator explicitly configured. Where none is configured, validation does
  not happen and the estimate stays labelled as an estimate.

Extrapolating from a sample to the full table is an assumption, and it is
recorded as one: every shadow-validated recommendation states the fraction it
measured, and `docs/methodology.md` lists it under threats to validity.

---

## 7. Reading this out yourself, without agentdb

```sql
-- Which columns can be skipped on at all?
SHOW TBLPROPERTIES main.analytics.events;
--   look for delta.dataSkippingNumIndexedCols and delta.dataSkippingStatsColumns

-- Column ordinals, because the limit counts in schema order:
SELECT column_name, ordinal_position
FROM main.information_schema.columns
WHERE table_schema = 'analytics' AND table_name = 'events'
ORDER BY ordinal_position;

-- Layout, file count, and average file size:
DESCRIBE DETAIL main.analytics.events;

-- Where did each predicate land?
EXPLAIN FORMATTED
SELECT ... ;
--   in the scan's detail block, read:
--     PartitionFilters      -> whole partitions eliminated
--     PushedFilters / RequiredDataFilters -> file skipping happened here
--     DataFilters / DictionaryFilters     -> evaluated after the read
```

If a column you filter on has an ordinal above `dataSkippingNumIndexedCols` and
is absent from `dataSkippingStatsColumns`, your predicate cannot skip a single
file, no matter how selective it is. That is the fact this whole document exists
to hand to a model before it writes the query rather than after.

---

## Corrections

If any of this is wrong, it is wrong in public and in a file with a git history.
Open an issue or a PR and it gets fixed with credit. Measurements against
Databricks are published under the disclosure rules in `SPEC.md` §15: the vendor
sees the methodology and the numbers privately, with a stated response window,
before anything goes out.

See also: `docs/clickhouse-advisor.md` for the same treatment of the other
engine, and `docs/methodology.md` for how any of this is measured.
