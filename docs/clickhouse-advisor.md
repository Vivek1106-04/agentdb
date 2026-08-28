# What an agent needs to know about ClickHouse that the schema does not say

A model handed `CREATE TABLE hits (CounterID UInt32, EventDate Date, UserID
UInt64, ...)` knows the columns. It does not know that `CounterID` is the leading
sort-key column, that filtering on it prunes granules, and that wrapping it in a
function throws that away entirely.

The measured cost of exactly that mistake, on ClickBench's 100M-row `hits` table:

```sql
-- what a model writes with a schema dump
SELECT count() FROM agentdb.hits WHERE toString(CounterID) = '62'

-- what it writes when told CounterID leads the sort key
SELECT count() FROM agentdb.hits WHERE CounterID = 62
```

Same answer — 738172 both times. **255 KB read against 64.0 KB, 4.0x**, measured
on a live 99,997,497-row instance; that is what `make demo` prints, and the
number comes out of the engine rather than out of this paragraph.

`EXPLAIN indexes = 1` says exactly where it went:

```
                    toString(CounterID) = '62'            CounterID = 62
Condition:          (toString(CounterID) in ['62','62'])  (CounterID in [62, 62])
Parts:              4/4                                   1/4
Granules:           108/12366                             92/12366
Search Algorithm:   generic exclusion search              binary search
Ranges:             18                                    2
```

Worth being precise about what broke, because it is not what people expect: the
index does **not** stop working. ClickHouse falls back to *generic exclusion
search* and still prunes to 108 granules. What it loses is locality — every part
instead of one, 18 disjoint ranges instead of 2 — and that is where the 4x lives.
On a cold mark cache the first run reads 575 KB (9.0x); the grounded side stays
at 64.0 KB either way.

This document is the ClickHouse half of what agentdb tells a model, and how to
read the same facts yourself. Companion to `docs/databricks-grounding.md`;
everything is implemented in `src/agentdb/adapters/clickhouse*.py` and
`src/agentdb/core/`.

---

## 1. The primary index is sparse, and that changes the ordering rule

ClickHouse's primary index is not a B-tree over rows. It stores **one mark per
granule** — 8192 rows by default (`index_granularity`) — holding the sort-key
values at that granule's boundary. A query with a predicate on the sort key
binary-searches those marks and reads only the granule ranges that can contain
matches.

Three consequences follow, and the third is the one people get wrong.

**It is prefix-ordered.** The key `(CounterID, EventDate, UserID)` is searched
left to right. A filter on `EventDate` alone, with nothing on `CounterID`, prunes
nothing at all — every granule is a candidate. This is `SORT_KEY_PREFIX_SKIPPED`,
and agentdb rates it CRITICAL rather than WARNING because the query *looks*
well-filtered:

> the filter reaches `EventDate` but not the leading sort-key column
> `CounterID`, so no granules can be pruned.
>
> *Suggested rewrite:* add a predicate on `CounterID` if the question allows one,
> even a wide range.

"Even a wide range" is doing real work in that sentence. `CounterID > 0` is
useless as a filter and enormously useful as a pruning hint, because it lets the
binary search start somewhere.

**Pruning is granule-granular, not row-granular.** One matching row in a granule
means the whole 8192-row granule is read. A predicate that matches 0.01% of rows
scattered uniformly prunes nothing, while the same selectivity clustered into a
few granules prunes almost everything. Selectivity alone does not predict cost;
*locality* does.

**Therefore: order the key lowest cardinality first.** This is the rule most
often stated backwards, because row-store instinct says "most selective column
first". A low-cardinality leading column produces long runs of equal values, so
whole ranges of granules can be excluded by their marks. A high-cardinality
leading column scatters matching rows across every granule, and the sparse index
has nothing to bite on.

agentdb's ClickHouse advisor ranks `ORDER BY` candidates by that rule. Its
Databricks counterpart deliberately does **not** — liquid clustering has no
sparse-mark structure to exploit, so porting the rule would be advice-shaped
noise. The refusal is written into both module docstrings so nobody later
"harmonizes" them.

---

## 2. `EXPLAIN indexes = 1` — and the 25.9 setting that makes it lie

The plan tells you what the index actually did:

```sql
EXPLAIN indexes = 1, json = 1
SELECT count() FROM agentdb.hits WHERE CounterID = 62
SETTINGS use_query_condition_cache = 0, use_skip_indexes_on_data_read = 0;
```

Read `Granules` — selected over total — out of the `Indexes` block, along with
`Parts` and `Search Algorithm`. Those are the whole efficiency story: 92 of
12,366 granules across 1 of 4 parts by binary search is a working key; 12,366 of
12,366 is a full scan wearing a `WHERE` clause. In `json = 1` output the same two
integers arrive as `Initial Granules` and `Selected Granules`, which is what the
plan IR reads.

**Both settings are load-bearing on ClickHouse ≥ 25.9.** Without them the
`indexes` output reflects cached condition results and skip-index behaviour at
data-read time rather than what this query's index actually pruned — so the
numbers a plan-grounded agent reasons from would be quietly wrong. agentdb keeps
the exact statement text in one place, `adapters/clickhouse_sql.py`, precisely so
a reviewer can check the claim without reading an adapter:

```python
EXPLAIN_SETTINGS: Final = "use_query_condition_cache = 0, use_skip_indexes_on_data_read = 0"
"""The 25.9 footgun. Without both disabled the ``indexes`` output is meaningless,
so the pruning evidence the plan IR is built on would silently be wrong."""
```

This is the sharpest difference from Databricks. Here the pruning evidence is
available **before** the query runs, at zero cost. On Delta, file counts exist
only after execution, from the query history. That asymmetry is why cross-engine
efficiency in the report is compared in bytes read — the one quantity both
engines measure the same way.

---

## 3. Skip indexes: when they earn their place, and when they are theatre

A data-skipping index stores a summary per *group of granules*
(`GRANULARITY n`) and lets the engine skip that group when the summary proves no
match. Types worth knowing:

| Type | Skips on | Good for |
|---|---|---|
| `minmax` | range | Columns correlated with the sort key — a timestamp written monotonically. |
| `set(N)` | membership | Low-cardinality columns with fewer than N distinct values per block. |
| `bloom_filter(fpp)` | membership | High-cardinality equality: ids, UUIDs, hashes. |
| `tokenbf_v1` / `ngrambf_v1` | substring | `LIKE '%…%'` on text. |

Three rules the advisor encodes:

- **A skip index on a column uncorrelated with physical order is theatre.** If
  matching values appear in every block, no block can be skipped, and you have
  paid write amplification and merge cost for nothing. The index does not make
  data local; it exploits locality that already exists.
- **Granularity is a real knob.** `GRANULARITY 1` means one summary per granule:
  finest skipping, largest index. Higher values coarsen both. The advisor
  proposes a value and states the trade rather than defaulting silently.
- **Bloom false-positive rate is a cost, not a detail.** agentdb proposes
  `BLOOM_FPP = 0.01` — one in a hundred granule-groups read for nothing — and
  says so in the recommendation.

---

## 4. Projection versus skip index

A **projection** is a second physical copy of the data, stored inside the table,
with its own sort order and optionally pre-aggregated. The optimizer picks it
when it covers the query. Choosing between the two:

**Use a skip index when** the query filters on a column and you want to read
*fewer* granules of the same data. It is cheap to maintain and costs little
storage.

**Use a projection when** the query has a fundamentally different *shape* — a
different sort order, or a `GROUP BY` that could be answered pre-aggregated. A
projection can turn a full aggregation into a lookup, which no skip index can do.
It costs a full second copy of the projected columns, and it is written on every
insert.

The advisor emits `PROJECTION_AVAILABLE_UNUSED` when a table already has a
projection covering the query's `GROUP BY` columns and the plan read the base
table anyway — a case where the fix costs nothing at all, because the structure
is already there and the query merely failed to match it.

A caution the advisor states rather than hides: on a large table, adding an
`ORDER BY` or a projection is a rewrite, not a metadata change. Unlike Delta's
liquid clustering, which clusters new data incrementally, a ClickHouse sort-key
change means building a new table and moving the data. The recommendation says
so, because the cost of *applying* it belongs next to the benefit of having it.

---

## 5. The rest of the warning set

Everything below is deterministic — plan facts plus layout facts plus query
shape, no model in the loop. Each warning cites the evidence it fired on, and
stays silent when that evidence is missing. A warning nobody can check is worse
than no warning.

| Code | Fires when |
|---|---|
| `SORT_KEY_UNUSED` | No filtered column appears in the sort key at all. |
| `SORT_KEY_PREFIX_SKIPPED` | A later key column is filtered but the leading one is not. CRITICAL. |
| `MISSING_PARTITION_PREDICATE` | The table is partitioned and the query constrains none of it. |
| `PROJECTION_AVAILABLE_UNUSED` | A matching projection exists; the plan read the base table. |
| `HIGH_CARD_GROUP_BY` | `GROUP BY` on a column whose sampled cardinality makes the state enormous. |
| `NULLABLE_IN_KEY` | A `Nullable` column in the sort key — the extra indirection is rarely intended. |
| `SELECT_STAR_WIDE` / `NO_LIMIT_UNBOUNDED` | Column-store hygiene: a wide `SELECT *` reads every column's data. |
| `JOIN_ORDER_SUSPECT` | The plan's build side looks like the larger relation. |

Cardinality figures come from sampled profiling, not from `count(DISTINCT …)`
over 100M rows — the profile records the sample fraction it used, and the
methodology lists sampling as a threat to validity rather than burying it.

---

## 6. Reading it out yourself

```sql
-- The layout the schema dump omits:
SELECT sorting_key, partition_key, primary_key
FROM system.tables WHERE database = 'agentdb' AND name = 'hits';

-- What each column actually costs to read:
SELECT name, type,
       formatReadableSize(data_compressed_bytes)   AS compressed,
       formatReadableSize(data_uncompressed_bytes) AS raw
FROM system.columns
WHERE database = 'agentdb' AND table = 'hits'
ORDER BY data_compressed_bytes DESC;

-- Existing skip indexes and projections:
SELECT name, type, expr, granularity FROM system.data_skipping_indices
WHERE database = 'agentdb' AND table = 'hits';
SELECT name, query FROM system.projections
WHERE database = 'agentdb' AND table = 'hits';

-- Did the index fire?
EXPLAIN indexes = 1, json = 1
SELECT count() FROM agentdb.hits WHERE CounterID = 62
SETTINGS use_query_condition_cache = 0, use_skip_indexes_on_data_read = 0;
--   compare "Initial Granules" against "Selected Granules"

-- What it actually read, after the fact:
SELECT read_rows, formatReadableSize(read_bytes), query_duration_ms
FROM system.query_log
WHERE type = 'QueryFinish' AND query_id = '…'
ORDER BY event_time DESC LIMIT 1;
```

If `Selected Granules` equals `Initial Granules`, the `WHERE` clause did nothing
for you. That is the number to look at first, and it is available before you
spend anything running the query.

---

## 7. Recommendations are text until a human says otherwise

Everything the advisor produces is DDL as **text**. agentdb never executes it.
Applying a schema change requires an explicit elicited confirmation, and the
read-only role the benchmark runs under could not execute it regardless —
read-only is enforced at the connection, not by inspecting SQL strings. A system
that let a model's output decide whether an operation was safe would have no
safety property at all.

Where a recommendation is validated rather than argued, it is built on a bounded
sample in a namespaced shadow table, re-planned, and labelled `measured` with the
sample fraction attached. The mechanics — opt-in flag, row cap, `finally` drop,
startup reaper, chaos test — are shared with the Databricks side and described in
`docs/databricks-grounding.md` §6.

---

## Corrections

Wrong in public, in a file with a git history. Issues and PRs get fixed with
credit. ClickHouse Agents and `mcp-clickhouse` measurements follow the disclosure
rules in [`disclosure.md`](disclosure.md): methodology and numbers privately to
the vendor first, with a stated response window.

See also `docs/databricks-grounding.md` and `docs/methodology.md`.
