# Observed Databricks responses

Captured by `scripts/verify_databricks.py` from a live workspace. These are what
the engine actually printed, as opposed to what the documentation says it
prints — a distinction that cost four defects on the first run (see below).

| | |
|---|---|
| **Captured** | 2026-08-15 |
| **Runtime** | DBSQL `2026.20` (`current_version().dbsql_version`) |
| **Workspace tier** | Free Edition, serverless SQL warehouse |
| **Table** | `samples.tpch.lineitem` — 10 files, 753,837,113 bytes, no clustering key, no partitioning |

SPEC §12 asks that these be committed with the runtime recorded and refreshed on
a schedule: the Databricks CI tier cannot run on fork PRs, so fixture-based unit
coverage is the only thing standing between a renamed column and a silently
wrong benchmark number. Re-run the script after any workspace upgrade and read
the diff.

## What the first live run corrected

1. **`PhotonScan` prints no `PushedFilters`.** It prints `RequiredDataFilters`
   and `DictionaryFilters` instead. The parser, written from the documented
   non-Photon `FileScan` shape, read every Photon plan as having pushed nothing.
2. **A Photon plan carries no file counts.** No `number of files read`, no
   `size of files read`. The summarizer treated the absent numerator as zero and
   reported `0.0% of files read after pruning` — a plan that measured nothing
   claiming it pruned everything. Pruning evidence on Databricks has to come
   from post-hoc metrics (`system.query.history.pruned_files` / `read_files`),
   not from `EXPLAIN`.
3. **`AdaptiveSparkPlan` wraps every plan** and never carries the `Photon`
   prefix, so Photon coverage read 93% on a plan that was entirely vectorized.
4. **`operationParameters` is a JSON string, not a map**, so Z-ORDER mining
   found nothing on every table.

## Where pruning evidence actually comes from

Point 2 above said pruning evidence has to come from post-hoc metrics. Finding
out *which* post-hoc source works took four probes, and the obvious one is wrong.

| Source | Answers by | Lag | Verdict |
|---|---|---|---|
| `EXPLAIN FORMATTED` | — | none | **carries no file counts on a Photon plan at all** |
| `system.query.history` | `statement_id` | **1,514 s and 23,290 s**, two runs | useless for the statement that just ran |
| Query History **API** | `statement_id` | **t+0s, `is_final=True`** | this is the one |

So `system.query.history` — the source SPEC §8.2 names — stays the workload
miner, and measured pruning is read from the history API instead. `make check`
cannot notice if that ever changes, so `scripts/verify_databricks.py` measures
the lag on every run and `tests/e2e` asserts it is still too large to use.

Three more facts, each of which would otherwise produce a wrong number rather
than an error:

1. **Zero files read has three unrelated meanings.** The result cache answered
   (`result_from_cache: true`); Delta metadata answered, as a bare `count(*)`
   does without opening a file; or the warehouse reported no metrics. Each one
   yields `0 read of 0 considered`, which a ratio renders as flawless pruning.
   `QueryMetrics.measured` exists to refuse all three.
2. **A comment does not defeat the result cache.** This was found the expensive
   way: the metrics check carried a nonce as a trailing `-- {hex}` and the second
   run came back cached with every counter zero. Comments are normalized out of
   the cache key; the nonce now lives in a predicate.
3. **File pruning is not the whole story.** On an aggregate that read all ten
   files of `lineitem` — nothing pruned, ratio 1.0 — only **21–23% of the bytes
   in those files** were fetched. That is column projection and row-group
   skipping, a different mechanism, so it is reported beside the file ratio and
   never folded into it.

The API and the system table spell the same quantities differently:
`read_files_count` / `pruned_files_count` / `read_files_bytes` against
`read_files` / `pruned_files` / `read_files_bytes`. Two vocabularies, one
warehouse.

Two further observations worth keeping, neither of them defects:

* `information_schema.columns.ordinal_position` is **0-based** on this runtime,
  not 1-based as the SQL standard specifies. Nothing reads the raw value —
  ordinals are derived positionally from the ordinal-ordered column list — but
  anything that starts reading it must normalize.
* `DESCRIBE DETAIL` has no `numRows` column here; the row count lives in a
  `statistics` struct which is `{}` on a table nobody has analyzed. `approx_rows`
  is therefore `None` on `samples.tpch`, and the rules that need a row count stay
  silent rather than firing on a zero that means "not measured".
