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

Two further observations worth keeping, neither of them defects:

* `information_schema.columns.ordinal_position` is **0-based** on this runtime,
  not 1-based as the SQL standard specifies. Nothing reads the raw value —
  ordinals are derived positionally from the ordinal-ordered column list — but
  anything that starts reading it must normalize.
* `DESCRIBE DETAIL` has no `numRows` column here; the row count lives in a
  `statistics` struct which is `{}` on a table nobody has analyzed. `approx_rows`
  is therefore `None` on `samples.tpch`, and the rules that need a row count stay
  silent rather than firing on a zero that means "not measured".
