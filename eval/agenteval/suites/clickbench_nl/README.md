# clickbench_nl

Natural-language questions over the ClickBench `hits` table, with the SQL that
answers them as gold. ClickBench ships 43 SQL queries and no questions; this
suite supplies the missing half, so an agent stack can be measured on the same
data an engine benchmark uses.

- **Table:** `hits` — 105 columns, ~99,997,497 rows, `ORDER BY (CounterID, EventDate, UserID)`.
- **Engine:** ClickHouse. TPC-H (`tpch_nl`) carries the cross-engine arm.
- **Provenance:** task ids keep the ClickBench query number
  (`clickbench_nl_007` ↔ ClickBench Q7), so any translation can be checked
  against [ClickBench](https://github.com/ClickHouse/ClickBench).

## Status

**20 of ~100 tasks.** This is the ClickBench-derived half. The authored half —
questions with no public SQL counterpart, exercising sort-key-hostile filters,
partition-predicate omission, wide `SELECT *`, and nullable join keys — is what
makes the contamination check possible and lands next.

## Rules this suite follows

**Gold must be deterministic.** ClickBench's top-N queries order by a measure
alone, so tied rows come back in whatever order the engine produced and the
tenth row is not a fact. Gold here appends the grouping keys to `ORDER BY`,
which fixes an order without changing which rows are in the answer. Those tasks
carry the `tiebroken` tag. ClickBench Q17 is omitted entirely: it is Q16 with no
`ORDER BY` at all, and a benchmark cannot grade an answer that has no gold.

**Questions are written from the schema, not from the SQL.** Where the original
query's behaviour is not implied by an English reading — Q16 counting hits with
an empty search phrase, Q20 matching case-sensitively — the question says so.
An ambiguous question grades a coin flip.

**Sentinels are the point.** `AdvEngineID = 0`, `MobilePhoneModel = ''`, and
`MobilePhone` as a vendor code are the cheapest tests of whether a system has
real schema semantics or a plausible guess. `notes` on each task records what
is being probed.

**Contamination is expected here.** These queries are public and likely in
training data. That is why the suite is tagged `clickbench_original` versus
authored, and why `docs/methodology.md` reports A0 accuracy on each half
separately. The gap is a result, not an embarrassment.

## Task format

See `agenteval/tasks.py`. `gold_result_hash` is committed once a suite has been
run against a verified copy of the data; it is the tripwire that turns silent
gold drift into a failed run.

## License

Apache-2.0, same as the harness. ClickBench is Apache-2.0; the question text is
original work.
