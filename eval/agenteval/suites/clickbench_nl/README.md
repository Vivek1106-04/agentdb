# clickbench_nl

Natural-language questions over the ClickBench `hits` table, with the SQL that
answers them as gold. ClickBench ships 43 SQL queries and no questions; this
suite supplies the missing half, so an agent stack can be measured on the same
data an engine benchmark uses.

- **Table:** `hits` — 105 columns, ~99,997,497 rows, sorted by
  `(CounterID, EventDate, UserID, EventTime, WatchID)` and unpartitioned, per
  ClickBench's own `create.sql`. Load it with `make load-clickbench`.
- **Engine:** ClickHouse. TPC-H (`tpch_nl`) carries the cross-engine arm.
- **Provenance:** task ids keep the ClickBench query number
  (`clickbench_nl_007` ↔ ClickBench Q7), so any translation can be checked
  against [ClickBench](https://github.com/ClickHouse/ClickBench).

## Status

**57 of ~100 tasks. Gold is not yet frozen** — the hashes require the full
100-part table, and `gold.lock.yaml` lands with it.

| file | ids | source |
|---|---|---|
| `aggregates.yaml`, `grouping.yaml`, `visitors.yaml` | 000–020 | ClickBench Q0–Q20 |
| `text.yaml` | 021–028 | ClickBench Q21–Q28 |
| `sessions.yaml` | 030–042 | ClickBench Q30–Q42 |
| `layout.yaml` | 043–062 | authored for this project |

ClickBench-derived translation is complete: every one of the 43 originals is
either present or excluded for a stated reason (below). Ids at 043 and above are
authored and carry no `clickbench_original` tag, which is what makes the
contamination comparison in `docs/methodology.md` possible.

### Originals deliberately excluded

Q17 is Q16 with no `ORDER BY`, so its ten rows are whichever ten the engine
produced. The others have no honest natural-language form:

| query | why |
|---|---|
| Q29 | sums `ResolutionWidth + N` across ninety columns — it measures expression throughput, not a question anyone asks |
| Q34 | selects the literal `1` beside `URL` and groups by it; in words it is just Q33 |
| Q35 | groups by `ClientIP` and `ClientIP - 1`, `- 2`, `- 3` — the same artifact |
| Q40, Q41 | filter on a raw `RefererHash` / `URLHash` constant; a question cannot name a hash value and stay a question |

Where ClickBench pages with `OFFSET` into an order that has ties (Q38, Q39,
Q42), the offset is dropped rather than reproduced — rows 1001–1010 of a
nondeterministic order are not a gold result. Each deviation is named in the
task's `notes`.

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
