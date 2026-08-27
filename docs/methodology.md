# Methodology and threats to validity

How the numbers in [`results/REPORT.md`](../results/REPORT.md) are produced, and
every reason they might be wrong that I know of. If you find one that is not
here, open an issue — a threat I did not think of is more useful to me than
agreement.

## What is measured

**Execution accuracy (EX).** A task is correct when the result set of the model's
query hashes identically to the result set of the committed gold query, run
against the same data. Column names are not compared; column *order* is, and row
order is compared only when the gold query has a top-level `ORDER BY`. A query
that returns the right rows in an order the question did not ask for is correct,
because SQL says it is.

**EX@1** is the same measure restricted to the first executed attempt, before any
self-correction. The gap between EX and EX@1 is what retries bought.

**Valid SQL** counts attempts the engine accepted, whatever they returned. An arm
can write valid SQL that answers the wrong question, and the two columns
separate those failures.

**Tokens and context bytes** are counted from the provider's own usage
accounting, never estimated. Context bytes are the grounding payload this harness
sent; where a system carries context the harness did not choose — Claude Code's
scaffolding, a managed service's system prompt — that is reported separately and
excluded from the arm's own totals, because it is not the arm's grounding.

Wall-clock time is recorded per attempt and **not** used to compare arms. See
"single-instance timings" below.

## How a run is constructed

Every cell is one (task, arm, model, seed). Arms are built from
[`eval/providers.yaml`](../eval/providers.yaml) and
[`eval/servers.yaml`](../eval/servers.yaml), both committed, and every arm's
effective configuration is hashed into a `config_fingerprint` recorded on each
trace. "A2 scored 61%" is not a claim this project makes; "A2 at fingerprint
`sha256:…` scored 61%" is.

An arm is a *provider pointed at an engine*. Each entry names the engine it
grounds against, and an arm with no entry for the engine under measurement is
refused rather than substituted — grounding a Databricks run in ClickHouse's
tables would complete, report numbers, and be nonsense.

Gold results are frozen with `make freeze-gold`, which runs every gold query
against the loaded data and commits the result hashes to `gold.lock.yaml`. A
later run whose gold no longer hashes the same fails loudly instead of grading
against drifted data. Task authoring for `clickbench_nl` and `tpch_nl` was
completed and git-timestamped before any Family S measurement, so no task was
written after seeing a system's answers.

## Statistics

- Each cell is repeated `N_SEEDS` times (default 5) at temperature > 0.
- Accuracy carries a 95% confidence interval from a bootstrap with 10,000
  resamples.
- Arms are compared **paired**: same tasks, same seeds, on the cells both arms
  actually ran, with McNemar's exact test on per-task correctness. An unpaired
  comparison between arms that ran different subsets is not reported at all.
- Every per-task trace is committed to `results/raw/*.jsonl`: the prompt, every
  emitted query, the engine's response, the error class, timings and tokens. Any
  single number in the report can be audited back to the records that produced
  it.

## Threats to validity

### Contamination

ClickBench's queries have been public for years and are plausibly in the
training data of every model measured here. 37 of `clickbench_nl`'s 100 tasks are
derived from them and carry the `clickbench_original` tag; the other 63 were
authored for this project.

The report splits every arm's accuracy along that line and prints the gap. A
large positive gap — better on the public questions than on the authored ones —
is a fact about memorization rather than about grounding, and it qualifies every
other number in the report. It is published whether or not it is flattering; the
check exists precisely because it can weaken the headline.

What the check cannot rule out: the *schema* is public too. A model that has
never seen a ClickBench query may still know that `hits` has a `CounterID`
column, and no split in this suite separates that from grounding.

### Gold-annotation errors

Wrong gold answers are a documented, systemic problem in text-to-SQL benchmarks —
the CIDR 2026 analysis of annotation errors in BIRD and Spider is directly
relevant, and there is no reason to believe this suite is exempt.

Mitigations: two-pass authoring, gold result hashes committed and re-verified
against the live data on every freeze, and an issue template for disputing any
task. If you think a gold query answers a different question than its prompt
asks, that is a bug and I want it filed. Corrections are credited by name.

What this does not fix: a question whose *wording* is ambiguous will be graded
against one reading of it. Where I noticed ambiguity I rewrote the question; the
ones I did not notice are, by definition, not in this list.

### Single-instance timings do not generalize

ClickHouse here is one container on one machine; the Databricks side is one
serverless warehouse whose size, cache state and neighbours I do not control. No
latency claim in this project should be read as a property of either engine.

That is why the plan layer reports **bytes read and granules or files pruned** —
engine-intrinsic quantities that mean the same thing on a laptop and on a
cluster — alongside any wall-clock figure, and why the advisor's effect estimates
never mention latency at all.

### The advisor is measured against a representative workload, not yours

Arm `A6_full` needs to know what a table is normally asked. In a deployment that
comes from the engine's query log; on a benchmark instance that log holds this
project's own gold executions, so mining it would hand the advisor the answers.
A6 therefore reads a committed reference workload of third-party query shapes —
ClickBench's published queries and TPC-H's — whose sha256 is part of the arm's
fingerprint.

The honest consequence: A6 measures the advisor against a *plausible* workload
rather than a real operator's. An advisor that helps here might help less on a
workload shaped differently.

### Shadow validation measures a sample

A recommendation marked `measured` was validated by building a sampled copy of
the relation with the proposed design and reading its plan. The pruning *ratio*
transfers; the absolute counts do not, because a sample has fewer of everything.
Every measurement records the sample fraction it used.

### Family S arms are not all controllable

Some systems under test pick their own model, their own temperature, or their own
retry policy. Where a system does not permit model control the report footnotes
it rather than silently comparing it to arms that do. Claude Code additionally
carries 16k–30k tokens of its own scaffolding and whatever instruction files the
operator's machine supplies; that is disclosed on every row, and its numbers
should be read as "this product on this machine", not "this model".

### A managed service's accuracy is partly a fact about its setup

ClickHouse Agents and Databricks AI/BI Genie are configured, not just connected:
a Genie space carries a table scope, instruction text, and curated example
queries, and its accuracy is a function of all three. Measuring one setup and
calling it "Genie's accuracy" would be the unfair benchmark this project exists
not to be. So Genie is measured in two configurations — `S4a_genie_minimal`,
scoped to the suite's tables with no instructions and no examples, and
`S4b_genie_curated`, a realistic deployment — and the report prints an
"incomplete pair" warning above the table if only one of them ran. Neither
number should be quoted without the other.

Everything each service was given is committed: `eval/managed.yaml` holds the
configurations, and every run copies them to `results/raw/<run-id>.managed.json`
beside its traces. No curated example may be a gold query from any suite or a
paraphrase of one; the check runs before the arm is built and fails the run, and
a unit test re-runs it against every shipped suite on every commit.

These arms select their own models, so their rows are never a controlled model
comparison against an agentdb-on-Opus-5 row. Where a service answers in prose
without writing a query, that cell is scored incorrect with `error_class`
`declined` and counted in the leaderboard's own `declined` column, because a
system that abstains is doing something different from one that guesses wrong.
What is graded is the SQL the service produced, re-executed through the
harness's own read-only connection — never the service's own formatted reply.

### Beta products move

Systems in public beta change between releases. Every Family S row is
date-stamped, and any published reference to one should be re-run first — `make
bench` is the whole procedure. Vendors are welcome to dispute a number; the
rules of engagement are private notification first, a two-week response window,
and corrections credited by name.

## Reproducing this

```bash
docker compose -f docker/docker-compose.yml up -d --wait
make load-clickbench CLICKBENCH_PARTS=100     # ~100M rows
make load-tpch                                # SF5, matching samples.tpch
export ANTHROPIC_API_KEY=...
make bench-quick                              # five tasks, one seed
make report
```

Or point `make bench-quick-dbx` at a free Databricks workspace and reproduce the
warehouse half with no local infrastructure at all.
