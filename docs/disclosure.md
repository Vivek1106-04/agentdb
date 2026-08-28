# Vendor disclosure

Three of the arms in this benchmark measure another company's shipped product:
`S3_clickhouse_agents` (ClickHouse Agents) and `S4a_genie_minimal` /
`S4b_genie_curated` (Databricks AI/BI Genie). Publishing an accuracy number for
someone else's product is not the same act as publishing one for my own code,
and it does not get the same process.

These are the rules those measurements are published under. They are stated here
rather than left implicit so a reader can hold the project to them.

## The sequence

1. **Read the beta terms first.** Both products are betas. Before a run, confirm
   the terms permit publishing benchmark results at all. `make bench-managed` is
   a separate target from `make bench` for exactly this reason — the vendor arms
   are never a silent step of the default matrix.

2. **Contribute to a product's ecosystem before measuring it.** Reading a
   codebase well enough to file a real fix is also how you find out whether you
   are measuring it fairly.

3. **Send the methodology and the numbers privately, before publication, with a
   stated response window.** To every vendor whose product appears in Family S.
   Not a courtesy note after the fact — the full method, the configurations, and
   the results, early enough that a correction can still change what gets
   published.

4. **Ask the questions only the vendor can answer.** For Genie specifically:
   whether the curated space configuration is representative of a real
   deployment. That number is a function of the curation, and I am guessing at
   what realistic looks like. If it is misconfigured I would rather fix it than
   publish it.

5. **Publish after the window closes**, incorporating any corrections, and
   credit each correction by name.

## Why

A benchmark that publishes cold makes an adversary out of the vendor. One that
shares first makes a collaborator, and a collaborator can tell you that your
configuration was wrong before the number is in front of anyone else.

There is also a self-interested reason, and it is the stronger one: a vendor who
has seen the methodology and did not dispute it makes the result *more*
believable, not less. The disclosure window is a measurement instrument.

## What this does not cover

Arms that measure my own code (`S2_mcp_agentdb`, `S5_agentdb`), open-source
software (`S1_mcp_clickhouse`), or the Family A ablation ladder carry no such
obligation and are published as soon as they are measured. Nothing here is a
promise to give a vendor editorial control — a correction is incorporated when
it is *correct*, and a disagreement that survives the window is published as a
disagreement.

## Related

- [`methodology.md`](methodology.md) — how the numbers are produced, and why
  they might be wrong.
- [`clickhouse-advisor.md`](clickhouse-advisor.md),
  [`databricks-grounding.md`](databricks-grounding.md) — the per-engine
  findings these rules govern.
