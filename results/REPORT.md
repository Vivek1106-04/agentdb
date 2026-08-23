# agentdb benchmark results

Generated from `results/raw/*.jsonl` by `make report`. No model or engine
is called: every number below is a function of the committed traces.

## Run

- runs: run-20260815T082806Z
- suite(s): tpch_nl
- engine(s): databricks
- tasks: 12
- graded cells: 36

## Execution accuracy

| arm | model | EX (95% CI) | EX@1 | valid SQL | retries | in tok | out tok | ctx B |
|---|---|---|---|---|---|---|---|---|
| `S5_claude_code` | claude-cli/sonnet | 100.0% [100.0%, 100.0%] | 100.0% | 100.0% | 0.00 | 2 | 104 | 4924 |

## Error taxonomy

No failures recorded.

## Paired comparison

None: this run contains no records for `A0_baseline`, so the arms above are reported on their own terms. A delta needs both arms to have run the same cells.

## Configuration

- `S5_claude_code` v1.0 — `sha256:53facc8ad8ad38ead243019a394672861c581253ae58faef73212fadbe49a52d`

## Context the arm did not choose

- `S5_claude_code` carried up to **19,462 tokens** of product context per call, beyond the prompt this harness sent.

> Counted from the product's own usage accounting, not estimated. It is excluded from the token columns above because it is not the arm's grounding — but a reader comparing rows should know it was there.
