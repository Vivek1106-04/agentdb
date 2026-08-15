# agentdb benchmark results

Generated from `results/raw/*.jsonl` by `make report`. No model or engine
is called: every number below is a function of the committed traces.

## Run

- runs: run-20260815T081607Z, run-20260815T081755Z, run-20260815T082257Z
- suite(s): tpch_nl
- engine(s): databricks
- tasks: 12
- graded cells: 27

## Execution accuracy

| arm | model | EX (95% CI) | EX@1 | valid SQL | retries | in tok | out tok | ctx B |
|---|---|---|---|---|---|---|---|---|
| `S5_claude_code` | claude-cli/sonnet | 96.3% [88.9%, 100.0%] | 96.3% | 100.0% | 0.00 | 2 | 100 | 4924 |

## Error taxonomy

No failures recorded.

## Configuration

- `S5_claude_code` v1.0 — `sha256:53facc8ad8ad38ead243019a394672861c581253ae58faef73212fadbe49a52d`
