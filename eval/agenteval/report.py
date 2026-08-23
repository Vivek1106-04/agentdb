"""``results/REPORT.md`` — generated from committed traces, never from memory.

The report is a pure function of ``results/raw/*.jsonl``. ``make report`` calls
no model and touches no engine, so anyone can regenerate every published number
from the committed evidence and get the same file back. That property is what
makes the traces worth committing.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agenteval.stats import (
    Interval,
    McNemarResult,
    bootstrap_mean,
    mcnemar,
    paired_bootstrap_difference,
)
from agenteval.traces import read_records

REPORT_TITLE = "# agentdb benchmark results"

CELL_KEY = ("task_id", "seed", "model")
"""What makes two rows the same measurement on different arms."""


class ReportError(ValueError):
    """The traces cannot support a report."""


@dataclass(frozen=True, slots=True)
class ArmSummary:
    """One arm x model row of the leaderboard."""

    system: str
    model: str | None
    version: str
    config_fingerprint: str
    controls_model: bool
    execution_accuracy: Interval
    accuracy_at_1: float
    valid_sql: float
    mean_retries: float
    errors: Mapping[str, int]
    mean_input_tokens: float
    mean_output_tokens: float
    mean_context_bytes: float

    @property
    def label(self) -> str:
        return f"{self.system} ({self.model or 'system-chosen'})"


@dataclass(frozen=True, slots=True)
class Comparison:
    """A paired comparison of one arm against the baseline."""

    system: str
    baseline: str
    paired_cells: int
    difference: Interval
    test: McNemarResult


def load_run(directory: Path) -> tuple[dict[str, Any], ...]:
    """Every trace record under ``directory``, in file then line order."""
    if not directory.is_dir():
        raise ReportError(f"no trace directory at {directory}")

    records = [
        record for path in sorted(directory.glob("*.jsonl")) for record in read_records(path)
    ]
    if not records:
        raise ReportError(f"no trace records found in {directory}")
    return tuple(records)


def summarize(records: Sequence[Mapping[str, Any]]) -> tuple[ArmSummary, ...]:
    """One summary per (system, model), ordered by arm name."""
    groups: dict[tuple[str, str | None], list[Mapping[str, Any]]] = {}
    for record in records:
        groups.setdefault((str(record["system"]), record["model"]), []).append(record)

    return tuple(
        _summarize_group(system, model, rows)
        for (system, model), rows in sorted(groups.items(), key=lambda item: item[0][0])
    )


def compare_to_baseline(
    records: Sequence[Mapping[str, Any]], baseline: str
) -> tuple[Comparison, ...]:
    """Paired difference of every other arm against ``baseline``.

    Only cells both arms actually ran are compared. A run that lost cells to a
    provider outage produces a narrower comparison, not a silently biased one.
    """
    by_arm = _index_by_cell(records)
    if baseline not in by_arm:
        raise ReportError(f"no records for baseline arm {baseline!r}")

    reference = by_arm[baseline]
    comparisons = []
    for system in sorted(by_arm):
        if system == baseline:
            continue
        shared = sorted(set(reference) & set(by_arm[system]))
        if not shared:
            continue
        first = [bool(reference[key]["execution_accuracy"]) for key in shared]
        second = [bool(by_arm[system][key]["execution_accuracy"]) for key in shared]
        comparisons.append(
            Comparison(
                system=system,
                baseline=baseline,
                paired_cells=len(shared),
                difference=paired_bootstrap_difference(
                    [float(value) for value in first], [float(value) for value in second]
                ),
                test=mcnemar(first, second),
            )
        )
    return tuple(comparisons)


def render(records: Sequence[Mapping[str, Any]], *, baseline: str | None = None) -> str:
    """The whole report, as markdown."""
    summaries = summarize(records)
    sections = [
        REPORT_TITLE,
        "",
        "Generated from `results/raw/*.jsonl` by `make report`. No model or engine",
        "is called: every number below is a function of the committed traces.",
        "",
        _render_provenance(records),
        "",
        _render_leaderboard(summaries),
        "",
        _render_errors(summaries),
        "",
        _render_contamination(contamination_check(records)),
    ]

    reference = baseline or summaries[0].system
    measured = {summary.system for summary in summaries}
    if reference not in measured:
        # A Family S run legitimately has no A0: somebody scoring one third-party
        # system against the suite never built agentdb's ladder. Say the
        # comparison is absent rather than failing the command that regenerates
        # the report from the traces already committed.
        sections += ["", _render_absent_baseline(reference)]
    elif any(summary.system != reference for summary in summaries):
        sections += ["", _render_comparisons(compare_to_baseline(records, reference))]

    sections += ["", _render_footnotes(summaries)]
    undeclared = _render_undeclared_context(records)
    if undeclared:
        sections += ["", undeclared]
    return "\n".join(sections) + "\n"


CONTAMINATION_TAG = "clickbench_original"
"""The tag marking a task derived from ClickBench's published queries.

Those queries have been public for years and are very likely in training data.
Tasks without the tag were authored for this project and are not (SPEC §11.4).
"""


@dataclass(frozen=True, slots=True)
class Contamination:
    """One arm's accuracy on public tasks against its accuracy on authored ones."""

    system: str
    derived_correct: int
    derived_total: int
    authored_correct: int
    authored_total: int

    @property
    def derived_accuracy(self) -> float:
        return self.derived_correct / self.derived_total

    @property
    def authored_accuracy(self) -> float:
        return self.authored_correct / self.authored_total

    @property
    def gap(self) -> float:
        """Public minus authored. A large positive gap is the memorization signal."""
        return self.derived_accuracy - self.authored_accuracy


def contamination_check(records: Sequence[Mapping[str, Any]]) -> tuple[Contamination, ...]:
    """Split each arm's tasks by whether the question predates this project.

    The check that separates this from a marketing benchmark, and it is computed
    even though it can only weaken the headline: if an arm scores far better on
    ClickBench's own published queries than on questions written for this suite,
    the difference is a fact about training data rather than about grounding.

    Arms whose run covered only one side of the split are omitted rather than
    reported with a denominator of zero.
    """
    by_system: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        by_system.setdefault(str(record["system"]), []).append(record)

    found: list[Contamination] = []
    for system, rows in sorted(by_system.items()):
        derived = [row for row in rows if CONTAMINATION_TAG in (row.get("tags") or ())]
        authored = [row for row in rows if CONTAMINATION_TAG not in (row.get("tags") or ())]
        if not derived or not authored:
            continue
        found.append(
            Contamination(
                system=system,
                derived_correct=sum(1 for row in derived if row["execution_accuracy"]),
                derived_total=len(derived),
                authored_correct=sum(1 for row in authored if row["execution_accuracy"]),
                authored_total=len(authored),
            )
        )
    return tuple(found)


def _render_contamination(found: Sequence[Contamination]) -> str:
    """The section that exists to be published even when it is unflattering."""
    if not found:
        return "\n".join(
            [
                "## Contamination check",
                "",
                "Not computable from this run: every arm covered only one side of the "
                f"split. The check compares accuracy on tasks tagged `{CONTAMINATION_TAG}` "
                "— ClickBench's own published queries, public for years and plausibly in "
                "training data — against tasks authored for this suite.",
            ]
        )

    header = "| arm | public tasks | authored tasks | gap |"
    divider = "|---|---|---|---|"
    rows = [
        f"| `{item.system}` | {item.derived_accuracy:.1%} "
        f"({item.derived_correct}/{item.derived_total}) | "
        f"{item.authored_accuracy:.1%} ({item.authored_correct}/{item.authored_total}) | "
        f"{item.gap:+.1%} |"
        for item in found
    ]
    return "\n".join(
        [
            "## Contamination check",
            "",
            "ClickBench's queries have been public for years and are plausibly in "
            "training data; the authored questions are not. A large positive gap is a "
            "fact about memorization rather than about grounding (SPEC §11.4).",
            "",
            header,
            divider,
            *rows,
        ]
    )


def _render_absent_baseline(baseline: str) -> str:
    """Say why there is no paired comparison, rather than omitting the section."""
    return "\n".join(
        [
            "## Paired comparison",
            "",
            f"None: this run contains no records for `{baseline}`, so the arms above "
            "are reported on their own terms. A delta needs both arms to have run "
            "the same cells.",
        ]
    )


def _summarize_group(
    system: str, model: str | None, rows: Sequence[Mapping[str, Any]]
) -> ArmSummary:
    first = rows[0]
    return ArmSummary(
        system=system,
        model=model,
        version=str(first["system_version"]),
        config_fingerprint=str(first["config_fingerprint"]),
        controls_model=bool(first["controls_model"]),
        execution_accuracy=bootstrap_mean([float(bool(r["execution_accuracy"])) for r in rows]),
        accuracy_at_1=_mean(float(bool(r["accuracy_at_1"])) for r in rows),
        valid_sql=_mean(float(bool(r["valid_sql"])) for r in rows),
        mean_retries=_mean(float(r["retries"]) for r in rows),
        errors=Counter(str(r["error_class"]) for r in rows if r["error_class"] != "none"),
        mean_input_tokens=_mean(float(r["input_tokens"]) for r in rows),
        mean_output_tokens=_mean(float(r["output_tokens"]) for r in rows),
        mean_context_bytes=_mean(float(r["context_bytes"]) for r in rows),
    )


def _index_by_cell(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[tuple[Any, ...], Mapping[str, Any]]]:
    indexed: dict[str, dict[tuple[Any, ...], Mapping[str, Any]]] = {}
    for record in records:
        key = tuple(record[field] for field in CELL_KEY)
        indexed.setdefault(str(record["system"]), {})[key] = record
    return indexed


def _render_provenance(records: Sequence[Mapping[str, Any]]) -> str:
    runs = sorted({str(record["run_id"]) for record in records})
    suites = sorted({str(record["suite"]) for record in records})
    engines = sorted({str(record["engine"]) for record in records})
    tasks = len({str(record["task_id"]) for record in records})
    return "\n".join(
        [
            "## Run",
            "",
            f"- runs: {', '.join(runs)}",
            f"- suite(s): {', '.join(suites)}",
            f"- engine(s): {', '.join(engines)}",
            f"- tasks: {tasks}",
            f"- graded cells: {len(records)}",
        ]
    )


def _render_leaderboard(summaries: Sequence[ArmSummary]) -> str:
    header = "| arm | model | EX (95% CI) | EX@1 | valid SQL | retries | in tok | out tok | ctx B |"
    divider = "|---|---|---|---|---|---|---|---|---|"
    rows = [
        f"| `{s.system}` | {s.model or '_system-chosen_'} | {s.execution_accuracy} | "
        f"{s.accuracy_at_1:.1%} | {s.valid_sql:.1%} | {s.mean_retries:.2f} | "
        f"{s.mean_input_tokens:.0f} | {s.mean_output_tokens:.0f} | {s.mean_context_bytes:.0f} |"
        for s in summaries
    ]
    return "\n".join(["## Execution accuracy", "", header, divider, *rows])


def _render_errors(summaries: Sequence[ArmSummary]) -> str:
    classes = sorted({name for summary in summaries for name in summary.errors})
    if not classes:
        return "\n".join(["## Error taxonomy", "", "No failures recorded."])

    header = "| arm | " + " | ".join(classes) + " |"
    divider = "|---" * (len(classes) + 1) + "|"
    rows = [
        f"| `{summary.label}` | "
        + " | ".join(str(summary.errors.get(name, 0)) for name in classes)
        + " |"
        for summary in summaries
    ]
    return "\n".join(["## Error taxonomy", "", header, divider, *rows])


def _render_comparisons(comparisons: Sequence[Comparison]) -> str:
    if not comparisons:
        return ""

    lines = [
        "## Paired comparisons",
        "",
        f"Against `{comparisons[0].baseline}`, on the cells both arms ran.",
        "Exact McNemar on per-cell correctness; the interval is a paired bootstrap.",
        "",
        "| arm | paired cells | EX difference (95% CI) | wins | losses | p |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| `{c.system}` | {c.paired_cells} | {c.difference} | "
        f"{c.test.only_second} | {c.test.only_first} | {c.test.p_value:.4f} |"
        for c in comparisons
    ]
    return "\n".join(lines)


def _render_footnotes(summaries: Sequence[ArmSummary]) -> str:
    lines = ["## Configuration", ""]
    lines += [
        f"- `{summary.system}` v{summary.version} — `{summary.config_fingerprint}`"
        for summary in summaries
    ]
    managed = [summary for summary in summaries if not summary.controls_model]
    if managed:
        lines += [
            "",
            "> The following arms choose their own model and are **not** a "
            "like-for-like comparison with the rest of the table: "
            + ", ".join(f"`{summary.system}`" for summary in managed)
            + ".",
        ]
    return "\n".join(lines)


def _render_undeclared_context(records: Sequence[Mapping[str, Any]]) -> str:
    """Context an arm carried but did not choose, per SPEC §11.1's honesty rule.

    Some systems are products, and a product ships its own prompt. The token
    columns above count only what the arm sent, so an arm that arrived with tens
    of thousands of tokens of its own would otherwise read as the leanest row in
    the table. What cannot be measured is at least named.
    """
    seen: dict[str, int] = {}
    for record in records:
        for note in record.get("notes") or ():
            key, separator, value = str(note).partition("=")
            if separator and value.isdigit() and key.endswith("scaffolding_tokens"):
                system = str(record.get("system", "unknown"))
                seen[system] = max(seen.get(system, 0), int(value))
    if not seen:
        return ""

    lines = ["## Context the arm did not choose", ""]
    lines += [
        f"- `{system}` carried up to **{tokens:,} tokens** of product context per call, "
        "beyond the prompt this harness sent."
        for system, tokens in sorted(seen.items())
    ]
    lines += [
        "",
        "> Counted from the product's own usage accounting, not estimated. It is "
        "excluded from the token columns above because it is not the arm's "
        "grounding — but a reader comparing rows should know it was there.",
    ]
    return "\n".join(lines)


def _mean(values: Iterable[float]) -> float:
    collected = list(values)
    return sum(collected) / len(collected)
