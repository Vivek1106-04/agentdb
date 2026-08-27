"""``results/charts/*.svg`` — the figures SPEC §11.6 requires.

Built from the same committed traces as ``REPORT.md`` and by the same command,
so a chart can never drift from the table above it. A chart whose data the run
does not contain is not emitted: an empty axis published next to five real ones
is a claim the traces do not support.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agenteval.stats import bootstrap_mean
from agenteval.svg import (
    Bar,
    Point,
    Segment,
    bytes_si,
    count,
    horizontal_bars,
    percent,
    scatter,
    stacked_bars,
)

ABLATION_PREFIX = "A"
"""Family A arms are named ``A0_baseline`` … ``A7_oracle`` (SPEC §11.2).

Everything else — the Family S systems, and any third-party product measured
against the suite — belongs on the leaderboard, so nothing is silently dropped
by a naming convention it never agreed to.
"""

CROSS_ENGINE_SUITE = "tpch_nl"
"""The suite whose gold results are engine-independent by construction."""

_RUN_DATE = re.compile(r"(\d{4})(\d{2})(\d{2})")

Record = Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Chart:
    """One SVG figure, named by the position it holds in SPEC §11.6."""

    filename: str
    title: str
    svg: str


def build_charts(records: Sequence[Record]) -> tuple[Chart, ...]:
    """Every chart these records can support, in the order the spec lists them."""
    candidates = (
        _leaderboard(records),
        _ablation(records),
        _cross_engine(records),
        _error_classes(records),
        _bytes_read(records),
        _accuracy_vs_cost(records),
    )
    return tuple(chart for chart in candidates if chart is not None)


def write_charts(charts: Sequence[Chart], directory: Path) -> tuple[Path, ...]:
    """Write each chart into ``directory``, creating it if need be."""
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for chart in charts:
        path = directory / chart.filename
        path.write_text(chart.svg, encoding="utf-8")
        written.append(path)
    return tuple(written)


def _leaderboard(records: Sequence[Record]) -> Chart | None:
    """Chart 1, the headline: execution accuracy by system, per suite."""
    rows = [record for record in records if not _is_ablation(record)]
    if not rows:
        return None

    return Chart(
        filename="01-family-s-leaderboard.svg",
        title="Family S leaderboard",
        svg=horizontal_bars(
            title="Execution accuracy by system",
            subtitle=(
                f"95% bootstrap intervals over graded cells. Measured {_dates(rows)}. "
                f"Engine(s): {_joined(rows, 'engine')}."
            ),
            bars=_accuracy_bars(rows, series_field="suite"),
            tick_format=percent,
            axis_max=1.0,
        ),
    )


def _ablation(records: Sequence[Record]) -> Chart | None:
    """Chart 2: the Family A ladder, which context each rung adds."""
    rows = [record for record in records if _is_ablation(record)]
    if not rows:
        return None

    return Chart(
        filename="02-family-a-ablation.svg",
        title="Family A ablation",
        svg=horizontal_bars(
            title="Execution accuracy by ablation arm",
            subtitle=(
                "Each arm adds one kind of context to the arm above it. "
                f"95% bootstrap intervals. Engine(s): {_joined(rows, 'engine')}."
            ),
            bars=_accuracy_bars(rows, series_field="suite"),
            tick_format=percent,
            axis_max=1.0,
        ),
    )


def _cross_engine(records: Sequence[Record]) -> Chart | None:
    """Chart 3: the one nobody has published — same questions, two engines."""
    rows = [record for record in records if str(record["suite"]) == CROSS_ENGINE_SUITE]
    if len({str(record["engine"]) for record in rows}) < 2:
        return None

    return Chart(
        filename="03-cross-engine-tpch.svg",
        title="Cross-engine accuracy",
        svg=horizontal_bars(
            title=f"Execution accuracy by arm, ClickHouse vs Databricks ({CROSS_ENGINE_SUITE})",
            subtitle=(
                "Identical questions, identical gold results, one warehouse each. "
                f"95% bootstrap intervals. Measured {_dates(rows)}."
            ),
            bars=_accuracy_bars(rows, series_field="engine"),
            tick_format=percent,
            axis_max=1.0,
        ),
    )


def _error_classes(records: Sequence[Record]) -> Chart | None:
    """Chart 4: *which* failures each kind of context actually fixes."""
    segments = [
        Segment(label=str(record["system"]), category=str(record["error_class"]), value=1.0)
        for record in _by_system(records)
        if str(record["error_class"]) != "none"
    ]
    if not segments:
        return None

    return Chart(
        filename="04-error-classes.svg",
        title="Error classes by arm",
        svg=stacked_bars(
            title="Failures by error class",
            subtitle=(
                "Graded cells that failed, by the class the grader assigned. Correct cells omitted."
            ),
            segments=segments,
            tick_format=count,
        ),
    )


def _bytes_read(records: Sequence[Record]) -> Chart | None:
    """Chart 5: the efficiency story, as far as the traces carry it.

    Pruning ratio belongs beside this and is deliberately absent: it comes from
    a plan, and the harness records what the engine reported for the executed
    query rather than running an extra ``EXPLAIN`` per cell. Saying so on the
    chart is better than plotting granule ratios and file ratios on one axis,
    which SPEC §11.6 forbids for good reason.
    """
    measured: dict[str, list[float]] = {}
    for record in _by_system(records):
        if record.get("bytes_read") is not None:
            measured.setdefault(str(record["system"]), []).append(float(record["bytes_read"]))
    if not measured:
        return None

    bars = [
        Bar(label=system, value=_mean(values), annotation=bytes_si(_mean(values)))
        for system, values in sorted(measured.items())
    ]
    return Chart(
        filename="05-bytes-read.svg",
        title="Bytes read by arm",
        svg=horizontal_bars(
            title="Mean bytes read by the graded query",
            subtitle=(
                "As reported by the engine for the executed query. Pruning ratio is not "
                "plotted here: it needs a plan per cell, which these traces do not carry."
            ),
            bars=bars,
            tick_format=bytes_si,
        ),
    )


def _accuracy_vs_cost(records: Sequence[Record]) -> Chart | None:
    """Chart 6: the honest trade-off an engineer actually studies."""
    grouped: dict[tuple[str, str], list[Record]] = {}
    for record in records:
        grouped.setdefault((str(record["system"]), str(record["suite"])), []).append(record)

    points = [
        Point(
            label=system,
            x=_mean([float(row["input_tokens"]) + float(row["output_tokens"]) for row in rows]),
            y=_mean([float(bool(row["execution_accuracy"])) for row in rows]),
            series=suite,
        )
        for (system, suite), rows in sorted(grouped.items())
    ]
    if not points:
        return None

    return Chart(
        filename="06-accuracy-vs-cost.svg",
        title="Accuracy against token cost",
        svg=scatter(
            title="Execution accuracy against token cost",
            subtitle="One point per arm and suite. Up and to the left is better.",
            points=points,
            x_label="mean tokens per task (input + output)",
            y_label="execution accuracy",
            x_format=count,
            y_format=percent,
            y_max=1.0,
        ),
    )


def _accuracy_bars(records: Sequence[Record], *, series_field: str) -> tuple[Bar, ...]:
    """One bar per (system, series), with the bootstrap interval drawn on it."""
    grouped: dict[tuple[str, str], list[float]] = {}
    for record in records:
        key = (str(record["system"]), str(record[series_field]))
        grouped.setdefault(key, []).append(float(bool(record["execution_accuracy"])))

    bars = []
    for (system, series), values in sorted(grouped.items()):
        interval = bootstrap_mean(values)
        bars.append(
            Bar(
                label=system,
                value=interval.mean,
                low=interval.low,
                high=interval.high,
                series=series,
                annotation=f"{interval.mean:.0%}",
            )
        )
    return tuple(bars)


def _by_system(records: Sequence[Record]) -> list[Record]:
    """Records sorted by arm, so every chart orders its bars the same way."""
    return sorted(records, key=lambda record: str(record["system"]))


def _is_ablation(record: Record) -> bool:
    return str(record["system"]).startswith(ABLATION_PREFIX)


def _dates(records: Sequence[Record]) -> str:
    """The measurement dates SPEC §11.6 requires on the headline chart."""
    found = sorted(
        {
            f"{match[1]}-{match[2]}-{match[3]}"
            for record in records
            if (match := _RUN_DATE.search(str(record["run_id"])))
        }
    )
    return ", ".join(found) if found else "date not recorded in the run id"


def _joined(records: Sequence[Record], field: str) -> str:
    return ", ".join(sorted({str(record[field]) for record in records}))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)
