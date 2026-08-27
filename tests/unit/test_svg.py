"""The SVG primitives: deterministic bytes, and axes that hold their numbers."""

from __future__ import annotations

from agenteval.svg import (
    Bar,
    Point,
    Segment,
    bytes_si,
    count,
    horizontal_bars,
    nice_maximum,
    percent,
    scatter,
    stacked_bars,
)


def test_the_same_bars_render_the_same_bytes() -> None:
    bars = [Bar(label="A0", value=0.4, low=0.3, high=0.5, series="clickbench_nl")]

    first = horizontal_bars(title="t", subtitle="s", bars=bars, tick_format=percent, axis_max=1.0)
    second = horizontal_bars(title="t", subtitle="s", bars=bars, tick_format=percent, axis_max=1.0)

    assert first == second
    assert first.startswith("<svg xmlns=")
    assert first.endswith("</svg>\n")


def test_a_single_series_chart_draws_no_legend() -> None:
    svg = horizontal_bars(
        title="bytes",
        subtitle="s",
        bars=[Bar(label="A0", value=1024, annotation="1 KB")],
        tick_format=bytes_si,
    )

    assert "1 KB" in svg
    assert svg.count("<rect") == 2  # the background and the one bar


def test_two_series_get_a_legend_and_distinct_colours() -> None:
    svg = horizontal_bars(
        title="t",
        subtitle="s",
        bars=[
            Bar(label="A0", value=0.5, low=0.4, high=0.6, series="clickhouse"),
            Bar(label="A0", value=0.7, low=0.6, high=0.8, series="databricks"),
        ],
        tick_format=percent,
        axis_max=1.0,
    )

    assert "clickhouse" in svg
    assert "databricks" in svg
    assert 'fill="#3b6ea5"' in svg
    assert 'fill="#c1553b"' in svg


def test_an_interval_is_drawn_as_a_capped_line() -> None:
    svg = horizontal_bars(
        title="t",
        subtitle="s",
        bars=[Bar(label="A0", value=0.5, low=0.4, high=0.6, annotation="50%")],
        tick_format=percent,
        axis_max=1.0,
    )

    assert "<path d=" in svg
    assert "50%" in svg


def test_labels_are_escaped_rather_than_injected() -> None:
    svg = horizontal_bars(
        title="t",
        subtitle="s",
        bars=[Bar(label="<script>", value=1.0)],
        tick_format=percent,
        axis_max=1.0,
    )

    assert "<script>" not in svg
    assert "&lt;script&gt;" in svg


def test_a_stacked_bar_totals_its_segments() -> None:
    svg = stacked_bars(
        title="errors",
        subtitle="s",
        segments=[
            Segment(label="A0", category="wrong_result", value=2.0),
            Segment(label="A0", category="sql_error", value=1.0),
            Segment(label="A1", category="wrong_result", value=1.0),
        ],
        tick_format=count,
    )

    assert "wrong_result" in svg
    assert "sql_error" in svg
    assert ">3</text>" in svg


def test_a_scatter_labels_every_point_and_both_axes() -> None:
    svg = scatter(
        title="cost",
        subtitle="s",
        points=[Point(label="A0", x=1200.0, y=0.4, series="clickbench_nl")],
        x_label="tokens",
        y_label="accuracy",
        x_format=count,
        y_format=percent,
        y_max=1.0,
    )

    assert "<circle" in svg
    assert "rotate(-90.00" in svg
    assert ">tokens</text>" in svg
    assert ">accuracy</text>" in svg


def test_an_axis_maximum_is_a_round_number_at_or_above_the_data() -> None:
    assert nice_maximum([0.0]) == 1.0
    assert nice_maximum([0.9]) == 1.0
    assert nice_maximum([1.5]) == 2.0
    assert nice_maximum([2.4]) == 2.5
    assert nice_maximum([4.0]) == 5.0
    assert nice_maximum([7.0]) == 10.0


def test_bytes_are_labelled_in_the_unit_a_reader_can_hold() -> None:
    assert bytes_si(50) == "50 B"
    assert bytes_si(512) == "512 B"
    assert bytes_si(2048) == "2.0 KB"
    assert bytes_si(500 * 1024) == "500 KB"
    assert bytes_si(5 * 1024**2) == "5.0 MB"
    assert bytes_si(3 * 1024**3) == "3.0 GB"
    assert bytes_si(2 * 1024**4) == "2.0 TB"


def test_a_value_past_the_axis_maximum_is_clamped_not_overdrawn() -> None:
    svg = horizontal_bars(
        title="t",
        subtitle="s",
        bars=[Bar(label="A0", value=2.0)],
        tick_format=percent,
        axis_max=1.0,
    )

    assert 'width="560.00"' in svg  # the full plot width, not more
