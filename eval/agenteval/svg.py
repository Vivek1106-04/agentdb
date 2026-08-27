"""Hand-written SVG primitives for the report's charts (SPEC §11.6).

Charts are committed next to the traces, so they have to behave like text: no
plotting library, no embedded raster, no timestamp in the output, every
coordinate rounded to two decimals. The same records produce the same bytes,
which is what makes a chart reviewable in a pull request.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from html import escape

FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"

PALETTE = (
    "#3b6ea5",
    "#c1553b",
    "#4f9d69",
    "#8c6bb1",
    "#c9a227",
    "#4aa3b5",
    "#9c6644",
    "#7a7a7a",
)
"""Series colours, ordered so the first three stay distinguishable under both
deuteranopia and protanopia. Charts 1, 3 and 6 rarely need more than three."""

INK = "#1c1c1c"
MUTED = "#6b6b6b"
GRID = "#dcdcdc"
AXIS = "#9a9a9a"
BACKGROUND = "#ffffff"

WIDTH = 900.0
LABEL_WIDTH = 240.0
RIGHT_MARGIN = 100.0
"""Wide enough that a bar at 100% still has room for its value label."""
TITLE_TOP = 58.0
LEGEND_HEIGHT = 22.0
LEGEND_BASELINE = 68.0
"""Below the subtitle, above the plot: TITLE_TOP + LEGEND_HEIGHT is where bars start."""
FOOTER_HEIGHT = 36.0
BAR_HEIGHT = 18.0
BAR_GAP = 4.0
GROUP_GAP = 12.0
TICKS = 5

Formatter = Callable[[float], str]


@dataclass(frozen=True, slots=True)
class Bar:
    """One horizontal bar, optionally with a bootstrap interval drawn on it."""

    label: str
    value: float
    low: float | None = None
    high: float | None = None
    series: str = ""
    annotation: str = ""


@dataclass(frozen=True, slots=True)
class Segment:
    """One slice of a stacked bar."""

    label: str
    category: str
    value: float


@dataclass(frozen=True, slots=True)
class Point:
    """One labelled dot in a scatter."""

    label: str
    x: float
    y: float
    series: str = ""


def horizontal_bars(
    *,
    title: str,
    subtitle: str,
    bars: Sequence[Bar],
    tick_format: Formatter,
    axis_max: float | None = None,
) -> str:
    """Grouped horizontal bars: one group per label, one bar per series."""
    groups = _ordered(bar.label for bar in bars)
    series = _ordered(bar.series for bar in bars)
    colours = _colours(series)
    top = TITLE_TOP + (LEGEND_HEIGHT if len(series) > 1 else 0.0)
    rows = len(bars) * (BAR_HEIGHT + BAR_GAP) + (len(groups) - 1) * GROUP_GAP
    bottom = top + rows
    height = bottom + FOOTER_HEIGHT
    limit = (
        axis_max
        if axis_max is not None
        else nice_maximum([bar.high if bar.high is not None else bar.value for bar in bars])
    )
    left, right = LABEL_WIDTH, WIDTH - RIGHT_MARGIN

    parts = _open(WIDTH, height)
    parts += _header(title, subtitle)
    if len(series) > 1:
        parts += _legend(series, colours, y=LEGEND_BASELINE)
    parts += _vertical_axis(left, right, top, bottom, limit, tick_format)

    y = top
    for group in groups:
        members = [bar for bar in bars if bar.label == group]
        block = len(members) * (BAR_HEIGHT + BAR_GAP)
        parts.append(
            _text(left - 10, y + block / 2, group, size=12, anchor="end", baseline="middle")
        )
        for bar in members:
            parts += _bar_row(
                bar, y=y, left=left, right=right, limit=limit, fill=colours[bar.series]
            )
            y += BAR_HEIGHT + BAR_GAP
        y += GROUP_GAP
    return _close(parts)


def stacked_bars(
    *,
    title: str,
    subtitle: str,
    segments: Sequence[Segment],
    tick_format: Formatter,
) -> str:
    """One bar per label, split by category. Chart 4: the error taxonomy."""
    groups = _ordered(segment.label for segment in segments)
    categories = _ordered(segment.category for segment in segments)
    colours = _colours(categories)
    top = TITLE_TOP + LEGEND_HEIGHT
    bottom = top + len(groups) * (BAR_HEIGHT + BAR_GAP + GROUP_GAP)
    height = bottom + FOOTER_HEIGHT
    totals = {
        group: sum(segment.value for segment in segments if segment.label == group)
        for group in groups
    }
    limit = nice_maximum(list(totals.values()))
    left, right = LABEL_WIDTH, WIDTH - RIGHT_MARGIN

    parts = _open(WIDTH, height)
    parts += _header(title, subtitle)
    parts += _legend(categories, colours, y=LEGEND_BASELINE)
    parts += _vertical_axis(left, right, top, bottom, limit, tick_format)

    y = top
    for group in groups:
        parts.append(
            _text(left - 10, y + BAR_HEIGHT / 2, group, size=12, anchor="end", baseline="middle")
        )
        x = left
        for category in categories:
            value = sum(
                segment.value
                for segment in segments
                if segment.label == group and segment.category == category
            )
            span = _span(value, limit, left, right)
            parts.append(_rect(x, y, span, BAR_HEIGHT, colours[category]))
            x += span
        parts.append(
            _text(
                x + 6,
                y + BAR_HEIGHT / 2,
                tick_format(totals[group]),
                size=11,
                fill=MUTED,
                baseline="middle",
            )
        )
        y += BAR_HEIGHT + BAR_GAP + GROUP_GAP
    return _close(parts)


def scatter(
    *,
    title: str,
    subtitle: str,
    points: Sequence[Point],
    x_label: str,
    y_label: str,
    x_format: Formatter,
    y_format: Formatter,
    y_max: float | None = None,
) -> str:
    """Chart 6: accuracy against cost, one dot per arm."""
    series = _ordered(point.series for point in points)
    colours = _colours(series)
    left, right = 90.0, WIDTH - 150.0
    top = TITLE_TOP + LEGEND_HEIGHT
    bottom = top + 330.0
    height = bottom + FOOTER_HEIGHT + 14

    x_limit = nice_maximum([point.x for point in points])
    y_limit = y_max if y_max is not None else nice_maximum([point.y for point in points])

    parts = _open(WIDTH, height)
    parts += _header(title, subtitle)
    parts += _legend(series, colours, y=LEGEND_BASELINE)
    parts += _vertical_axis(left, right, top, bottom, x_limit, x_format)
    parts += _horizontal_axis(left, top, bottom, y_limit, y_format)
    parts.append(
        _text(
            (left + right) / 2,
            bottom + FOOTER_HEIGHT + 4,
            x_label,
            size=11,
            fill=MUTED,
            anchor="middle",
        )
    )
    parts.append(
        _text(18, (top + bottom) / 2, y_label, size=11, fill=MUTED, anchor="middle", rotate=-90)
    )

    for point in points:
        x = left + _span(point.x, x_limit, left, right)
        y = bottom - _span(point.y, y_limit, top, bottom)
        parts.append(
            f'<circle cx="{_num(x)}" cy="{_num(y)}" r="5" fill="{colours[point.series]}" '
            f'fill-opacity="0.85" stroke="{BACKGROUND}" stroke-width="1"/>'
        )
        parts.append(_text(x + 9, y, point.label, size=10, fill=INK, baseline="middle"))
    return _close(parts)


def nice_maximum(values: Sequence[float]) -> float:
    """A round axis maximum at or above every value, never zero."""
    largest = max([*values, 0.0])
    if largest <= 0.0:
        return 1.0
    base = 10.0 ** math.floor(math.log10(largest))
    for step in (1.0, 2.0, 2.5, 5.0):
        if largest <= step * base:
            return step * base
    return 10.0 * base


def percent(value: float) -> str:
    """Axis labels for a rate expressed as a fraction."""
    return f"{value:.0%}"


def count(value: float) -> str:
    return f"{value:,.0f}"


def bytes_si(value: float) -> str:
    """Bytes on a chart axis, in the unit a reader can hold in their head.

    A decimal below 100 of a unit, because rounding 1.2 GB and 2.4 GB both to
    "2 GB" is how a chart quietly stops distinguishing the arms it exists to
    distinguish.
    """
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0:
            whole = value >= 100.0 or unit == "B"
            return f"{value:,.0f} {unit}" if whole else f"{value:,.1f} {unit}"
        value /= 1024.0
    return f"{value:,.1f} TB"


def _bar_row(
    bar: Bar, *, y: float, left: float, right: float, limit: float, fill: str
) -> list[str]:
    span = _span(bar.value, limit, left, right)
    parts = [_rect(left, y, span, BAR_HEIGHT, fill)]
    if bar.low is not None and bar.high is not None:
        low = left + _span(bar.low, limit, left, right)
        high = left + _span(bar.high, limit, left, right)
        middle = y + BAR_HEIGHT / 2
        parts.append(
            f'<path d="M{_num(low)} {_num(middle - 4)} V{_num(middle + 4)} M{_num(low)} '
            f"{_num(middle)} H{_num(high)} M{_num(high)} {_num(middle - 4)} "
            f'V{_num(middle + 4)}" stroke="{INK}" stroke-width="1.2" fill="none"/>'
        )
        parts.append(
            _text(high + 8, middle, bar.annotation, size=11, fill=MUTED, baseline="middle")
        )
        return parts
    parts.append(
        _text(
            left + span + 8,
            y + BAR_HEIGHT / 2,
            bar.annotation,
            size=11,
            fill=MUTED,
            baseline="middle",
        )
    )
    return parts


def _vertical_axis(
    left: float, right: float, top: float, bottom: float, limit: float, tick_format: Formatter
) -> list[str]:
    """Gridlines running up the plot, labelled along the bottom."""
    parts = []
    for index in range(TICKS):
        fraction = index / (TICKS - 1)
        x = left + fraction * (right - left)
        parts.append(
            f'<line x1="{_num(x)}" y1="{_num(top)}" x2="{_num(x)}" y2="{_num(bottom)}" '
            f'stroke="{GRID if index else AXIS}" stroke-width="1"/>'
        )
        parts.append(
            _text(
                x, bottom + 18, tick_format(fraction * limit), size=11, fill=MUTED, anchor="middle"
            )
        )
    return parts


def _horizontal_axis(
    left: float, top: float, bottom: float, limit: float, tick_format: Formatter
) -> list[str]:
    parts = []
    for index in range(TICKS):
        fraction = index / (TICKS - 1)
        y = bottom - fraction * (bottom - top)
        parts.append(
            _text(
                left - 10,
                y,
                tick_format(fraction * limit),
                size=11,
                fill=MUTED,
                anchor="end",
                baseline="middle",
            )
        )
    return parts


def _header(title: str, subtitle: str) -> list[str]:
    return [
        _text(24, 30, title, size=16, weight="600"),
        _text(24, 48, subtitle, size=11, fill=MUTED),
    ]


def _legend(names: Sequence[str], colours: dict[str, str], *, y: float) -> list[str]:
    parts = []
    x = 24.0
    for name in names:
        parts.append(_rect(x, y - 8, 10, 10, colours[name]))
        parts.append(_text(x + 15, y, name, size=11, fill=MUTED, baseline="middle"))
        x += 26 + 6.5 * len(name)
    return parts


def _colours(names: Sequence[str]) -> dict[str, str]:
    return {name: PALETTE[index % len(PALETTE)] for index, name in enumerate(names)}


def _span(value: float, limit: float, low: float, high: float) -> float:
    return max(0.0, min(1.0, value / limit)) * (high - low)


def _rect(x: float, y: float, width: float, height: float, fill: str) -> str:
    return (
        f'<rect x="{_num(x)}" y="{_num(y)}" width="{_num(width)}" '
        f'height="{_num(height)}" fill="{fill}"/>'
    )


def _text(
    x: float,
    y: float,
    content: str,
    *,
    size: float = 12,
    fill: str = INK,
    anchor: str = "start",
    weight: str = "normal",
    baseline: str = "alphabetic",
    rotate: float | None = None,
) -> str:
    transform = f' transform="rotate({_num(rotate)} {_num(x)} {_num(y)})"' if rotate else ""
    return (
        f'<text x="{_num(x)}" y="{_num(y)}" font-size="{_num(size)}" fill="{fill}" '
        f'text-anchor="{anchor}" font-weight="{weight}" dominant-baseline="{baseline}"'
        f"{transform}>{escape(content)}</text>"
    )


def _open(width: float, height: float) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_num(width)}" '
        f'height="{_num(height)}" viewBox="0 0 {_num(width)} {_num(height)}" '
        f'font-family="{FONT}">',
        _rect(0, 0, width, height, BACKGROUND),
    ]


def _close(parts: list[str]) -> str:
    return "\n".join([*parts, "</svg>"]) + "\n"


def _ordered(names: Iterable[str]) -> tuple[str, ...]:
    """First appearance wins, so a chart's order is the caller's order."""
    return tuple(dict.fromkeys(names))


def _num(value: float) -> str:
    return f"{value:.2f}"
