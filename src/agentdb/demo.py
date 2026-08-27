"""``agentdb demo`` — the before/after panel the README shows (SPEC §14.1).

The same question twice against the same table: once as an agent writes it from
a schema dump, once as it writes it knowing the physical layout. Every number
printed is measured on the connected engine in the moment — the answers, the
bytes read, the plan warnings. Nothing here is a stored example, because a
side-by-side with rehearsed numbers is a screenshot, not evidence.

The demo deliberately refuses to smooth over a disappointing result: if the two
queries disagree, it says so and stops claiming anything; if the grounded query
reads more, it prints that ratio just the same.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentdb.adapters.base import Adapter
from agentdb.adapters.models import Limits, PhysicalLayout, RelationRef
from agentdb.config import Config
from agentdb.core import PlanExplainer, PlanSummary

DEMO_RESULT_ROWS = 10
"""The demo asks aggregate questions; ten rows is more than any of them return."""


@dataclass(frozen=True, slots=True)
class DemoCase:
    """One question, written twice."""

    question: str
    namespace: str
    relation: str
    naive_sql: str
    naive_note: str
    grounded_sql: str
    grounded_note: str

    @property
    def ref(self) -> RelationRef:
        return RelationRef(namespace=self.namespace, name=self.relation)


CLICKHOUSE_CASE = DemoCase(
    question="How many hits did counter 62 record?",
    namespace="agentdb",
    relation="hits",
    naive_sql="SELECT count() FROM agentdb.hits WHERE toString(CounterID) = '62'",
    naive_note=(
        "Nothing in a schema dump says CounterID is the leading sort-key column, "
        "so an agent treats it as an ordinary attribute and compares it as text. "
        "The answer is right and the cost is invisible from where it is standing."
    ),
    grounded_sql="SELECT count() FROM agentdb.hits WHERE CounterID = 62",
    grounded_note=(
        "grounded_context names CounterID as the leading sort-key column, so the "
        "agent compares it in its own type and the primary index prunes granules "
        "before any of them are read."
    ),
)

DATABRICKS_CASE = DemoCase(
    question="How much did customers in the BUILDING market segment order in 1995?",
    namespace="samples.tpch",
    relation="orders",
    naive_sql=(
        "SELECT sum(o.o_totalprice) FROM samples.tpch.orders o "
        "JOIN samples.tpch.customer c ON c.c_custkey = o.o_custkey "
        "WHERE c.c_mktsegment = 'BUILDING' AND year(o.o_orderdate) = 1995"
    ),
    naive_note=(
        "year() on the date column hides the range from Delta's file statistics, "
        "so every file in the table is a candidate."
    ),
    grounded_sql=(
        "SELECT sum(o.o_totalprice) FROM samples.tpch.orders o "
        "JOIN samples.tpch.customer c ON c.c_custkey = o.o_custkey "
        "WHERE c.c_mktsegment = 'BUILDING' "
        "AND o.o_orderdate >= DATE'1995-01-01' AND o.o_orderdate < DATE'1996-01-01'"
    ),
    grounded_note=(
        "The same predicate as a half-open range on the raw column, which the "
        "min/max statistics in the Delta log can actually skip on."
    ),
)

CASES = {"clickhouse": CLICKHOUSE_CASE, "databricks": DATABRICKS_CASE}


@dataclass(frozen=True, slots=True)
class Panel:
    """What one of the two queries actually did."""

    heading: str
    sql: str
    note: str
    answer: str
    bytes_read: int | None
    rows_read: int | None
    duration_ms: int | None
    summary: PlanSummary


async def measure(
    adapter: Adapter,
    explainer: PlanExplainer,
    case: DemoCase,
    *,
    heading: str,
    sql: str,
    note: str,
    config: Config,
) -> Panel:
    """Explain ``sql``, run it, and record what the engine reported."""
    summary = await explainer.explain(sql, case.namespace)
    result = await adapter.execute(
        sql,
        Limits(
            timeout_s=config.query_timeout_s,
            max_result_rows=DEMO_RESULT_ROWS,
            max_rows_to_read=config.max_rows_to_read,
        ),
    )
    return Panel(
        heading=heading,
        sql=sql,
        note=note,
        answer=_answer(result.rows),
        bytes_read=result.bytes_read,
        rows_read=result.rows_read,
        duration_ms=result.duration_ms,
        summary=summary,
    )


async def run_demo(adapter: Adapter, case: DemoCase, *, config: Config | None = None) -> str:
    """Run both halves of ``case`` and render the panel, as text."""
    effective = config or Config()
    explainer = PlanExplainer(adapter=adapter, config=effective)
    layout = await adapter.physical_layout(case.ref)

    naive = await measure(
        adapter,
        explainer,
        case,
        heading="1. Without agentdb — written from a schema dump",
        sql=case.naive_sql,
        note=case.naive_note,
        config=effective,
    )
    grounded = await measure(
        adapter,
        explainer,
        case,
        heading="2. With agentdb — written knowing the physical layout",
        sql=case.grounded_sql,
        note=case.grounded_note,
        config=effective,
    )
    return render(case, layout, naive, grounded)


def render(case: DemoCase, layout: PhysicalLayout, naive: Panel, grounded: Panel) -> str:
    """The panel, as the README shows it."""
    lines = [
        f"agentdb demo — {layout.engine} / {case.ref}",
        "",
        "Question",
        f"  {case.question}",
        "",
        *_layout_lines(layout),
        "",
        *_panel_lines(naive),
        "",
        *_panel_lines(grounded),
        "",
        *_verdict_lines(naive, grounded),
    ]
    return "\n".join(lines) + "\n"


def _layout_lines(layout: PhysicalLayout) -> list[str]:
    """What grounded_context knows and a schema dump cannot say."""
    lines = ["What agentdb reports about the table"]
    if layout.order_by:
        lines.append(f"  ORDER BY ({', '.join(layout.order_by)})")
    if layout.partition_by:
        lines.append(f"  PARTITION BY {', '.join(layout.partition_by)}")
    if layout.approx_rows is not None:
        lines.append(f"  ~{layout.approx_rows:,} rows")
    return lines


def _panel_lines(panel: Panel) -> list[str]:
    lines = [panel.heading, f"  {panel.note}", "", f"  {panel.sql}", ""]
    lines += [f"  answer   {panel.answer}", f"  read     {_cost(panel)}"]
    lines += [
        f"  plan     [{warning.severity.value.upper()}] {warning.code.value} — "
        f"{warning.human_message}"
        for warning in panel.summary.warnings
    ]
    if not panel.summary.warnings:
        lines.append("  plan     no warnings")
    return lines


def _verdict_lines(naive: Panel, grounded: Panel) -> list[str]:
    """The claim, made only when the two halves answered the same question."""
    if naive.answer != grounded.answer:
        return [
            "The two queries did not agree, so there is no comparison to make:",
            f"  {naive.answer} vs {grounded.answer}.",
            "That is a bug in the demo case, not a result. It is printed rather than hidden.",
        ]
    if naive.bytes_read is None or grounded.bytes_read is None or grounded.bytes_read == 0:
        return [f"Same answer ({naive.answer}). The engine reported no byte counts to compare."]

    ratio = naive.bytes_read / grounded.bytes_read
    return [
        f"Same answer ({naive.answer}), "
        f"{_bytes(naive.bytes_read)} read against {_bytes(grounded.bytes_read)} — "
        f"{ratio:.1f}x."
    ]


def _cost(panel: Panel) -> str:
    parts = [_bytes(panel.bytes_read) if panel.bytes_read is not None else "bytes unreported"]
    if panel.rows_read is not None:
        parts.append(f"{panel.rows_read:,} rows")
    if panel.duration_ms is not None:
        parts.append(f"{panel.duration_ms:,} ms")
    return ", ".join(parts)


def _answer(rows: tuple[tuple[object, ...], ...]) -> str:
    """The scalar these questions ask for, or a plain description of what came back."""
    if len(rows) == 1 and len(rows[0]) == 1:
        return str(rows[0][0])
    return f"{len(rows)} rows"


def _bytes(value: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0:
            return (
                f"{value:,.0f} {unit}" if value >= 100.0 or unit == "B" else f"{value:,.1f} {unit}"
            )
        value /= 1024.0
    return f"{value:,.1f} TB"
