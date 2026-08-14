"""What the query *asks for*, as distinct from what the engine plans to do.

A plan says how many granules survived pruning; it does not say that the agent
filtered on a column that is not in the sort key, grouped by something with forty
million distinct values, or wrote ``SELECT *`` on a 105-column table. Those are
properties of the text, and half the warnings in SPEC §7 need them.

Parsing is delegated to ``sqlglot`` rather than hand-rolled: dialect-correct SQL
parsing is a solved problem and a regex approximation of it would be wrong in
exactly the cases that matter — subqueries, CTEs, quoted identifiers.

An unparseable query yields an empty shape with :attr:`QueryShape.parsed` false,
and every rule that depends on the text is then skipped. A warning derived from a
misparse is worse than a missing warning: it teaches an agent to distrust all of
them.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import sqlglot
from sqlglot import exp

DIALECTS = {"clickhouse": "clickhouse", "databricks": "databricks"}
"""Engine name to ``sqlglot`` dialect. Both engines in scope are supported."""


@dataclass(frozen=True, slots=True)
class QueryShape:
    """The structural facts of one query."""

    parsed: bool
    tables: tuple[str, ...] = ()
    """Relations referenced, in the order they appear — the left-to-right order a
    join warning reasons about."""

    filter_columns: frozenset[str] = frozenset()
    """Columns constrained in ``WHERE`` or ``PREWHERE``, including inside functions."""

    group_by_columns: tuple[str, ...] = ()
    order_by_columns: tuple[str, ...] = ()
    joined_tables: tuple[str, ...] = ()
    """Right-hand sides of joins. ClickHouse builds these in memory."""

    selects_star: bool = False
    has_limit: bool = False
    has_aggregate: bool = False

    @property
    def is_single_relation(self) -> bool:
        return len(self.tables) == 1


UNPARSED = QueryShape(parsed=False)
"""What a query nobody could parse yields. Every text-derived rule skips it."""


def analyze(sql: str, engine: str) -> QueryShape:
    """Extract the structural facts of ``sql``, or :data:`UNPARSED`."""
    try:
        tree = sqlglot.parse_one(sql, dialect=DIALECTS.get(engine))
    except sqlglot.ParseError:
        return UNPARSED

    return QueryShape(
        parsed=True,
        tables=_unique(table.name for table in tree.find_all(exp.Table)),
        filter_columns=frozenset(_filter_columns(tree)),
        group_by_columns=_group_by(tree),
        order_by_columns=_order_by(tree),
        joined_tables=_unique(
            table.name for join in tree.find_all(exp.Join) for table in join.find_all(exp.Table)
        ),
        selects_star=any(True for _ in tree.find_all(exp.Star)),
        has_limit=tree.args.get("limit") is not None,
        has_aggregate=any(True for _ in tree.find_all(exp.AggFunc)),
    )


def _filter_columns(tree: exp.Expr) -> set[str]:
    """Every column mentioned in a filtering clause.

    Columns inside a function still count: ``toDate(EventTime) = today()`` filters
    on ``EventTime``, and whether the engine can prune on it is exactly the
    question the sort-key rules ask.
    """
    columns: set[str] = set()
    for clause in (*tree.find_all(exp.Where), *tree.find_all(exp.PreWhere)):
        columns.update(column.name for column in clause.find_all(exp.Column))
    return columns


def _group_by(tree: exp.Expr) -> tuple[str, ...]:
    group = tree.find(exp.Group)
    if group is None:
        return ()
    return _unique(
        column.name for term in group.expressions for column in term.find_all(exp.Column)
    )


def _order_by(tree: exp.Expr) -> tuple[str, ...]:
    order = tree.find(exp.Order)
    if order is None:
        return ()
    return _unique(
        column.name for term in order.expressions for column in term.find_all(exp.Column)
    )


def _unique(names: Iterable[str]) -> tuple[str, ...]:
    """Deduplicate while keeping first-appearance order — plans are read in order."""
    return tuple(dict.fromkeys(names))
