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
    """Relations referenced by bare name, in the order they appear — the
    left-to-right order a join warning reasons about."""

    qualified_tables: tuple[str, ...] = ()
    """The same relations, as the query actually wrote them.

    Kept alongside the bare names because under Unity Catalog the qualification
    *is* the fact: ``FROM lineitem`` and ``FROM samples.tpch.lineitem`` name the
    same table only if the session's ``USE`` context cooperates (SPEC §8.2)."""

    filter_columns: frozenset[str] = frozenset()
    """Columns constrained in ``WHERE`` or ``PREWHERE``, including inside functions."""

    equality_columns: frozenset[str] = frozenset()
    """Columns constrained by ``=`` or ``IN``.

    Separated from :attr:`range_columns` because the advisors ask different
    questions of each: an equality predicate on a high-cardinality column is what
    a bloom filter is for, and a range predicate is what a min/max index and a
    sort-key prefix are for (SPEC §9.1.B)."""

    range_columns: frozenset[str] = frozenset()
    """Columns constrained by ``<``, ``<=``, ``>``, ``>=`` or ``BETWEEN``."""

    wrapped_filter_columns: frozenset[str] = frozenset()
    """Filter columns reached only through a function call.

    ``year(l_shipdate) = 1995`` filters on ``l_shipdate`` and prunes nothing:
    the engine cannot use a key or a statistic on a column it never sees bare.
    Both advisors' rewrite rules turn on this (SPEC §9.1.D, §9.2.E)."""

    text_search_columns: frozenset[str] = frozenset()
    """Columns under ``LIKE``, ``ILIKE`` or a token/substring search — the
    predicates the text skip-index types exist for."""

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
        qualified_tables=_unique(_qualified(table) for table in tree.find_all(exp.Table)),
        filter_columns=frozenset(_filter_columns(tree)),
        equality_columns=frozenset(_predicate_columns(tree, (exp.EQ, exp.In))),
        range_columns=frozenset(
            _predicate_columns(tree, (exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Between))
        ),
        wrapped_filter_columns=frozenset(_wrapped_filter_columns(tree)),
        text_search_columns=frozenset(_predicate_columns(tree, (exp.Like, exp.ILike))),
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


def _predicate_columns(tree: exp.Expr, kinds: tuple[type[exp.Expression], ...]) -> set[str]:
    """Columns constrained by one family of predicate, inside filtering clauses only.

    A join condition is not a filter: ``ON a.id = b.id`` says nothing about which
    values a scan can skip, and counting it as an equality predicate would send
    the sort-key rule chasing a key column.
    """
    columns: set[str] = set()
    for clause in (*tree.find_all(exp.Where), *tree.find_all(exp.PreWhere)):
        for kind in kinds:
            for predicate in clause.find_all(kind):
                columns.update(column.name for column in predicate.find_all(exp.Column))
    return columns


def _wrapped_filter_columns(tree: exp.Expr) -> set[str]:
    """Filter columns that only ever appear inside a function call."""
    bare: set[str] = set()
    wrapped: set[str] = set()
    for clause in (*tree.find_all(exp.Where), *tree.find_all(exp.PreWhere)):
        for column in clause.find_all(exp.Column):
            target = wrapped if _under_function(column, clause) else bare
            target.add(column.name)
    return wrapped - bare


def _under_function(column: exp.Expression, clause: exp.Expression) -> bool:
    """Whether ``column`` is reached only through a real function call.

    ``sqlglot`` models connectives and comparisons as ``Func`` subclasses too —
    ``AND`` is one — so a bare ``isinstance`` check would call every filtered
    column wrapped. Only nodes that are functions and *not* operators count.
    """
    node = column.parent
    while node is not None and node is not clause:
        if isinstance(node, exp.Func) and not isinstance(
            node, exp.Binary | exp.Connector | exp.Predicate | exp.Unary
        ):
            return True
        node = node.parent
    return False


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


def _qualified(table: exp.Table) -> str:
    """A table reference with every name part the query supplied, and no more."""
    return ".".join(part for part in (table.catalog, table.db, table.name) if part)


def _unique(names: Iterable[str]) -> tuple[str, ...]:
    """Deduplicate while keeping first-appearance order — plans are read in order."""
    return tuple(dict.fromkeys(names))


def identifiers_in(expression: str) -> frozenset[str]:
    """The bare identifiers inside a key expression.

    Key terms are expressions as often as columns — ``toYYYYMM(EventDate)`` on
    ClickHouse, ``date_trunc('month', l_shipdate)`` on Databricks — so a rule
    asking "does this filter touch the key" has to look inside the term rather
    than compare strings. Deliberately lexical: a key expression comes from the
    engine's own catalogue, and parsing it as SQL would fail on dialect-specific
    forms that carry no extra information here.
    """
    token: list[str] = []
    found: set[str] = set()
    for char in expression:
        if char.isalnum() or char == "_":
            token.append(char)
            continue
        if token:
            found.add("".join(token))
            token = []
    if token:
        found.add("".join(token))
    return frozenset(found)


def mentions(key_expression: str, columns: frozenset[str]) -> bool:
    """Whether any of ``columns`` appears inside ``key_expression``."""
    return bool(columns & identifiers_in(key_expression))
