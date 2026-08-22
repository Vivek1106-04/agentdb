"""Literal-parameterized SQL, for deduplicating exemplars (SPEC §10.2).

Two agents asking "revenue in 1994" and "revenue in 1995" write the same query
with one literal changed. Storing both teaches the store nothing and crowds
retrieval with near-duplicates, so what is stored beside the original SQL is a
normalized form with every literal replaced by a placeholder.

Parsing is delegated to ``sqlglot`` for the same reason as
:mod:`agentdb.core.query_shape`: a regex that strips quoted strings is wrong on
exactly the queries that matter. Unparseable SQL still normalizes — to
whitespace-collapsed, case-folded text — because refusing to record an exemplar
whose SQL this project cannot parse would silently drop the failures that are
most worth remembering.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

from agentdb.core.query_shape import DIALECTS

PLACEHOLDER = "?"
"""What every literal collapses to. Positional, so two queries differing only in
their constants normalize identically regardless of the constants' types.

Emitted as a bare token rather than as ``sqlglot``'s own placeholder node,
which renders dialect-specifically — ``?`` on Databricks but ``{?: }`` on
ClickHouse. A dedup key that changes shape per engine would split one query into
two exemplars the moment the same question was asked of the other engine.

Identifier case is left alone for the same reason it matters at execution:
ClickHouse identifiers are case-sensitive, so folding ``CounterID`` to
``counterid`` would merge two columns that the engine considers distinct.
"""


def normalize_sql(sql: str, engine: str) -> str:
    """Return ``sql`` with literals parameterized and formatting canonicalized."""
    try:
        tree = sqlglot.parse_one(sql, dialect=DIALECTS.get(engine))
    except sqlglot.ParseError:
        return _flatten(sql)

    parameterized = tree.transform(_placeholder)
    return parameterized.sql(dialect=DIALECTS.get(engine), comments=False)


def _placeholder(node: exp.Expression) -> exp.Expression:
    """Replace a literal with a placeholder, leaving identifiers untouched.

    ``LIMIT 10`` and ``GROUP BY 1`` keep their numbers: those literals are part
    of the query's shape rather than its parameters, and collapsing them would
    make a top-10 and a top-1000 query indistinguishable.
    """
    if not isinstance(node, exp.Literal):
        return node
    if isinstance(node.parent, exp.Limit | exp.Offset | exp.Ordered | exp.Group):
        return node
    return exp.Var(this=PLACEHOLDER)


def _flatten(sql: str) -> str:
    return " ".join(sql.split()).casefold()
