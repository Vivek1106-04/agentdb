"""Deterministic query rewrites, shared by both engines (SPEC §9.1.D, §9.2.E).

Every rewrite here is mechanical: the rewritten SQL is produced by transforming
the parsed query, not by asking a model, and each one either applies exactly or
does not fire. That is the bar a rewrite has to clear before it goes in front of
an agent — a wrong rewrite costs more trust than a missing one, because the agent
cannot tell which of the two it is holding.

Where a fix is *not* mechanical the recommendation still exists but
``rewritten_sql`` stays ``None`` and the rationale says what the author has to
decide. ``SELECT *`` on a hundred-column table is the case that matters: naming
the columns is the fix, but only the person asking the question knows which ones
they meant, and a rewrite that guessed would silently change the answer.

Dialect facts are read from :class:`~agentdb.adapters.models.DialectRules`,
never hardcoded. The two engines quote differently and reserve different words,
and a rule that assumed one engine's list would emit broken SQL on the other.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlglot
from sqlglot import exp

from agentdb.adapters import DialectRules, PhysicalLayout, RelationDetail, RelationRef
from agentdb.core.advisor.base import (
    Confidence,
    EffectEstimate,
    Evidence,
    Kind,
    Recommendation,
)
from agentdb.core.query_shape import DIALECTS, analyze

WIDE_SELECT_STAR_METHOD = (
    "compressed bytes of the columns the query actually references, over compressed "
    "bytes of every column, from the engine's own catalogue"
)


def rewrites(
    *,
    sql: str,
    ref: RelationRef,
    rules: DialectRules,
    layout: PhysicalLayout | None = None,
    detail: RelationDetail | None = None,
) -> tuple[Recommendation, ...]:
    """Every rewrite the query's own text supports, in the order they were found."""
    shape = analyze(sql, rules.engine)
    if not shape.parsed:
        return ()

    found = [
        *_qualify(sql=sql, ref=ref, rules=rules),
        *_unwrap_temporal(sql=sql, ref=ref, rules=rules, layout=layout),
        *_quote_reserved(sql=sql, ref=ref, rules=rules, detail=detail),
        *_narrow_projection(
            sql=sql,
            ref=ref,
            rules=rules,
            detail=detail,
            shape_columns=_referenced(sql, rules.engine),
        ),
    ]
    return tuple(found)


def rewritten_query(*, sql: str, ref: RelationRef, rules: DialectRules) -> str | None:
    """``sql`` with every mechanical fix applied at once, or ``None`` if none apply.

    The per-issue recommendations each fix one thing and explain it, which is
    what a reader wants; an agent about to re-run the query wants the composed
    result rather than a stack of alternatives it has to merge itself.
    """
    parsed = analyze(sql, rules.engine)
    if not parsed.parsed:
        return None

    tree = sqlglot.parse_one(sql, dialect=DIALECTS.get(rules.engine))
    fixed = tree
    if ref.catalog is not None:
        fixed = fixed.transform(lambda node: _qualified(node, ref))
    fixed = fixed.transform(_range_for_year)

    rendered = fixed.sql(dialect=DIALECTS.get(rules.engine))
    original = tree.sql(dialect=DIALECTS.get(rules.engine))
    return rendered if rendered != original else None


def _qualify(*, sql: str, ref: RelationRef, rules: DialectRules) -> tuple[Recommendation, ...]:
    """Under-qualified table names, made whole (SPEC §9.2.E).

    The most common Databricks agent failure, and a correctness bug rather than a
    performance one: a two-part name resolves against session ``USE`` state that a
    stateless connection does not have, so the query either fails or — worse —
    reads a same-named table in another schema.
    """
    if ref.catalog is None:
        return ()

    tree = sqlglot.parse_one(sql, dialect=DIALECTS.get(rules.engine))
    under = [
        table
        for table in tree.find_all(exp.Table)
        if table.name == ref.name and not (table.catalog and table.db)
    ]
    if not under:
        return ()

    rewritten = tree.transform(lambda node: _qualified(node, ref))
    return (
        Recommendation(
            kind=Kind.REWRITE,
            relation=ref,
            rationale=(
                f"{ref.name} is written without its catalog and schema. A two-part or "
                "bare name resolves against session state this connection does not "
                f"have, so the query reads whatever {ref.name} the session's current "
                "schema happens to hold — or fails outright."
            ),
            evidence=Evidence(source="query"),
            expected_effect=EffectEstimate(
                metric="files_read",
                before=None,
                after=None,
                method="no performance effect: this is a correctness fix",
            ),
            confidence=Confidence.MEASURED,
            rewritten_sql=rewritten.sql(dialect=DIALECTS.get(rules.engine)),
            risk_notes=(),
        ),
    )


def _unwrap_temporal(
    *, sql: str, ref: RelationRef, rules: DialectRules, layout: PhysicalLayout | None
) -> tuple[Recommendation, ...]:
    """``year(ts) = 2026`` into a half-open range on the raw column.

    A predicate the engine cannot see through prunes nothing: the key or the
    file statistics are on ``ts``, and ``year(ts)`` is an expression the pruner
    has no bounds for. The rewritten form is exactly equivalent and prunes.
    """
    tree = sqlglot.parse_one(sql, dialect=DIALECTS.get(rules.engine))
    replacements: list[tuple[str, int]] = []

    for predicate in tree.find_all(exp.EQ):
        column, year = _year_equality(predicate)
        if column is None or year is None:
            continue
        replacements.append((column, year))

    if not replacements:
        return ()

    rewritten = tree.transform(_range_for_year)
    column, year = replacements[0]
    keyed = _is_key_column(column, layout)
    key_note = f" — and {column} carries this table's pruning" if keyed else ""
    return (
        Recommendation(
            kind=Kind.REWRITE,
            relation=ref,
            rationale=(
                f"the filter wraps {column} in a function, so the engine has no bounds "
                f"to prune with{key_note}. A half-open range on the raw column is exactly "
                f"equivalent and can prune. Rewritten for {year}."
            ),
            evidence=Evidence(source="query+layout" if layout is not None else "query"),
            expected_effect=EffectEstimate(
                metric="granules_read" if rules.engine == "clickhouse" else "files_read",
                before=1.0,
                after=None,
                method=(
                    "not estimated: how much the range prunes depends on how the values "
                    "distribute across the key, which the query text does not say"
                ),
            ),
            confidence=Confidence.MEASURED if keyed else Confidence.HEURISTIC,
            rewritten_sql=rewritten.sql(dialect=DIALECTS.get(rules.engine)),
        ),
    )


def _quote_reserved(
    *, sql: str, ref: RelationRef, rules: DialectRules, detail: RelationDetail | None
) -> tuple[Recommendation, ...]:
    """Identifiers that collide with this engine's reserved words (SPEC §9.1.D)."""
    if detail is None or not rules.reserved_words:
        return ()

    shape = analyze(sql, rules.engine)
    colliding = sorted(
        column.name
        for column in detail.columns
        if column.name.upper() in {word.upper() for word in rules.reserved_words}
        and column.name in (shape.filter_columns | frozenset(shape.group_by_columns))
    )
    if not colliding:
        return ()

    quote = rules.identifier_quote
    quoted = ", ".join(f"{quote}{name}{quote}" for name in colliding)
    return (
        Recommendation(
            kind=Kind.REWRITE,
            relation=ref,
            rationale=(
                f"{', '.join(colliding)} {'is' if len(colliding) == 1 else 'are'} reserved on "
                f"{rules.engine} {rules.version}. Unquoted, the parser reads them as syntax "
                f"rather than as columns. Written as {quoted} they resolve."
            ),
            evidence=Evidence(source="dialect"),
            expected_effect=EffectEstimate(
                metric="files_read",
                before=None,
                after=None,
                method="no performance effect: this is a correctness fix",
            ),
            confidence=Confidence.MEASURED,
        ),
    )


def _narrow_projection(
    *,
    sql: str,
    ref: RelationRef,
    rules: DialectRules,
    detail: RelationDetail | None,
    shape_columns: frozenset[str],
) -> tuple[Recommendation, ...]:
    """``SELECT *`` on a wide table — named, costed, and left for a human to decide.

    No ``rewritten_sql``. The fix is to name the columns, and only the person who
    asked the question knows which ones they meant; a rewrite that guessed from
    the ``WHERE`` clause would return a different answer while looking correct.
    """
    shape = analyze(sql, rules.engine)
    if not shape.selects_star or detail is None:
        return ()

    total = sum(column.compressed_bytes or 0 for column in detail.columns)
    referenced = sum(
        column.compressed_bytes or 0 for column in detail.columns if column.name in shape_columns
    )
    ratio = referenced / total if total else None

    return (
        Recommendation(
            kind=Kind.REWRITE,
            relation=ref,
            rationale=(
                f"SELECT * reads all {len(detail.columns)} columns of {ref.name}. A columnar "
                "engine reads only the columns a query names, so the width of the projection "
                "is the cost of the query — the filter is not what makes this expensive."
            ),
            evidence=Evidence(source="catalogue"),
            expected_effect=EffectEstimate(
                metric="bytes_read",
                before=1.0,
                after=ratio,
                method=WIDE_SELECT_STAR_METHOD,
            ),
            confidence=Confidence.ESTIMATED if ratio is not None else Confidence.HEURISTIC,
            risk_notes=(
                "naming columns changes the result shape: only the author knows which "
                "columns the answer needs, so this is not rewritten automatically",
            ),
        ),
    )


def _referenced(sql: str, engine: str) -> frozenset[str]:
    """Every column the query names anywhere outside the star."""
    tree = sqlglot.parse_one(sql, dialect=DIALECTS.get(engine))
    return frozenset(column.name for column in tree.find_all(exp.Column))


def _qualified(node: exp.Expression, ref: RelationRef) -> exp.Expression:
    if not isinstance(node, exp.Table) or node.name != ref.name:
        return node
    return exp.Table(
        this=exp.to_identifier(ref.name),
        db=exp.to_identifier(ref.namespace),
        catalog=exp.to_identifier(ref.catalog) if ref.catalog else None,
        alias=node.args.get("alias"),
    )


def _year_equality(predicate: exp.EQ) -> tuple[str | None, int | None]:
    """``year(col) = 2026`` decomposed, or ``(None, None)``."""
    left, right = predicate.this, predicate.expression
    if not isinstance(left, exp.Year | exp.Anonymous):
        return None, None
    if isinstance(left, exp.Anonymous) and left.name.lower() not in {"year", "toyear"}:
        return None, None
    columns = list(left.find_all(exp.Column))
    if len(columns) != 1 or not isinstance(right, exp.Literal) or not right.is_int:
        return None, None
    return columns[0].name, int(right.name)


def _range_for_year(node: exp.Expression) -> exp.Expression:
    if not isinstance(node, exp.EQ):
        return node
    column, year = _year_equality(node)
    if column is None or year is None:
        return node
    lower = exp.GTE(this=exp.column(column), expression=exp.Literal.string(f"{year}-01-01"))
    upper = exp.LT(this=exp.column(column), expression=exp.Literal.string(f"{year + 1}-01-01"))
    return exp.And(this=lower, expression=upper)


def _is_key_column(column: str, layout: PhysicalLayout | None) -> bool:
    """Whether this column carries the table's pruning — sort key, partition or cluster."""
    if layout is None:
        return False
    key_terms: Sequence[str] = (
        *(layout.order_by or ()),
        *(layout.partition_by or ()),
        *(layout.clustering_columns or ()),
        *(layout.stats_columns or ()),
    )
    return any(column in term for term in key_terms)
