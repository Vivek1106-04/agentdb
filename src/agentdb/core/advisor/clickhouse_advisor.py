"""The ClickHouse advisor: sort keys, skip indexes, projections (SPEC §9.1).

Everything here is derived from evidence the engine already reported — the plan's
granule counts, ``system.columns`` sizes, sampled cardinalities, and the query
log. No model is called, and nothing is executed: a recommendation is DDL *text*
until a human elicits its execution (SPEC §13.3).

The ordering rule is ClickHouse's own, and it is the opposite of what a
row-store instinct suggests: **low cardinality first, then increasing**. The
primary index is sparse — one mark per granule — so a leading column with long
runs of equal values lets whole ranges of granules be excluded, while a leading
high-cardinality column scatters matching rows across every granule and prunes
almost nothing. The Databricks advisor deliberately does *not* port this rule,
because liquid clustering does not have a sparse-mark structure to exploit
(SPEC §9.2.B).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from agentdb.adapters import ColumnProfile, PhysicalLayout, RelationRef
from agentdb.config import Config
from agentdb.core.advisor.base import (
    ColumnDemand,
    Confidence,
    Demand,
    EffectEstimate,
    Evidence,
    Kind,
    Recommendation,
    rank,
)
from agentdb.core.plan_ir import PlanSummary
from agentdb.core.query_shape import identifiers_in, mentions

SHADOW_SUFFIX = "__agentdb_shadow"
"""Namespace every table this project creates, so an orphan is identifiable."""

INDEX_PREFIX = "idx_agentdb_"
PROJECTION_PREFIX = "proj_agentdb_"

BLOOM_FPP = 0.01
"""False-positive rate for a proposed bloom filter. One in a hundred granules
read needlessly is the usual sweet spot; below it the index stops being smaller
than the data it is skipping."""

DEFAULT_GRANULARITY = 4
"""Index granularity, in granules. ClickHouse's own default for skip indexes is
1; 4 trades a little precision for a quarter of the index size, which is the
right default for a wide fact table where the index competes for page cache."""

TEXT_TYPES = ("String", "FixedString", "LowCardinality(String)", "Nullable(String)")


@dataclass(frozen=True, slots=True)
class ClickHouseAdvisor:
    """Deterministic physical-design advice for one ClickHouse relation."""

    config: Config = field(default_factory=Config)

    def advise(
        self,
        *,
        ref: RelationRef,
        layout: PhysicalLayout,
        profiles: Sequence[ColumnProfile],
        demand: Demand,
        plan: PlanSummary | None = None,
    ) -> tuple[Recommendation, ...]:
        """Every recommendation the evidence supports, best first.

        ``plan`` is optional and only sharpens what is already known: without it
        the rules still fire from layout and profiles, and the estimates say so
        by naming their method.
        """
        by_name = {profile.name: profile for profile in profiles}
        recommendations = [
            *self._sort_key(ref=ref, layout=layout, profiles=by_name, demand=demand, plan=plan),
            *self._skip_indexes(ref=ref, layout=layout, profiles=by_name, demand=demand),
            *self._projections(ref=ref, layout=layout, demand=demand),
        ]
        return rank(recommendations)

    # -- A. the sort key ---------------------------------------------------

    def _sort_key(
        self,
        *,
        ref: RelationRef,
        layout: PhysicalLayout,
        profiles: Mapping[str, ColumnProfile],
        demand: Demand,
        plan: PlanSummary | None,
    ) -> tuple[Recommendation, ...]:
        existing = layout.order_by or ()
        candidate = self._rank_key_columns(profiles=profiles, demand=demand)
        if not candidate or list(candidate) == list(existing):
            return ()
        distincts = tuple(
            (column, profiles[column].approx_distinct or 0)
            for column in candidate
            if column in profiles and profiles[column].approx_distinct is not None
        )
        evidence = Evidence(
            source="profile+workload" if demand.queries > 1 else "profile+query",
            pruning_ratio=plan.pruning_ratio if plan is not None else None,
            pruning_unit=plan.pruning_unit if plan is not None else None,
            relation_rows=layout.approx_rows,
            distinct_counts=distincts,
            workload_queries=demand.queries if demand.queries > 1 else None,
        )
        return (
            Recommendation(
                kind=Kind.ORDER_BY,
                relation=ref,
                rationale=self._sort_key_rationale(existing, candidate, demand, profiles),
                evidence=evidence,
                expected_effect=EffectEstimate(
                    metric="granules_read",
                    before=plan.pruning_ratio if plan is not None else None,
                    after=self._estimated_pruning(candidate, profiles, layout),
                    method=(
                        "1 / approx_distinct of the leading proposed column, floored at the "
                        "granule size — an upper bound on pruning, not a measurement"
                    ),
                ),
                confidence=Confidence.ESTIMATED,
                ddl=_sort_key_migration(ref, layout, candidate),
                risk_notes=self._sort_key_risks(existing, candidate, demand),
            ),
        )

    def _rank_key_columns(
        self, *, profiles: Mapping[str, ColumnProfile], demand: Demand
    ) -> tuple[str, ...]:
        """Frequently filtered first, then lowest cardinality, truncated by budget.

        The budget bounds the *product* of distinct counts, because that product
        is roughly how many index marks the key generates: past it the sparse
        index degenerates toward one mark per granule and stops pruning.
        """
        candidates = [
            item
            for item in demand.filtered()
            if item.column in profiles and profiles[item.column].approx_distinct is not None
        ]
        if not candidates:
            return ()

        ordered = sorted(
            candidates,
            key=lambda item: (
                -item.filters,
                profiles[item.column].approx_distinct or 0,
                item.column,
            ),
        )
        chosen: list[str] = []
        product = 1.0
        for item in ordered:
            distinct = max(profiles[item.column].approx_distinct or 1, 1)
            if chosen and product * distinct > self.config.sort_key_cardinality_budget:
                break
            product *= distinct
            chosen.append(item.column)
        return tuple(chosen)

    def _sort_key_rationale(
        self,
        existing: Sequence[str],
        candidate: Sequence[str],
        demand: Demand,
        profiles: Mapping[str, ColumnProfile],
    ) -> str:
        lead = candidate[0]
        distinct = profiles[lead].approx_distinct
        current = ", ".join(existing) if existing else "no sort key"
        share = demand.of(lead).share
        return (
            f"{share:.0%} of the queries considered filter on {lead} "
            f"(~{distinct:,} distinct values), and the table is sorted by {current}. "
            "ClickHouse's primary index is sparse, so a leading column with long runs "
            "of equal values excludes whole granule ranges; a key that never leads with "
            "a filtered column prunes nothing no matter how selective the filter is."
        )

    def _sort_key_risks(
        self, existing: Sequence[str], candidate: Sequence[str], demand: Demand
    ) -> tuple[str, ...]:
        notes = [
            "ClickHouse cannot change ORDER BY in place: this is a rebuild, and the "
            "table is unavailable to writers for its duration unless you swap through "
            "a staging table",
        ]
        if not existing:
            return tuple(notes)

        dropped = [column for column in existing if column not in candidate]
        leading = existing[0]
        share = demand.of(leading).share
        if leading in dropped and share >= self.config.sort_key_protect_threshold:
            notes.append(
                f"REGRESSION RISK: {leading} leads the current key and {share:.0%} of the "
                "logged workload filters on it. Those queries lose their pruning under "
                "this key. Consider a projection instead, which keeps both orders."
            )
        elif dropped:
            notes.append(
                f"drops {', '.join(dropped)} from the key; queries filtering only on "
                "those columns will scan more"
            )
        return tuple(notes)

    def _estimated_pruning(
        self,
        candidate: Sequence[str],
        profiles: Mapping[str, ColumnProfile],
        layout: PhysicalLayout,
    ) -> float | None:
        """An upper bound on the granule ratio the proposed key could reach."""
        distinct = profiles[candidate[0]].approx_distinct
        if not distinct or not layout.approx_rows:
            return None
        rows_per_granule = 8192
        granules = max(layout.approx_rows / rows_per_granule, 1.0)
        return min(1.0, max(1.0 / distinct, 1.0 / granules))

    # -- B. skip indexes ---------------------------------------------------

    def _skip_indexes(
        self,
        *,
        ref: RelationRef,
        layout: PhysicalLayout,
        profiles: Mapping[str, ColumnProfile],
        demand: Demand,
    ) -> tuple[Recommendation, ...]:
        """One candidate index per filtered column the sort key does not already serve."""
        key_columns = frozenset(layout.order_by or ()) | frozenset(layout.partition_by or ())
        indexed = frozenset(
            column for index in layout.skip_indexes for column in identifiers_in(index.expression)
        )
        found: list[Recommendation] = []
        for item in demand.filtered():
            profile = profiles.get(item.column)
            if profile is None or item.column in indexed:
                continue
            if any(mentions(term, frozenset({item.column})) for term in key_columns):
                continue
            proposal = self._index_for(item, profile, layout)
            if proposal is None:
                continue
            found.append(_index_recommendation(ref, item, profile, proposal, layout, demand))
        return tuple(found[: self.config.max_index_candidates])

    def _index_for(
        self, item: ColumnDemand, profile: ColumnProfile, layout: PhysicalLayout
    ) -> tuple[str, str] | None:
        """``(index type, expression)`` for this predicate shape, or ``None``.

        The table of SPEC §9.1.B, in the order that matters: text predicates
        first, because a ``LIKE`` on a high-cardinality string column would
        otherwise be handed a bloom filter that its substring search can never use.
        """
        column = item.column
        distinct = profile.approx_distinct
        rows = layout.approx_rows

        if item.text_searches and profile.data_type in TEXT_TYPES:
            return ("tokenbf_v1(32768, 3, 0)", column)
        if distinct is not None and distinct <= self.config.set_index_max_distinct:
            return (f"set({self.config.set_index_max_distinct})", column)
        if (
            item.equality
            and distinct is not None
            and rows
            and distinct / rows > self.config.bloom_min_card_ratio
        ):
            return (f"bloom_filter({BLOOM_FPP})", column)
        if item.ranges:
            return ("minmax", column)
        return None

    # -- C. projections ----------------------------------------------------

    def _projections(
        self, *, ref: RelationRef, layout: PhysicalLayout, demand: Demand
    ) -> tuple[Recommendation, ...]:
        """A projection for a recurring GROUP BY the base sort key cannot serve."""
        recurring = _recurring_shape(demand.group_by_shapes)
        if recurring is None:
            return ()

        key = layout.order_by or ()
        if key and list(key[: len(recurring)]) == list(recurring):
            return ()
        existing = {projection.name for projection in layout.projections}
        name = f"{PROJECTION_PREFIX}{'_'.join(recurring).lower()}"
        if name in existing:
            return ()

        columns = ", ".join(recurring)
        return (
            Recommendation(
                kind=Kind.PROJECTION,
                relation=ref,
                rationale=(
                    f"GROUP BY ({columns}) recurs in the workload and the sort key "
                    f"({', '.join(key) or 'none'}) does not lead with it, so every one of "
                    "those queries aggregates over a full read. A projection stores a "
                    "second copy in that order and ClickHouse picks it automatically."
                ),
                evidence=Evidence(
                    source="workload",
                    relation_rows=layout.approx_rows,
                    workload_queries=demand.queries,
                ),
                expected_effect=EffectEstimate(
                    metric="granules_read",
                    before=1.0,
                    after=None,
                    method=(
                        "not estimated: the reduction depends on the aggregation's "
                        "grouping factor, which no statistic here reports"
                    ),
                ),
                confidence=Confidence.HEURISTIC,
                ddl=(
                    f"ALTER TABLE {ref} ADD PROJECTION {name} (\n"
                    f"  SELECT {columns}, count()\n"
                    f"  GROUP BY {columns}\n"
                    f");\n"
                    f"ALTER TABLE {ref} MATERIALIZE PROJECTION {name};"
                ),
                risk_notes=(
                    "a projection is a second physical copy of the columns it names: "
                    "disk grows and every insert writes twice",
                    "MATERIALIZE PROJECTION rewrites existing parts in the background "
                    "and competes with merges while it runs",
                ),
            ),
        )


def _index_recommendation(
    ref: RelationRef,
    item: ColumnDemand,
    profile: ColumnProfile,
    proposal: tuple[str, str],
    layout: PhysicalLayout,
    demand: Demand,
) -> Recommendation:
    index_type, expression = proposal
    name = f"{INDEX_PREFIX}{item.column.lower()}"
    selectivity = 1.0 / profile.approx_distinct if profile.approx_distinct else None
    return Recommendation(
        kind=Kind.SKIP_INDEX,
        relation=ref,
        rationale=(
            f"{demand.of(item.column).share:.0%} of the queries considered filter on "
            f"{item.column}, which is in neither the sort key nor the partition key, so "
            f"the sparse index cannot exclude a single granule for them. A "
            f"{index_type.split('(')[0]} index on it can."
        ),
        evidence=Evidence(
            source="profile+workload" if demand.queries > 1 else "profile+query",
            relation_rows=layout.approx_rows,
            distinct_counts=(
                ((item.column, profile.approx_distinct),) if profile.approx_distinct else ()
            ),
            workload_queries=demand.queries if demand.queries > 1 else None,
        ),
        expected_effect=EffectEstimate(
            metric="granules_read",
            before=1.0,
            after=selectivity,
            method=(
                "1 / approx_distinct from a sampled profile, as an upper bound on "
                "selectivity; the index prunes at granule granularity, so the realised "
                "figure is higher"
            ),
        ),
        confidence=Confidence.ESTIMATED,
        ddl=(
            f"ALTER TABLE {ref}\n"
            f"  ADD INDEX {name} {expression} TYPE {index_type} "
            f"GRANULARITY {DEFAULT_GRANULARITY};\n"
            f"ALTER TABLE {ref} MATERIALIZE INDEX {name};"
        ),
        risk_notes=(
            "MATERIALIZE INDEX rewrites every existing part; on a large table it runs "
            "for a long time and competes with merges",
            "a skip index that does not prune still costs write throughput and disk",
        ),
    )


def _sort_key_migration(ref: RelationRef, layout: PhysicalLayout, candidate: Sequence[str]) -> str:
    """The honest DDL: ClickHouse cannot reorder a table in place."""
    columns = ", ".join(candidate)
    partition = f"PARTITION BY {', '.join(layout.partition_by)}\n" if layout.partition_by else ""
    shadow = f"{ref.name}{SHADOW_SUFFIX}_reordered"
    return (
        f"-- ORDER BY cannot be altered in place; this is a rebuild.\n"
        f"CREATE TABLE {ref.namespace}.{shadow}\n"
        f"ENGINE = {layout.table_engine or 'MergeTree'}\n"
        f"{partition}"
        f"ORDER BY ({columns})\n"
        f"AS SELECT * FROM {ref} WHERE 0;\n"
        f"INSERT INTO {ref.namespace}.{shadow} SELECT * FROM {ref};\n"
        f"EXCHANGE TABLES {ref} AND {ref.namespace}.{shadow};\n"
        f"-- verify, then: DROP TABLE {ref.namespace}.{shadow};"
    )


def _recurring_shape(shapes: Sequence[tuple[str, ...]]) -> tuple[str, ...] | None:
    """The most common GROUP BY shape, if any shape appears more than once."""
    counts: dict[tuple[str, ...], int] = {}
    for shape in shapes:
        if shape:
            counts[shape] = counts.get(shape, 0) + 1
    if not counts:
        return None
    best, occurrences = max(counts.items(), key=lambda item: (item[1], -len(item[0])))
    return best if occurrences > 1 else None
