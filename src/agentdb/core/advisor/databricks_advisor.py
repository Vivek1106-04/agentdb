"""The Databricks advisor: skipping statistics, clustering, compaction (SPEC §9.2).

Keep the asymmetry with §9.1 rather than smoothing it over. Databricks already
tunes physical layout on managed tables through predictive optimization, so the
centre of gravity here is not "pick a better clustering key" — the engine may be
doing that already. It is the two things the engine does not do: **tell the agent
which columns are skippable at all**, and fix a query shape that cannot use the
layout that exists.

Two rules are deliberately *not* ported from the ClickHouse advisor:

* **Cardinality ordering.** §9.1.A leads a sort key with the lowest-cardinality
  column because ClickHouse's sparse primary index prunes on long runs of equal
  values. Liquid clustering has no sparse-mark structure to exploit, so
  clustering keys are ranked by filter frequency first, then selectivity.
* **Confident effect estimates.** Delta reports files, not granules, and a file
  count says nothing about how much of each file was read. Estimates here are
  bounded by what ``DESCRIBE DETAIL`` and the query history actually carry, and
  latency is never claimed at all — a shared warehouse's wall clock is not this
  project's to attribute.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from agentdb.adapters import ColumnProfile, PhysicalLayout, RelationDetail, RelationRef
from agentdb.config import Config
from agentdb.core.advisor.base import (
    Confidence,
    Demand,
    EffectEstimate,
    Evidence,
    Kind,
    Recommendation,
    rank,
)
from agentdb.core.plan_ir import PlanSummary, WarningCode

SHADOW_SUFFIX = "__agentdb_shadow"

NOT_RETROACTIVE = (
    "delta.dataSkippingStatsColumns is not retroactive: files already written carry "
    "no statistics for the newly named columns until they are rewritten, so OPTIMIZE "
    "(or a full rewrite) is part of the change, not an optional follow-up"
)

WIDER_STATS_COSTS = (
    "widening the statistics set increases write cost and Delta log size; every "
    "column named is one more min/max pair per file in the transaction log"
)

MANAGED_TABLE_NOTE = (
    "this table is managed, so predictive optimization may already be choosing its "
    "clustering key and running OPTIMIZE unasked. Verify what the engine is doing "
    "before acting on this — advising against a tuner that is already working is "
    "worse than advising nothing"
)


@dataclass(frozen=True, slots=True)
class DatabricksAdvisor:
    """Deterministic physical-design advice for one Delta relation."""

    config: Config = field(default_factory=Config)

    def advise(
        self,
        *,
        ref: RelationRef,
        layout: PhysicalLayout,
        detail: RelationDetail,
        profiles: Sequence[ColumnProfile],
        demand: Demand,
        plan: PlanSummary | None = None,
    ) -> tuple[Recommendation, ...]:
        """Every recommendation the evidence supports, best first."""
        by_name = {profile.name: profile for profile in profiles}
        return rank(
            [
                *self._stats_columns(ref=ref, layout=layout, detail=detail, demand=demand),
                *self._clustering(
                    ref=ref, layout=layout, profiles=by_name, demand=demand, plan=plan
                ),
                *self._compaction(ref=ref, layout=layout, plan=plan),
                *self._join_strategy(ref=ref, layout=layout, plan=plan),
            ]
        )

    # -- A. data-skipping statistics coverage ------------------------------

    def _stats_columns(
        self,
        *,
        ref: RelationRef,
        layout: PhysicalLayout,
        detail: RelationDetail,
        demand: Demand,
    ) -> tuple[Recommendation, ...]:
        """The headline recommendation, and the one nobody else ships.

        A filter on a column outside the statistics set cannot skip a single
        file, however selective it is. Nothing in a ``CREATE TABLE`` says which
        columns those are, which is why an agent — and most humans — never think
        to ask.
        """
        covered = _covered_columns(layout, detail)
        if covered is None:
            return ()

        uncovered = tuple(
            item.column
            for item in demand.filtered()
            if item.column in _column_names(detail) and item.column not in covered
        )
        if not uncovered:
            return ()

        proposed = tuple(dict.fromkeys((*sorted(covered), *uncovered)))
        return (
            Recommendation(
                kind=Kind.STATS_COLUMNS,
                relation=ref,
                rationale=(
                    f"{', '.join(uncovered)} {'is' if len(uncovered) == 1 else 'are'} filtered "
                    f"by the workload but outside this table's data-skipping statistics set "
                    f"({_coverage_description(layout)}). Delta keeps per-file min/max only for "
                    "columns in that set, so a filter on any other column skips no files at "
                    "all — the scan reads every file and discards rows afterwards."
                ),
                evidence=Evidence(
                    source="layout+workload" if demand.queries > 1 else "layout+query",
                    relation_rows=layout.approx_rows,
                    workload_queries=demand.queries if demand.queries > 1 else None,
                ),
                expected_effect=EffectEstimate(
                    metric="files_read",
                    before=1.0,
                    after=None,
                    method=(
                        "not estimated: a column with no statistics prunes nothing, and how "
                        "much it would prune once collected depends on the values' clustering "
                        "across files, which no statistic here reports. Validate on a shadow "
                        "table to make this measured"
                    ),
                ),
                confidence=Confidence.HEURISTIC,
                ddl=(
                    f"ALTER TABLE {ref} SET TBLPROPERTIES (\n"
                    f"  'delta.dataSkippingStatsColumns' = '{', '.join(proposed)}'\n"
                    f");\n"
                    f"-- statistics apply to files written after the change:\n"
                    f"OPTIMIZE {ref};"
                ),
                risk_notes=(NOT_RETROACTIVE, WIDER_STATS_COSTS),
            ),
        )

    # -- B. liquid clustering ----------------------------------------------

    def _clustering(
        self,
        *,
        ref: RelationRef,
        layout: PhysicalLayout,
        profiles: Mapping[str, ColumnProfile],
        demand: Demand,
        plan: PlanSummary | None,
    ) -> tuple[Recommendation, ...]:
        """Rank by filter frequency, then selectivity — deliberately not §9.1.A's rule."""
        candidates = [
            item
            for item in demand.filtered()
            if item.column in profiles and profiles[item.column].approx_distinct
        ]
        if not candidates:
            return ()

        ordered = sorted(
            candidates,
            key=lambda item: (
                -item.filters,
                -(1.0 / (profiles[item.column].approx_distinct or 1)),
                item.column,
            ),
        )
        proposed = tuple(item.column for item in ordered[: self.config.clustering_key_max_columns])
        existing = layout.clustering_columns or ()
        if list(proposed) == list(existing):
            return ()

        legacy = layout.zorder_columns or ()
        return (
            Recommendation(
                kind=Kind.CLUSTER_BY,
                relation=ref,
                rationale=self._clustering_rationale(proposed, existing, legacy, demand, layout),
                evidence=Evidence(
                    source="profile+workload" if demand.queries > 1 else "profile+query",
                    pruning_ratio=plan.pruning_ratio if plan is not None else None,
                    pruning_unit=plan.pruning_unit if plan is not None else None,
                    relation_rows=layout.approx_rows,
                    distinct_counts=tuple(
                        (column, profiles[column].approx_distinct or 0) for column in proposed
                    ),
                    workload_queries=demand.queries if demand.queries > 1 else None,
                ),
                expected_effect=EffectEstimate(
                    metric="files_read",
                    before=plan.pruning_ratio if plan is not None else None,
                    after=None,
                    method=(
                        "not estimated: file pruning under liquid clustering depends on how "
                        "the values distribute across files after reclustering, which is not "
                        "derivable from the current layout"
                    ),
                ),
                confidence=Confidence.HEURISTIC,
                ddl=(
                    f"ALTER TABLE {ref} CLUSTER BY ({', '.join(proposed)});\n"
                    f"OPTIMIZE {ref};  -- reclusters existing data"
                ),
                risk_notes=self._clustering_risks(layout, legacy),
            ),
        )

    def _clustering_rationale(
        self,
        proposed: Sequence[str],
        existing: Sequence[str],
        legacy: Sequence[str],
        demand: Demand,
        layout: PhysicalLayout,
    ) -> str:
        lead = proposed[0]
        current = ", ".join(existing) if existing else "no clustering key"
        note = ""
        if legacy:
            note = (
                f" The table is Z-ORDERed by {', '.join(legacy)}; liquid clustering replaces "
                "Z-ORDER rather than complementing it, and the two are mutually exclusive."
            )
        if layout.is_managed:
            note += f" {MANAGED_TABLE_NOTE.split(':')[0]}."
        return (
            f"{demand.of(lead).share:.0%} of the queries considered filter on {lead}, and the "
            f"table is clustered by {current}. Ranked by filter frequency and then selectivity, "
            "which is deliberately not the cardinality-ordering rule the ClickHouse advisor "
            "uses: that rule exists for a sparse primary index and does not transfer to "
            f"liquid clustering.{note}"
        )

    def _clustering_risks(self, layout: PhysicalLayout, legacy: Sequence[str]) -> tuple[str, ...]:
        notes = [
            "OPTIMIZE rewrites data to recluster it: it costs compute proportional to the "
            "table and competes with concurrent writers",
        ]
        if legacy:
            notes.append(
                "migrating off Z-ORDER is one-way for this table: CLUSTER BY and ZORDER BY "
                "cannot both apply"
            )
        if layout.is_managed:
            notes.append(MANAGED_TABLE_NOTE)
        return tuple(notes)

    # -- C. compaction -----------------------------------------------------

    def _compaction(
        self, *, ref: RelationRef, layout: PhysicalLayout, plan: PlanSummary | None
    ) -> tuple[Recommendation, ...]:
        """From ``numFiles`` and average size alone — never from a latency claim."""
        files = layout.num_files
        average = layout.avg_file_bytes
        if files is None or average is None:
            return ()
        small = (
            files >= self.config.small_file_count_threshold
            and average < self.config.small_file_bytes_threshold
        )
        if not small:
            return ()

        target_files = max(int(files * average / self.config.small_file_bytes_threshold), 1)
        return (
            Recommendation(
                kind=Kind.COMPACTION,
                relation=ref,
                rationale=(
                    f"{files:,} files averaging {average / 1024 / 1024:.1f} MiB. Every query "
                    "pays per-file overhead — a metadata read and a task — before any row is "
                    "returned, and file skipping cannot help with a file that is mostly "
                    "overhead."
                ),
                evidence=Evidence(
                    source="layout",
                    pruning_ratio=plan.pruning_ratio if plan is not None else None,
                    pruning_unit=plan.pruning_unit if plan is not None else None,
                    relation_rows=layout.approx_rows,
                ),
                expected_effect=EffectEstimate(
                    metric="files_read",
                    before=float(files),
                    after=float(target_files),
                    method=(
                        "current bytes divided by the target file size threshold; a file-count "
                        "projection only, with no claim about latency, which this project does "
                        "not measure on a shared warehouse"
                    ),
                ),
                confidence=Confidence.ESTIMATED,
                ddl=(
                    f"OPTIMIZE {ref};\n"
                    f"-- where the write pattern is the cause, also:\n"
                    f"ALTER TABLE {ref} SET TBLPROPERTIES ("
                    f"'delta.targetFileSize' = '{self.config.small_file_bytes_threshold}');"
                ),
                risk_notes=(
                    "OPTIMIZE rewrites data and bills for the compute it uses",
                    "on a managed table predictive optimization may already compact on its "
                    "own schedule; check before scheduling a second one",
                ),
            ),
        )

    # -- D. join strategy --------------------------------------------------

    def _join_strategy(
        self, *, ref: RelationRef, layout: PhysicalLayout, plan: PlanSummary | None
    ) -> tuple[Recommendation, ...]:
        """Statistics first, hint second — the usual cause is missing statistics."""
        if plan is None:
            return ()
        if not any(warning.code is WarningCode.JOIN_ORDER_SUSPECT for warning in plan.warnings):
            return ()

        return (
            Recommendation(
                kind=Kind.BROADCAST_HINT,
                relation=ref,
                rationale=(
                    "the planner chose a sort-merge join where the smaller side may fit the "
                    "broadcast threshold. The usual cause is missing column statistics rather "
                    "than a bad planner, so collect them first and re-plan before pinning a "
                    "strategy the query will then carry forever."
                ),
                evidence=Evidence(
                    source="plan",
                    pruning_ratio=plan.pruning_ratio,
                    pruning_unit=plan.pruning_unit,
                    relation_rows=layout.approx_rows,
                ),
                expected_effect=EffectEstimate(
                    metric="bytes_read",
                    before=None,
                    after=None,
                    method=(
                        "not estimated: whether a broadcast helps depends on the build side's "
                        "size after filtering, which the plan does not report before execution"
                    ),
                ),
                confidence=Confidence.HEURISTIC,
                ddl=f"ANALYZE TABLE {ref} COMPUTE STATISTICS FOR ALL COLUMNS;",
                rewritten_sql=(
                    f"-- only if statistics do not fix the plan:\n"
                    f"SELECT /*+ BROADCAST({ref.name}) */ ..."
                ),
                risk_notes=(
                    "a broadcast hint outlives the conditions that justified it: the table it "
                    "names will keep being broadcast after it has grown past the threshold",
                ),
            ),
        )


def _covered_columns(layout: PhysicalLayout, detail: RelationDetail) -> frozenset[str] | None:
    """Columns Delta collects file statistics for, or ``None`` when unknown.

    Two ways a table says it: an explicit ``delta.dataSkippingStatsColumns``
    list, or the default of the first N columns in schema order. Ordinal
    position is not cosmetic here — it decides whether a column has statistics
    at all (SPEC §9.2.A).
    """
    if layout.stats_columns is not None:
        return frozenset(layout.stats_columns)
    indexed = layout.stats_indexed_columns
    if indexed is None:
        return None
    return frozenset(column.name for column in detail.columns[:indexed])


def _coverage_description(layout: PhysicalLayout) -> str:
    if layout.stats_columns is not None:
        return f"delta.dataSkippingStatsColumns = {', '.join(layout.stats_columns)}"
    return f"the first {layout.stats_indexed_columns} columns in schema order"


def _column_names(detail: RelationDetail) -> frozenset[str]:
    return frozenset(column.name for column in detail.columns)
