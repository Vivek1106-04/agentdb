"""Tunable constants for agentdb.

Every threshold used by the plan analyzer (SPEC §7), the advisor (SPEC §9), the
memory store (SPEC §10) and the benchmark harness (SPEC §11.4) lives here as a
named constant with a documented origin, and is overridable through an
``AGENTDB_*`` environment variable. No module outside this one may hardcode a
threshold.

Configuration is read once into a frozen :class:`Config`; nothing mutates it.
Invalid values fail fast at construction rather than surfacing as a strange
number halfway through a benchmark run.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import ClassVar, Final

ENV_PREFIX: Final = "AGENTDB_"
"""Prefix for every environment variable agentdb reads."""

_TRUE_VALUES: Final = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES: Final = frozenset({"0", "false", "no", "off"})


class ConfigError(ValueError):
    """Raised when an ``AGENTDB_*`` environment variable cannot be used."""


# ---------------------------------------------------------------------------
# defaults (SPEC Appendix A)
# ---------------------------------------------------------------------------

LOW_CARD_THRESHOLD: Final = 10_000
"""``approx_distinct`` at or below which a column is treated as low cardinality.

Chosen to match ClickHouse's own guidance for the ``LowCardinality`` type, whose
dictionary encoding stops paying off in the low tens of thousands of values.
"""

HIGH_CARD_THRESHOLD: Final = 1_000_000
"""``approx_distinct`` above which a ``GROUP BY`` earns a ``HIGH_CARD_GROUP_BY`` warning.

At a million groups the aggregate state, not the scan, dominates memory.
"""

PRUNING_RATIO_THRESHOLD: Final = 0.9
"""Granules-read ratio above which a scan counts as unpruned (SPEC §7).

At 0.9 the index removed a tenth of the data; that is noise, not pruning, and the
query behaves as a full scan.
"""

FULL_SCAN_ROW_THRESHOLD: Final = 1_000_000
"""Relation size below which an unpruned scan is not worth warning about.

Under a million rows a MergeTree full scan is typically sub-second, so a
``FULL_SCAN`` warning would be noise the agent learns to ignore.
"""

WIDE_TABLE_COLUMN_THRESHOLD: Final = 30
"""Column count above which ``SELECT *`` earns a ``SELECT_STAR_WIDE`` warning.

ClickBench's ``hits`` table has 105 columns; 30 separates ordinary fact tables
from the wide tables where column pruning is the dominant cost factor.
"""

UNBOUNDED_ROW_THRESHOLD: Final = 100_000
"""Estimated result rows above which a missing ``LIMIT`` is flagged."""

SORT_KEY_CARDINALITY_BUDGET: Final = 1e9
"""Cumulative distinct-value product cap for a proposed ``ORDER BY`` key.

Sparse primary indexes stop pruning once the leading key columns' combined
cardinality approaches the row count; the budget truncates the proposal there.
"""

SORT_KEY_PROTECT_THRESHOLD: Final = 0.10
"""Logged-workload share below which dropping a leading sort-key column goes unflagged.

Above this share the advisor must state the regression in ``risk_notes``.
"""

BLOOM_MIN_CARD_RATIO: Final = 0.01
"""``approx_distinct / rows`` above which a ``bloom_filter`` skip index is a candidate.

Below it a ``set()`` index is smaller and strictly better.
"""

SET_INDEX_MAX_DISTINCT: Final = 1_000
"""Maximum distinct values for a ``set()`` skip index candidate."""

DEFAULT_SAMPLE_FRACTION: Final = 0.01
"""Fraction of a relation read when profiling columns.

Column profiling must never full-scan a hundred-million-row table (SPEC §8.1).
"""

PROFILE_MAX_ROWS: Final = 1_000_000
"""Row ceiling for one column-profiling probe.

Separate from :data:`MAX_ROWS_TO_READ`, which bounds a query an agent asked for.
A profile is work the agent did not ask to wait for, so it is bounded far lower:
on a table with no sampling key this is the size of the prefix actually read.
"""

MAX_PROFILED_COLUMNS: Final = 30
"""Columns profiled per relation when assembling a grounded context (SPEC §13.1).

ClickBench's ``hits`` has 105 columns and each profile costs a probe, so the
budget bounds both latency and prompt size. Key columns are profiled first and
the payload states how many of the relation's columns it covered, so a partial
profile is never mistaken for a complete one.
"""

SHADOW_TABLE_MAX_ROWS: Final = 10_000_000
"""Hard row cap on an advisor shadow-validation table (SPEC §9.1.B)."""

ALLOW_SHADOW: Final = False
"""Whether shadow-table validation may run at all. Opt-in, per SPEC §13.3."""

QUERY_TIMEOUT_S: Final = 30
"""Wall-clock ceiling applied to every query agentdb issues."""

MAX_ROWS_TO_READ: Final = 500_000_000
"""Engine-side scan ceiling; ~5x the ClickBench ``hits`` row count."""

MAX_RESULT_ROWS: Final = 10_000
"""Rows returned to an agent. Bounded egress is a correctness property, not a nicety."""

MAX_INDEX_CANDIDATES: Final = 50
"""Upper bound on index candidates enumerated per advisor run.

Each candidate costs a hypopg round trip; the ranking is stable well before 50.
"""

EXEMPLAR_RECENCY_TAU_DAYS: Final = 30.0
"""Time constant of the exponential recency term in exemplar ranking (SPEC §10.4)."""

BOOTSTRAP_RESAMPLES: Final = 10_000
"""Bootstrap resamples behind every confidence interval in the report (SPEC §11.4)."""

N_SEEDS: Final = 5
"""Repetitions per (task, arm, model) at temperature > 0."""


@dataclass(frozen=True, slots=True)
class RetrievalWeights:
    """Weights of the hybrid exemplar ranking (SPEC §10.4).

    Each weight is an ablation arm in the benchmark: the report publishes what
    happens when each one is zeroed, so these defaults are a starting point to
    be measured, not a tuned result.
    """

    sem: float = 0.40
    rel: float = 0.30
    success: float = 0.15
    recency: float = 0.10
    cost: float = 0.05

    def __post_init__(self) -> None:
        for name, value in self.as_mapping().items():
            if value < 0:
                raise ConfigError(f"retrieval weight {name!r} must be >= 0, got {value}")

    def as_mapping(self) -> Mapping[str, float]:
        """Return the weights keyed by name, for scoring and for report tables."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> RetrievalWeights:
        """Build weights, overriding each from ``AGENTDB_RETRIEVAL_WEIGHT_<NAME>``."""
        source = os.environ if env is None else env
        defaults = cls()
        return cls(
            **{
                f.name: _env_float(
                    source,
                    f"RETRIEVAL_WEIGHT_{f.name.upper()}",
                    getattr(defaults, f.name),
                )
                for f in fields(cls)
            }
        )


@dataclass(frozen=True, slots=True)
class Config:
    """Effective agentdb configuration. Immutable; build one per process."""

    low_card_threshold: int = LOW_CARD_THRESHOLD
    high_card_threshold: int = HIGH_CARD_THRESHOLD
    pruning_ratio_threshold: float = PRUNING_RATIO_THRESHOLD
    full_scan_row_threshold: int = FULL_SCAN_ROW_THRESHOLD
    wide_table_column_threshold: int = WIDE_TABLE_COLUMN_THRESHOLD
    unbounded_row_threshold: int = UNBOUNDED_ROW_THRESHOLD
    sort_key_cardinality_budget: float = SORT_KEY_CARDINALITY_BUDGET
    sort_key_protect_threshold: float = SORT_KEY_PROTECT_THRESHOLD
    bloom_min_card_ratio: float = BLOOM_MIN_CARD_RATIO
    set_index_max_distinct: int = SET_INDEX_MAX_DISTINCT
    default_sample_fraction: float = DEFAULT_SAMPLE_FRACTION
    profile_max_rows: int = PROFILE_MAX_ROWS
    max_profiled_columns: int = MAX_PROFILED_COLUMNS
    shadow_table_max_rows: int = SHADOW_TABLE_MAX_ROWS
    allow_shadow: bool = ALLOW_SHADOW
    query_timeout_s: int = QUERY_TIMEOUT_S
    max_rows_to_read: int = MAX_ROWS_TO_READ
    max_result_rows: int = MAX_RESULT_ROWS
    max_index_candidates: int = MAX_INDEX_CANDIDATES
    exemplar_recency_tau_days: float = EXEMPLAR_RECENCY_TAU_DAYS
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES
    n_seeds: int = N_SEEDS
    retrieval_weights: RetrievalWeights = RetrievalWeights()

    _POSITIVE_FIELDS: ClassVar[tuple[str, ...]] = (
        "low_card_threshold",
        "high_card_threshold",
        "pruning_ratio_threshold",
        "full_scan_row_threshold",
        "wide_table_column_threshold",
        "unbounded_row_threshold",
        "sort_key_cardinality_budget",
        "bloom_min_card_ratio",
        "set_index_max_distinct",
        "profile_max_rows",
        "max_profiled_columns",
        "shadow_table_max_rows",
        "query_timeout_s",
        "max_rows_to_read",
        "max_result_rows",
        "max_index_candidates",
        "exemplar_recency_tau_days",
        "bootstrap_resamples",
        "n_seeds",
    )

    def __post_init__(self) -> None:
        for name in self._POSITIVE_FIELDS:
            value: float = getattr(self, name)
            if value <= 0:
                raise ConfigError(f"{name} must be > 0, got {value}")
        if not 0.0 < self.default_sample_fraction <= 1.0:
            raise ConfigError(
                f"default_sample_fraction must be in (0, 1], got {self.default_sample_fraction}"
            )
        if not 0.0 <= self.sort_key_protect_threshold <= 1.0:
            raise ConfigError(
                f"sort_key_protect_threshold must be in [0, 1], "
                f"got {self.sort_key_protect_threshold}"
            )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Config:
        """Build a config from ``env`` (defaults to the process environment).

        Passing the mapping explicitly keeps every caller — tests included —
        free of global state.
        """
        source = os.environ if env is None else env
        return cls(
            low_card_threshold=_env_int(source, "LOW_CARD_THRESHOLD", LOW_CARD_THRESHOLD),
            high_card_threshold=_env_int(source, "HIGH_CARD_THRESHOLD", HIGH_CARD_THRESHOLD),
            pruning_ratio_threshold=_env_float(
                source, "PRUNING_RATIO_THRESHOLD", PRUNING_RATIO_THRESHOLD
            ),
            full_scan_row_threshold=_env_int(
                source, "FULL_SCAN_ROW_THRESHOLD", FULL_SCAN_ROW_THRESHOLD
            ),
            wide_table_column_threshold=_env_int(
                source, "WIDE_TABLE_COLUMN_THRESHOLD", WIDE_TABLE_COLUMN_THRESHOLD
            ),
            unbounded_row_threshold=_env_int(
                source, "UNBOUNDED_ROW_THRESHOLD", UNBOUNDED_ROW_THRESHOLD
            ),
            sort_key_cardinality_budget=_env_float(
                source, "SORT_KEY_CARDINALITY_BUDGET", SORT_KEY_CARDINALITY_BUDGET
            ),
            sort_key_protect_threshold=_env_float(
                source, "SORT_KEY_PROTECT_THRESHOLD", SORT_KEY_PROTECT_THRESHOLD
            ),
            bloom_min_card_ratio=_env_float(source, "BLOOM_MIN_CARD_RATIO", BLOOM_MIN_CARD_RATIO),
            set_index_max_distinct=_env_int(
                source, "SET_INDEX_MAX_DISTINCT", SET_INDEX_MAX_DISTINCT
            ),
            default_sample_fraction=_env_float(
                source, "DEFAULT_SAMPLE_FRACTION", DEFAULT_SAMPLE_FRACTION
            ),
            profile_max_rows=_env_int(source, "PROFILE_MAX_ROWS", PROFILE_MAX_ROWS),
            max_profiled_columns=_env_int(source, "MAX_PROFILED_COLUMNS", MAX_PROFILED_COLUMNS),
            shadow_table_max_rows=_env_int(source, "SHADOW_TABLE_MAX_ROWS", SHADOW_TABLE_MAX_ROWS),
            allow_shadow=_env_bool(source, "ALLOW_SHADOW", ALLOW_SHADOW),
            query_timeout_s=_env_int(source, "QUERY_TIMEOUT_S", QUERY_TIMEOUT_S),
            max_rows_to_read=_env_int(source, "MAX_ROWS_TO_READ", MAX_ROWS_TO_READ),
            max_result_rows=_env_int(source, "MAX_RESULT_ROWS", MAX_RESULT_ROWS),
            max_index_candidates=_env_int(source, "MAX_INDEX_CANDIDATES", MAX_INDEX_CANDIDATES),
            exemplar_recency_tau_days=_env_float(
                source, "EXEMPLAR_RECENCY_TAU_DAYS", EXEMPLAR_RECENCY_TAU_DAYS
            ),
            bootstrap_resamples=_env_int(source, "BOOTSTRAP_RESAMPLES", BOOTSTRAP_RESAMPLES),
            n_seeds=_env_int(source, "N_SEEDS", N_SEEDS),
            retrieval_weights=RetrievalWeights.from_env(source),
        )


def _lookup(env: Mapping[str, str], name: str) -> str | None:
    """Return the raw value of ``AGENTDB_<name>``, or None when unset or blank."""
    raw = env.get(ENV_PREFIX + name)
    if raw is None:
        return None
    stripped = raw.strip()
    return stripped or None


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = _lookup(env, name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{ENV_PREFIX}{name} must be an integer, got {raw!r}") from exc


def _env_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = _lookup(env, name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{ENV_PREFIX}{name} must be a number, got {raw!r}") from exc


def _env_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = _lookup(env, name)
    if raw is None:
        return default
    lowered = raw.lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    raise ConfigError(
        f"{ENV_PREFIX}{name} must be one of {sorted(_TRUE_VALUES | _FALSE_VALUES)}, got {raw!r}"
    )
