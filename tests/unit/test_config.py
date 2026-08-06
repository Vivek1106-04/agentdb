"""Config is the only place thresholds live, so it is the only place they can break."""

from __future__ import annotations

import pytest

from agentdb import config as cfg
from agentdb.config import Config, ConfigError, RetrievalWeights


def test_defaults_match_the_documented_constants() -> None:
    # Arrange / Act
    conf = Config.from_env({})

    # Assert
    assert conf.low_card_threshold == cfg.LOW_CARD_THRESHOLD
    assert conf.high_card_threshold == cfg.HIGH_CARD_THRESHOLD
    assert conf.full_scan_row_threshold == cfg.FULL_SCAN_ROW_THRESHOLD
    assert conf.wide_table_column_threshold == cfg.WIDE_TABLE_COLUMN_THRESHOLD
    assert conf.unbounded_row_threshold == cfg.UNBOUNDED_ROW_THRESHOLD
    assert conf.sort_key_cardinality_budget == cfg.SORT_KEY_CARDINALITY_BUDGET
    assert conf.sort_key_protect_threshold == cfg.SORT_KEY_PROTECT_THRESHOLD
    assert conf.bloom_min_card_ratio == cfg.BLOOM_MIN_CARD_RATIO
    assert conf.set_index_max_distinct == cfg.SET_INDEX_MAX_DISTINCT
    assert conf.default_sample_fraction == cfg.DEFAULT_SAMPLE_FRACTION
    assert conf.shadow_table_max_rows == cfg.SHADOW_TABLE_MAX_ROWS
    assert conf.allow_shadow is cfg.ALLOW_SHADOW
    assert conf.query_timeout_s == cfg.QUERY_TIMEOUT_S
    assert conf.max_rows_to_read == cfg.MAX_ROWS_TO_READ
    assert conf.max_result_rows == cfg.MAX_RESULT_ROWS
    assert conf.max_index_candidates == cfg.MAX_INDEX_CANDIDATES
    assert conf.exemplar_recency_tau_days == cfg.EXEMPLAR_RECENCY_TAU_DAYS
    assert conf.bootstrap_resamples == cfg.BOOTSTRAP_RESAMPLES
    assert conf.n_seeds == cfg.N_SEEDS
    assert conf.retrieval_weights == RetrievalWeights()


def test_reads_the_process_environment_when_none_is_passed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setenv("AGENTDB_N_SEEDS", "9")

    # Act
    conf = Config.from_env()

    # Assert
    assert conf.n_seeds == 9


def test_every_numeric_field_is_overridable_by_environment_variable() -> None:
    # Arrange — one non-default value per env-backed field
    env = {
        "AGENTDB_LOW_CARD_THRESHOLD": "11",
        "AGENTDB_HIGH_CARD_THRESHOLD": "12",
        "AGENTDB_FULL_SCAN_ROW_THRESHOLD": "13",
        "AGENTDB_WIDE_TABLE_COLUMN_THRESHOLD": "14",
        "AGENTDB_UNBOUNDED_ROW_THRESHOLD": "15",
        "AGENTDB_SORT_KEY_CARDINALITY_BUDGET": "16.5",
        "AGENTDB_SORT_KEY_PROTECT_THRESHOLD": "0.5",
        "AGENTDB_BLOOM_MIN_CARD_RATIO": "0.25",
        "AGENTDB_SET_INDEX_MAX_DISTINCT": "17",
        "AGENTDB_DEFAULT_SAMPLE_FRACTION": "0.5",
        "AGENTDB_SHADOW_TABLE_MAX_ROWS": "18",
        "AGENTDB_ALLOW_SHADOW": "true",
        "AGENTDB_QUERY_TIMEOUT_S": "19",
        "AGENTDB_MAX_ROWS_TO_READ": "20",
        "AGENTDB_MAX_RESULT_ROWS": "21",
        "AGENTDB_MAX_INDEX_CANDIDATES": "22",
        "AGENTDB_EXEMPLAR_RECENCY_TAU_DAYS": "23.5",
        "AGENTDB_BOOTSTRAP_RESAMPLES": "24",
        "AGENTDB_N_SEEDS": "25",
        "AGENTDB_RETRIEVAL_WEIGHT_SEM": "0.6",
    }

    # Act
    conf = Config.from_env(env)

    # Assert
    assert conf.low_card_threshold == 11
    assert conf.high_card_threshold == 12
    assert conf.full_scan_row_threshold == 13
    assert conf.wide_table_column_threshold == 14
    assert conf.unbounded_row_threshold == 15
    assert conf.sort_key_cardinality_budget == 16.5
    assert conf.sort_key_protect_threshold == 0.5
    assert conf.bloom_min_card_ratio == 0.25
    assert conf.set_index_max_distinct == 17
    assert conf.default_sample_fraction == 0.5
    assert conf.shadow_table_max_rows == 18
    assert conf.allow_shadow is True
    assert conf.query_timeout_s == 19
    assert conf.max_rows_to_read == 20
    assert conf.max_result_rows == 21
    assert conf.max_index_candidates == 22
    assert conf.exemplar_recency_tau_days == 23.5
    assert conf.bootstrap_resamples == 24
    assert conf.n_seeds == 25
    assert conf.retrieval_weights.sem == 0.6
    assert conf.retrieval_weights.rel == RetrievalWeights().rel


def test_config_is_immutable() -> None:
    # Arrange
    conf = Config.from_env({})

    # Act / Assert
    with pytest.raises(AttributeError):
        conf.n_seeds = 7  # type: ignore[misc]


@pytest.mark.parametrize("value", ["", "   "])
def test_blank_environment_values_fall_back_to_the_default(value: str) -> None:
    # Arrange / Act
    conf = Config.from_env({"AGENTDB_N_SEEDS": value})

    # Assert
    assert conf.n_seeds == cfg.N_SEEDS


def test_non_integer_value_fails_fast() -> None:
    with pytest.raises(ConfigError, match="AGENTDB_N_SEEDS must be an integer"):
        Config.from_env({"AGENTDB_N_SEEDS": "five"})


def test_non_numeric_float_value_fails_fast() -> None:
    with pytest.raises(ConfigError, match="AGENTDB_BLOOM_MIN_CARD_RATIO must be a number"):
        Config.from_env({"AGENTDB_BLOOM_MIN_CARD_RATIO": "small"})


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1", True),
        ("TRUE", True),
        ("yes", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("off", False),
    ],
)
def test_boolean_values_accept_the_usual_spellings(raw: str, expected: bool) -> None:
    assert Config.from_env({"AGENTDB_ALLOW_SHADOW": raw}).allow_shadow is expected


def test_unparseable_boolean_fails_fast() -> None:
    with pytest.raises(ConfigError, match="AGENTDB_ALLOW_SHADOW must be one of"):
        Config.from_env({"AGENTDB_ALLOW_SHADOW": "maybe"})


@pytest.mark.parametrize("field_name", Config._POSITIVE_FIELDS)
def test_every_positive_field_rejects_zero(field_name: str) -> None:
    with pytest.raises(ConfigError, match=f"{field_name} must be > 0"):
        Config(**{field_name: 0})  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0.0, 1.5, -0.1])
def test_sample_fraction_must_be_a_usable_fraction(value: float) -> None:
    with pytest.raises(ConfigError, match="default_sample_fraction must be in"):
        Config(default_sample_fraction=value)


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_sort_key_protect_threshold_must_be_a_share(value: float) -> None:
    with pytest.raises(ConfigError, match="sort_key_protect_threshold must be in"):
        Config(sort_key_protect_threshold=value)


def test_retrieval_weights_expose_a_name_keyed_mapping() -> None:
    # Arrange / Act
    weights = RetrievalWeights().as_mapping()

    # Assert
    assert weights == {"sem": 0.40, "rel": 0.30, "success": 0.15, "recency": 0.10, "cost": 0.05}


def test_retrieval_weights_reject_negative_values() -> None:
    with pytest.raises(ConfigError, match="retrieval weight 'cost' must be >= 0"):
        RetrievalWeights(cost=-0.1)


def test_retrieval_weights_may_be_zeroed_for_an_ablation_arm() -> None:
    # Arrange / Act — zeroing a weight is a benchmark arm, not an error
    weights = RetrievalWeights.from_env({"AGENTDB_RETRIEVAL_WEIGHT_RECENCY": "0"})

    # Assert
    assert weights.recency == 0.0
    assert weights.sem == RetrievalWeights().sem


def test_retrieval_weights_read_the_process_environment_when_none_is_passed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setenv("AGENTDB_RETRIEVAL_WEIGHT_COST", "0.9")

    # Act / Assert
    assert RetrievalWeights.from_env().cost == 0.9
