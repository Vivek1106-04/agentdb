"""The ClickHouse statement text and its parsing helpers (SPEC §8.1).

The assertions here are on strings on purpose. The ``EXPLAIN`` settings and the
sampling clause are the specification; if either drifts, the pruning evidence the
plan layer reports becomes quietly wrong rather than loudly broken.
"""

from __future__ import annotations

import pytest

from agentdb.adapters.clickhouse_sql import (
    EXPLAIN_SETTINGS,
    IdentifierError,
    explain_statement,
    is_nullable,
    profile_statement,
    qualified,
    quote_identifier,
    relation_kind,
    split_key_expression,
)
from agentdb.adapters.models import ExplainMode

# --------------------------------------------------------------------------
# identifiers
# --------------------------------------------------------------------------


def test_quote_identifier_backticks_a_bare_name() -> None:
    assert quote_identifier("hits") == "`hits`"


@pytest.mark.parametrize("name", ["", "1hits", "hits; DROP", "hits`", "a b"])
def test_quote_identifier_refuses_anything_that_is_not_an_identifier(name: str) -> None:
    with pytest.raises(IdentifierError):
        quote_identifier(name)


def test_qualified_validates_both_halves() -> None:
    assert qualified("agentdb", "hits") == "`agentdb`.`hits`"
    with pytest.raises(IdentifierError):
        qualified("agentdb", "not valid")


# --------------------------------------------------------------------------
# explain
# --------------------------------------------------------------------------


def test_estimate_plan_disables_the_two_settings_that_hide_index_evidence() -> None:
    statement = explain_statement("SELECT 1", ExplainMode.ESTIMATE)

    assert "indexes = 1, projections = 1" in statement
    assert EXPLAIN_SETTINGS in statement
    assert "use_query_condition_cache = 0" in EXPLAIN_SETTINGS
    assert "use_skip_indexes_on_data_read = 0" in EXPLAIN_SETTINGS


def test_the_settings_clause_trails_the_explained_query() -> None:
    """ClickHouse parses EXPLAIN <options> <query> SETTINGS <settings>.

    Putting the SETTINGS block between the options and the query is a syntax
    error, not a style choice — and an easy one to make, since EXPLAIN's own
    options are written in the same place without the keyword.
    """
    statement = explain_statement("SELECT 1", ExplainMode.ESTIMATE)

    assert statement.index("SELECT 1") < statement.index("SETTINGS")
    assert statement.endswith(EXPLAIN_SETTINGS)


def test_pipeline_and_syntax_modes_use_their_own_statements() -> None:
    assert explain_statement("SELECT 1", ExplainMode.PIPELINE) == "EXPLAIN PIPELINE SELECT 1"
    assert explain_statement("SELECT 1", ExplainMode.SYNTAX) == "EXPLAIN SYNTAX SELECT 1"


# --------------------------------------------------------------------------
# profiling probe
# --------------------------------------------------------------------------


def test_profile_statement_samples_when_the_table_has_a_sampling_key() -> None:
    statement = profile_statement(
        source="`agentdb`.`hits`",
        column="UserID",
        top_k=10,
        sample_fraction=0.01,
        max_rows=100_000,
    )

    assert "FROM `agentdb`.`hits` SAMPLE 0.01" in statement
    assert "uniqCombined64(`UserID`)" in statement
    assert "approx_top_k(10)(`UserID`)" in statement


def test_the_top_k_result_is_stripped_of_its_field_names() -> None:
    """``approx_top_k`` returns a named tuple, which the driver hands back as a dict.

    The parser stores ``(value, count)`` pairs, so the statement projects the
    names away rather than leaving the returned shape to depend on which
    ClickHouse driver happens to be installed.
    """
    statement = profile_statement(
        source="`agentdb`.`hits`",
        column="UserID",
        top_k=10,
        sample_fraction=0.01,
        max_rows=100_000,
    )

    assert "arrayMap(entry -> (entry.1, entry.2), approx_top_k(10)(`UserID`))" in statement


def test_profile_statement_reads_a_bounded_prefix_without_a_sampling_key() -> None:
    statement = profile_statement(
        source="`agentdb`.`hits`",
        column="UserID",
        top_k=10,
        sample_fraction=None,
        max_rows=50_000,
    )

    assert "SELECT `UserID` FROM `agentdb`.`hits` LIMIT 50000" in statement
    assert "SAMPLE" not in statement


def test_profile_statement_refuses_a_column_name_that_is_not_an_identifier() -> None:
    with pytest.raises(IdentifierError):
        profile_statement(
            source="`agentdb`.`hits`",
            column="UserID) --",
            top_k=10,
            sample_fraction=None,
            max_rows=10,
        )


# --------------------------------------------------------------------------
# key parsing
# --------------------------------------------------------------------------


def test_split_key_expression_keeps_function_calls_whole() -> None:
    assert split_key_expression("toDate(EventTime), CounterID") == (
        "toDate(EventTime)",
        "CounterID",
    )


def test_split_key_expression_handles_nested_calls_and_arrays() -> None:
    assert split_key_expression("cityHash64(concat(a, b)), arrayElement([x, y], 1), c") == (
        "cityHash64(concat(a, b))",
        "arrayElement([x, y], 1)",
        "c",
    )


def test_split_key_expression_reports_absence_as_none_not_as_an_empty_key() -> None:
    assert split_key_expression("   ") is None


def test_split_key_expression_drops_a_trailing_comma_rather_than_yielding_a_blank_term() -> None:
    assert split_key_expression("CounterID,") == ("CounterID",)


# --------------------------------------------------------------------------
# type and engine mapping
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("data_type", "expected"),
    [
        ("String", False),
        ("Nullable(String)", True),
        ("LowCardinality(Nullable(String))", True),
        ("LowCardinality(String)", False),
    ],
)
def test_is_nullable_sees_through_the_low_cardinality_wrapper(
    data_type: str, expected: bool
) -> None:
    assert is_nullable(data_type) is expected


@pytest.mark.parametrize(
    ("engine", "kind"),
    [
        ("MergeTree", "table"),
        ("ReplacingMergeTree", "table"),
        ("View", "view"),
        ("MaterializedView", "materialized_view"),
    ],
)
def test_relation_kind_maps_the_engine_column(engine: str, kind: str) -> None:
    assert relation_kind(engine) == kind
