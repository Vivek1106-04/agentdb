"""Extraction is part of the measurement, so its rules are pinned by tests."""

from __future__ import annotations

import pytest

from agenteval.models.extract import extract_sql


def test_extracts_a_fenced_sql_block() -> None:
    text = "Here you go:\n\n```sql\nSELECT count() FROM hits\n```\n\nHope that helps."

    assert extract_sql(text) == "SELECT count() FROM hits"


def test_fence_language_tag_is_case_insensitive() -> None:
    assert extract_sql("```SQL\nSELECT 1\n```") == "SELECT 1"


def test_the_last_sql_block_wins() -> None:
    # Arrange — a model that reconsiders meant the query it finished with
    text = "```sql\nSELECT 1\n```\nOn reflection:\n```sql\nSELECT 2\n```"

    assert extract_sql(text) == "SELECT 2"


def test_falls_back_to_an_untagged_fence() -> None:
    assert extract_sql("```\nSELECT 1\n```") == "SELECT 1"


def test_a_tagged_sql_fence_beats_an_earlier_untagged_one() -> None:
    text = "```\nnot sql\n```\n```sql\nSELECT 1\n```"

    assert extract_sql(text) == "SELECT 1"


@pytest.mark.parametrize("keyword", ["SELECT", "select", "WITH", "with"])
def test_extracts_an_unfenced_query(keyword: str) -> None:
    assert extract_sql(f"Sure.\n{keyword} 1") == f"{keyword} 1"


def test_prose_before_an_unfenced_query_is_dropped() -> None:
    text = "The table is large, so:\nSELECT count()\nFROM hits"

    assert extract_sql(text) == "SELECT count()\nFROM hits"


def test_trailing_semicolons_are_trimmed() -> None:
    assert extract_sql("```sql\nSELECT 1;\n```") == "SELECT 1"
    assert extract_sql("```sql\nSELECT 1; ;\n```") == "SELECT 1"


def test_a_reply_with_no_query_extracts_nothing() -> None:
    assert extract_sql("I cannot answer that from this schema.") is None


def test_an_empty_fence_extracts_nothing() -> None:
    assert extract_sql("```sql\n\n```") is None
