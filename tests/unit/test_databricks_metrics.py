"""Measured query metrics, pinned to payloads observed on a live warehouse.

Every payload below was captured from the Databricks Query History API on a Free
Edition workspace running DBSQL 2026.20, against ``samples.tpch``. That matters
because the interesting cases here are not parse failures — they are three
different situations in which a successful query reports **zero files read**:

* the result cache answered it;
* Delta metadata answered it, so no file was ever opened;
* the warehouse reported no metrics section at all.

Each one, taken at face value, becomes ``0 files read of 0 considered`` and then
"pruned everything, perfectly". These tests exist to keep that from happening
again — it is the same defect the first live run found in the Photon plan parser,
arriving through a different door.
"""

from __future__ import annotations

from agentdb.adapters import databricks_metrics as metrics
from agentdb.adapters.models import QueryMetrics

# Observed: an aggregate with a pushdown-friendly range over samples.tpch.lineitem.
# Ten files read, none pruned (the table has no clustering key), but only 12% of
# the bytes in those files fetched — column projection, not file pruning.
MEASURED_ENTRY = {
    "query_id": "01f198bb-b784-1d68-a885-2c971cdffb43",
    "status": "FINISHED",
    "is_final": True,
    "plans_state": "EXISTS",
    "metrics": {
        "compilation_time_ms": 656,
        "execution_time_ms": 1139,
        "photon_total_time_ms": 2262,
        "pruned_bytes": 0,
        "pruned_files_count": 0,
        "read_bytes": 90720611,
        "read_cache_bytes": 52861021,
        "read_files_bytes": 753837113,
        "read_files_count": 10,
        "read_partitions_count": 0,
        "result_from_cache": False,
        "rows_produced_count": 3,
        "rows_read_count": 4545830,
        "spill_to_disk_bytes": 0,
        "total_time_ms": 1842,
    },
}

# Observed: the identical statement, re-submitted. Every counter is zero and the
# query still "succeeded" in 672ms.
CACHED_ENTRY = {
    "query_id": "01f198bb-88c2-1e83-868c-785230752676",
    "is_final": True,
    "plans_state": "EMPTY",
    "metrics": {
        "pruned_files_count": 0,
        "read_bytes": 0,
        "read_files_count": 0,
        "read_partitions_count": 0,
        "result_from_cache": True,
        "rows_produced_count": 3,
        "rows_read_count": 0,
        "total_time_ms": 1001,
    },
}

# Observed: SELECT count(*) FROM samples.tpch.region. Not cached, and still zero
# files — Delta answers a bare count from the transaction log without opening one.
METADATA_ONLY_ENTRY = {
    "query_id": "01f198bb-8bd5-1d2b-a088-83557d64b918",
    "is_final": True,
    "plans_state": "EXISTS",
    "metrics": {
        "pruned_files_count": 0,
        "read_bytes": 0,
        "read_files_count": 0,
        "result_from_cache": False,
        "rows_produced_count": 1,
        "rows_read_count": 0,
        "total_time_ms": 1325,
    },
}


def test_a_measured_entry_carries_the_counts_the_plan_could_not() -> None:
    result = metrics.from_query_info(MEASURED_ENTRY)

    assert result is not None
    assert result.engine == "databricks"
    assert result.source == "query_history_api"
    assert result.statement_id == "01f198bb-b784-1d68-a885-2c971cdffb43"
    assert result.files_read == 10
    assert result.files_pruned == 0
    assert result.rows_read == 4_545_830
    assert result.photon_time_ms == 2262.0


def test_ten_files_read_and_none_pruned_is_a_ratio_of_one() -> None:
    result = metrics.from_query_info(MEASURED_ENTRY)

    assert result is not None
    assert result.measured is True
    assert result.files_considered == 10
    assert result.pruning_ratio == 1.0  # no clustering key: nothing to prune, truthfully


def test_the_byte_ratio_reports_what_file_pruning_could_not() -> None:
    result = metrics.from_query_info(MEASURED_ENTRY)

    assert result is not None
    ratio = result.bytes_ratio
    assert ratio is not None
    # 90,720,611 of 753,837,113: projection and row-group skipping threw away 88%
    # of the bytes in files that file pruning kept every one of.
    assert 0.11 < ratio < 0.13


def test_a_cache_hit_is_not_a_measurement() -> None:
    result = metrics.from_query_info(CACHED_ENTRY)

    assert result is not None
    assert result.from_result_cache is True
    assert result.measured is False
    assert result.pruning_ratio is None  # never 0.0, which would read as perfect pruning
    assert result.files_considered is None
    assert result.bytes_ratio is None


def test_a_metadata_only_answer_is_not_a_measurement_either() -> None:
    result = metrics.from_query_info(METADATA_ONLY_ENTRY)

    assert result is not None
    assert result.from_result_cache is False
    assert result.measured is False  # count(*) opened no file; it pruned nothing
    assert result.pruning_ratio is None


def test_an_entry_without_metrics_is_unmeasured_rather_than_zeroed() -> None:
    assert metrics.from_query_info({"query_id": "abc", "is_final": True}) is None


def test_an_absent_entry_is_none() -> None:
    assert metrics.from_query_info(None) is None
    assert metrics.from_query_info({}) is None


def test_an_entry_without_an_id_cannot_be_attributed_so_it_is_dropped() -> None:
    assert metrics.from_query_info({"metrics": {"read_files_count": 4}}) is None


def test_the_system_table_spelling_of_the_id_is_accepted() -> None:
    result = metrics.from_query_info({"statement_id": "sid-1", "metrics": {"read_files_count": 2}})

    assert result is not None
    assert result.statement_id == "sid-1"


def test_string_counters_are_read_and_unreadable_ones_stay_unknown() -> None:
    result = metrics.from_query_info(
        {
            "query_id": "sid-2",
            "metrics": {
                "read_files_count": "12",
                "pruned_files_count": "not a number",
                "photon_total_time_ms": "4.5",
                "execution_time_ms": "nope",
                "result_from_cache": "false",
                "pruned_bytes": True,
            },
        }
    )

    assert result is not None
    assert result.files_read == 12
    assert result.files_pruned is None
    assert result.photon_time_ms == 4.5
    assert result.execution_time_ms is None
    assert result.from_result_cache is False
    assert result.bytes_pruned is None  # a bool is not a count


def test_an_unreadable_cache_flag_is_unknown() -> None:
    result = metrics.from_query_info(
        {"query_id": "sid-3", "metrics": {"read_files_count": 1, "result_from_cache": "maybe"}}
    )

    assert result is not None
    assert result.from_result_cache is None
    assert result.measured is True  # a file was read; the flag's absence does not undo that


def test_a_true_cache_flag_spelled_as_a_string_is_still_a_cache_hit() -> None:
    result = metrics.from_query_info(
        {"query_id": "sid-4", "metrics": {"read_files_count": "1", "result_from_cache": "true"}}
    )

    assert result is not None
    assert result.measured is False


def test_a_negative_counter_is_read_rather_than_discarded() -> None:
    result = metrics.from_query_info(
        {"query_id": "sid-5", "metrics": {"read_files_count": 1, "spill_to_disk_bytes": "-1"}}
    )

    assert result is not None
    assert result.spill_bytes == -1


def test_a_running_statement_is_not_final() -> None:
    assert metrics.is_final({"query_id": "sid", "is_final": False}) is False
    assert metrics.is_final({"query_id": "sid", "is_final": True}) is True
    assert metrics.is_final(None) is False
    assert metrics.is_final({}) is False


def test_an_entry_that_stopped_reporting_finality_is_treated_as_final() -> None:
    # An API that drops the flag must not turn every lookup into an endless poll.
    assert metrics.is_final({"query_id": "sid"}) is True


def test_a_pruned_file_count_alone_still_counts_as_measured() -> None:
    # A scan that pruned every file reads none of them; that is the best possible
    # outcome, and reporting it as unmeasured would hide the one result the whole
    # pruning story is about.
    result = QueryMetrics(
        statement_id="sid",
        engine="databricks",
        source="query_history_api",
        from_result_cache=False,
        files_read=0,
        files_pruned=40,
    )

    assert result.measured is True
    assert result.files_considered == 40
    assert result.pruning_ratio == 0.0


def test_metrics_render_for_an_agent_says_which_mechanism_did_what() -> None:
    result = metrics.from_query_info(MEASURED_ENTRY)

    assert result is not None
    rendered = result.render()
    assert "files read after pruning: 100.0% (10 of 10 considered)" in rendered
    assert "column projection and in-file skipping, not file pruning" in rendered
    assert "rows read: 4,545,830" in rendered
    assert "Photon time: 2,262 ms" in rendered


def test_a_cache_hit_renders_as_a_cache_hit_and_claims_nothing_else() -> None:
    result = metrics.from_query_info(CACHED_ENTRY)

    assert result is not None
    assert result.render() == "Measured: the result cache answered; no data was read."


def test_an_unmeasured_statement_says_so() -> None:
    result = metrics.from_query_info(METADATA_ONLY_ENTRY)

    assert result is not None
    assert "reported no file access" in result.render()


def test_a_measured_statement_without_byte_totals_renders_the_counts_it_has() -> None:
    result = QueryMetrics(
        statement_id="sid",
        engine="databricks",
        source="query_history_api",
        files_read=2,
        files_pruned=8,
    )

    rendered = result.render()
    assert "files read after pruning: 20.0% (2 of 10 considered)" in rendered
    assert "bytes fetched" not in rendered
    assert "rows read" not in rendered
    assert "Photon" not in rendered


def test_a_missing_file_count_leaves_every_derived_number_unknown() -> None:
    result = QueryMetrics(
        statement_id="sid",
        engine="databricks",
        source="query_history_api",
        files_read=None,
        files_pruned=3,
        bytes_read=10,
        bytes_in_files_read=100,
    )

    assert result.files_considered is None
    assert result.pruning_ratio is None
    assert result.bytes_ratio == 0.1  # measured via files_pruned; the byte pair is complete
    assert "files read after pruning" not in result.render()
