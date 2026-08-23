"""The memory tools' argument handling and error paths (SPEC §13.1).

The happy paths are covered by the contract tests, which validate real responses
against the declared schemas. What is here is everything an agent can get wrong,
because each of these has to come back as a result it can act on rather than as
a traceback across the protocol boundary — plus the one behaviour no schema can
state: that what gets remembered about a query is parsed from the query.
"""

from __future__ import annotations

from agentdb.adapters import QuerySemanticError
from agentdb.server import ToolCatalog, build_catalog
from agentdb.server.base import ServerContext
from agentdb.server.tools.memory import memory_tools
from tests.fakes import clickhouse_hits_fixture
from tests.memory_fakes import FakeConnection
from tests.server_fakes import RESULT, clickhouse_catalog, memory_connection, memory_store


def catalog_over(connection: FakeConnection) -> ToolCatalog:
    adapter = clickhouse_hits_fixture()
    adapter.result = RESULT
    return build_catalog(adapter, store=memory_store(connection))


async def test_a_recorded_query_comes_back_with_both_time_axes_open() -> None:
    catalog, _ = clickhouse_catalog()

    response = await catalog.call(
        "record_outcome",
        {
            "namespace": "agentdb",
            "question": "how many hits per counter?",
            "sql": "SELECT CounterID, count() FROM hits GROUP BY CounterID",
            "outcome": "success",
            "provenance": "curated",
        },
    )

    assert not response.is_error
    assert response.structured["valid_to"] is None
    assert response.structured["tx_to"] is None


async def test_the_names_re_validation_will_check_are_parsed_from_the_sql() -> None:
    """An agent asked to list them would list them wrong exactly when it mattered."""
    connection = memory_connection()

    await catalog_over(connection).call(
        "record_outcome",
        {
            "namespace": "agentdb",
            "question": "hits by url since July",
            "sql": "SELECT URL, count() FROM hits WHERE EventDate > '2013-07-01' GROUP BY URL",
            "outcome": "success",
        },
    )

    _, params = connection.executed("INSERT INTO agentdb_exemplar")[0]
    assert params is not None
    assert params[5] == ["hits"]
    assert params[6] == ["EventDate", "URL"]


async def test_a_projected_column_is_not_something_a_schema_change_should_break() -> None:
    """Only filter and grouping columns are recorded; a widened SELECT is not a break."""
    connection = memory_connection()

    await catalog_over(connection).call(
        "record_outcome",
        {
            "namespace": "agentdb",
            "question": "everything about counter 62",
            "sql": "SELECT URL, UserID FROM hits WHERE CounterID = 62",
            "outcome": "success",
        },
    )

    _, params = connection.executed("INSERT INTO agentdb_exemplar")[0]
    assert params is not None
    assert params[6] == ["CounterID"]


async def test_sql_naming_no_relation_is_refused_rather_than_remembered() -> None:
    catalog, _ = clickhouse_catalog()

    response = await catalog.call(
        "record_outcome",
        {
            "namespace": "agentdb",
            "question": "what time is it?",
            "sql": "SELECT now()",
            "outcome": "success",
        },
    )

    assert response.is_error
    assert "names no relation" in str(response.structured["error"])


async def test_an_unknown_outcome_lists_the_ones_that_exist() -> None:
    catalog, _ = clickhouse_catalog()

    response = await catalog.call(
        "record_outcome",
        {
            "namespace": "agentdb",
            "question": "q",
            "sql": "SELECT count() FROM hits",
            "outcome": "worked_i_think",
        },
    )

    assert response.is_error
    assert "success" in str(response.structured["suggestion"])


async def test_an_unknown_provenance_lists_the_ones_that_exist() -> None:
    catalog, _ = clickhouse_catalog()

    response = await catalog.call(
        "record_outcome",
        {
            "namespace": "agentdb",
            "question": "q",
            "sql": "SELECT count() FROM hits",
            "outcome": "success",
            "provenance": "vibes",
        },
    )

    assert response.is_error
    assert "workload_mined" in str(response.structured["suggestion"])


async def test_a_success_carrying_an_error_class_is_refused() -> None:
    catalog, _ = clickhouse_catalog()

    response = await catalog.call(
        "record_outcome",
        {
            "namespace": "agentdb",
            "question": "q",
            "sql": "SELECT count() FROM hits",
            "outcome": "success",
            "error_class": "semantic",
        },
    )

    assert response.is_error


async def test_a_failure_without_an_error_class_is_refused() -> None:
    """A negative exemplar with no error class teaches an agent nothing."""
    catalog, _ = clickhouse_catalog()

    response = await catalog.call(
        "record_outcome",
        {
            "namespace": "agentdb",
            "question": "q",
            "sql": "SELECT count() FROM hits",
            "outcome": "error",
        },
    )

    assert response.is_error
    assert "error_class" in str(response.structured["error"])


async def test_relations_must_be_strings_when_given_at_all() -> None:
    catalog, _ = clickhouse_catalog()

    response = await catalog.call(
        "retrieve_exemplars",
        {"question": "q", "namespace": "agentdb", "relations": [7]},
    )

    assert response.is_error
    assert "array of strings" in str(response.structured["error"])


async def test_positives_and_negatives_are_asked_for_separately() -> None:
    connection = memory_connection()

    response = await catalog_over(connection).call(
        "retrieve_exemplars", {"question": "q", "namespace": "agentdb"}
    )

    queries = [query for query, _ in connection.executed("ORDER BY (relations &&")]
    assert len(queries) == 2
    assert any("outcome = 'success'" in query for query in queries)
    assert any("outcome <> 'success'" in query for query in queries)
    assert not response.is_error


async def test_history_looks_up_the_parameterized_form_of_the_query() -> None:
    catalog, _ = clickhouse_catalog()

    response = await catalog.call(
        "explain_exemplar_history",
        {
            "namespace": "agentdb",
            "sql": "SELECT count() FROM hits WHERE EventDate > '2013-07-01'",
        },
    )

    assert not response.is_error
    assert response.structured["normalized_sql"] == "SELECT count() FROM hits WHERE EventDate > ?"


# --------------------------------------------------------------------------
# what run_query remembers on its own (SPEC §13.1)
# --------------------------------------------------------------------------


async def test_a_successful_execution_is_remembered_with_its_measured_cost() -> None:
    connection = memory_connection()

    response = await catalog_over(connection).call(
        "run_query",
        {
            "sql": "SELECT CounterID, count() FROM hits GROUP BY CounterID",
            "question": "how many hits per counter?",
            "namespace": "agentdb",
        },
    )

    assert not response.is_error
    _, params = connection.executed("INSERT INTO agentdb_exemplar")[0]
    assert params is not None
    assert params[8] == "success"
    assert params[13] == RESULT.bytes_read


async def test_an_execution_with_no_question_runs_and_is_not_remembered() -> None:
    """An exemplar with no question can never be retrieved by meaning."""
    connection = memory_connection()

    response = await catalog_over(connection).call(
        "run_query", {"sql": "SELECT count() FROM hits", "namespace": "agentdb"}
    )

    assert not response.is_error
    assert connection.executed("INSERT INTO agentdb_exemplar") == []


async def test_an_execution_naming_no_parseable_relation_is_not_remembered() -> None:
    connection = memory_connection()

    await catalog_over(connection).call(
        "run_query",
        {"sql": "SELECT now()", "question": "what time is it?", "namespace": "agentdb"},
    )

    assert connection.executed("INSERT INTO agentdb_exemplar") == []


async def test_a_failed_execution_is_remembered_as_a_negative_exemplar_and_still_reported() -> None:
    connection = memory_connection()
    adapter = clickhouse_hits_fixture()
    adapter.execute_error = QuerySemanticError("Code: 47. UNKNOWN_IDENTIFIER")
    catalog = build_catalog(adapter, store=memory_store(connection))

    response = await catalog.call(
        "run_query",
        {
            "sql": "SELECT UserId FROM hits",
            "question": "who visited?",
            "namespace": "agentdb",
        },
    )

    assert response.is_error
    _, params = connection.executed("INSERT INTO agentdb_exemplar")[0]
    assert params is not None
    assert params[8] == "error"
    assert params[15] == "semantic"


async def test_an_execution_is_not_remembered_where_there_is_no_store() -> None:
    adapter = clickhouse_hits_fixture()
    adapter.result = RESULT
    catalog = build_catalog(adapter)

    response = await catalog.call(
        "run_query",
        {
            "sql": "SELECT count() FROM hits",
            "question": "how many?",
            "namespace": "agentdb",
        },
    )

    assert not response.is_error


def test_no_store_means_no_tools() -> None:
    assert memory_tools(ServerContext(adapter=clickhouse_hits_fixture())) == ()
