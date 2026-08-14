"""Capability flags and the error taxonomy are load-bearing, so they get tests."""

from __future__ import annotations

import pytest

from agentdb.adapters import (
    Adapter,
    AdapterError,
    BaseAdapter,
    Capability,
    ErrorClass,
    UnsupportedCapabilityError,
)
from agentdb.adapters.base import (
    EngineConnectionError,
    LimitExceededError,
    PlanRejectionError,
    QueryPermissionError,
    QuerySemanticError,
    QuerySyntaxError,
    QueryTimeoutError,
)
from tests.fakes import FakeAdapter


class _StubAdapter(BaseAdapter):
    """Minimal adapter used to exercise the shared capability bookkeeping."""

    engine = "clickhouse"
    capabilities = frozenset({Capability.SORT_KEY, Capability.SKIP_INDEX})


def test_supports_reports_declared_capabilities() -> None:
    adapter = _StubAdapter()
    assert adapter.supports(Capability.SORT_KEY) is True
    assert adapter.supports(Capability.CLUSTERING_KEY) is False


def test_require_passes_for_a_declared_capability() -> None:
    _StubAdapter().require(Capability.SKIP_INDEX)


def test_require_refuses_rather_than_faking_an_unsupported_capability() -> None:
    # Arrange
    adapter = _StubAdapter()

    # Act
    with pytest.raises(UnsupportedCapabilityError) as caught:
        adapter.require(Capability.CLUSTERING_KEY)

    # Assert — the failure names the capability so a caller can branch on it
    assert caught.value.capability is Capability.CLUSTERING_KEY
    assert "clickhouse adapter does not support clustering_key" in str(caught.value)


def test_base_adapter_declares_no_capabilities_by_default() -> None:
    assert BaseAdapter.capabilities == frozenset()


def test_a_complete_implementation_satisfies_the_adapter_protocol() -> None:
    assert isinstance(FakeAdapter(), Adapter) is True


def test_a_partial_implementation_does_not_satisfy_the_protocol() -> None:
    # A class with the capability bookkeeping but none of the engine methods is
    # not an Adapter — the protocol is the contract, inheritance is not.
    assert isinstance(_StubAdapter(), Adapter) is False


@pytest.mark.parametrize(
    ("error_type", "expected"),
    [
        (QuerySyntaxError, ErrorClass.SYNTAX),
        (QuerySemanticError, ErrorClass.SEMANTIC),
        (PlanRejectionError, ErrorClass.PLAN_REJECTION),
        (QueryTimeoutError, ErrorClass.TIMEOUT),
        (QueryPermissionError, ErrorClass.PERMISSION),
        (LimitExceededError, ErrorClass.LIMIT_EXCEEDED),
        (EngineConnectionError, ErrorClass.CONNECTION),
    ],
)
def test_every_error_carries_the_class_the_benchmark_buckets_it_under(
    error_type: type[AdapterError], expected: ErrorClass
) -> None:
    assert error_type("boom").error_class is expected


def test_errors_render_structured_output_not_a_traceback() -> None:
    # Arrange
    error = QuerySyntaxError(
        "Unknown identifier: Order",
        suggestion="quote reserved words with backticks on ClickHouse",
    )

    # Act
    payload = error.as_dict()

    # Assert
    assert payload == {
        "error_class": "syntax",
        "message": "Unknown identifier: Order",
        "suggestion": "quote reserved words with backticks on ClickHouse",
    }


def test_a_suggestion_is_optional_but_the_shape_is_not() -> None:
    assert QuerySemanticError("no such column").as_dict()["suggestion"] is None


def test_unsupported_capability_tells_the_caller_how_to_avoid_it() -> None:
    error = UnsupportedCapabilityError("databricks", Capability.PROJECTION)
    assert error.as_dict()["suggestion"] == ("check adapter.supports('projection') before calling")
