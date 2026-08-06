"""Engine adapters and the contract they satisfy (SPEC §6).

Core imports from this package only — never from ``agentdb.adapters.postgres``
or ``agentdb.adapters.clickhouse`` directly. That rule is an import-linter
contract, not a convention.
"""

from __future__ import annotations

from agentdb.adapters.base import (
    Adapter,
    AdapterError,
    BaseAdapter,
    Capability,
    EngineConnectionError,
    LimitExceededError,
    PlanRejectionError,
    QueryPermissionError,
    QuerySemanticError,
    QuerySyntaxError,
    QueryTimeoutError,
    UnsupportedCapabilityError,
)
from agentdb.adapters.models import (
    ColumnDef,
    ColumnProfile,
    DialectRules,
    Engine,
    ErrorClass,
    ExplainMode,
    IndexDef,
    Limits,
    PhysicalLayout,
    Projection,
    RawPlan,
    Relation,
    RelationDetail,
    RelationRef,
    ResultSet,
    SampleMethod,
    SamplePolicy,
    SkipIndex,
    TimeWindow,
    WorkloadEntry,
)

__all__ = [
    "Adapter",
    "AdapterError",
    "BaseAdapter",
    "Capability",
    "ColumnDef",
    "ColumnProfile",
    "DialectRules",
    "Engine",
    "EngineConnectionError",
    "ErrorClass",
    "ExplainMode",
    "IndexDef",
    "LimitExceededError",
    "Limits",
    "PhysicalLayout",
    "PlanRejectionError",
    "Projection",
    "QueryPermissionError",
    "QuerySemanticError",
    "QuerySyntaxError",
    "QueryTimeoutError",
    "RawPlan",
    "Relation",
    "RelationDetail",
    "RelationRef",
    "ResultSet",
    "SampleMethod",
    "SamplePolicy",
    "SkipIndex",
    "TimeWindow",
    "UnsupportedCapabilityError",
    "WorkloadEntry",
]
