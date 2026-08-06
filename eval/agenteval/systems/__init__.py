"""Adapters for the systems agenteval scores (SPEC §11.5).

One module per system under test. Each one is a thin translation layer: agenteval
must be able to score a stack it did not write, so nothing here may assume
anything about how a system produces its SQL.
"""

from __future__ import annotations

from agenteval.systems.base import (
    Attempt,
    BlindAttempt,
    EmittedQuery,
    ErrorClass,
    ModelSpec,
    SystemUnderTest,
    TokenUsage,
)

__all__ = [
    "Attempt",
    "BlindAttempt",
    "EmittedQuery",
    "ErrorClass",
    "ModelSpec",
    "SystemUnderTest",
    "TokenUsage",
]
