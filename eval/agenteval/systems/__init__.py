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
from agenteval.systems.fingerprint import config_fingerprint

# Concrete arms are imported from their own modules, never re-exported here: an
# arm imports the model layer, the model layer imports these value types, and a
# package that pulled the arms in eagerly would close that loop.
__all__ = [
    "Attempt",
    "BlindAttempt",
    "EmittedQuery",
    "ErrorClass",
    "ModelSpec",
    "SystemUnderTest",
    "TokenUsage",
    "config_fingerprint",
]
