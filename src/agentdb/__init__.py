"""agentdb — plan-grounded context for analytical database agents.

The package is the *reference implementation under test*; the benchmark that
measures it lives in ``eval/agenteval`` and must never import from here
(SPEC §4.1.6, enforced by an import-linter contract in CI).
"""

from __future__ import annotations

from agentdb.config import Config, ConfigError, RetrievalWeights

__version__ = "0.1.0"

__all__ = ["Config", "ConfigError", "RetrievalWeights", "__version__"]
