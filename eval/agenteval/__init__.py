"""agenteval — a vendor-neutral harness for NL→SQL execution accuracy.

agenteval scores *any* agent stack: MCP servers, managed services, or a bare
model. It reaches every system under test through the ``SystemUnderTest``
protocol (SPEC §11.5) and therefore imports nothing from ``agentdb`` — a rule
enforced by an import-linter contract in CI, because a benchmark that depends on
one of its contestants is not a benchmark.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
