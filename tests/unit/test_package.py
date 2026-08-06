"""The package surface is small on purpose; keep it honest."""

from __future__ import annotations

import agentdb


def test_package_exports_config_surface_and_version() -> None:
    assert agentdb.__version__ == "0.1.0"
    assert set(agentdb.__all__) == {"Config", "ConfigError", "RetrievalWeights", "__version__"}
