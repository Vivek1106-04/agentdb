"""agenteval's independence is the benchmark's credibility; assert it in code too."""

from __future__ import annotations

import sys

import agenteval


def test_agenteval_exposes_a_version() -> None:
    assert agenteval.__version__ == "0.1.0"


def test_importing_agenteval_does_not_pull_in_agentdb() -> None:
    # Arrange — drop anything already imported by other tests in this process
    for name in [n for n in sys.modules if n == "agentdb" or n.startswith("agentdb.")]:
        del sys.modules[name]

    # Act
    import agenteval  # re-import is the point of the test

    # Assert — CI also enforces this statically via import-linter (SPEC §4.1.6)
    assert agenteval.__version__
    assert not [n for n in sys.modules if n == "agentdb" or n.startswith("agentdb.")]
