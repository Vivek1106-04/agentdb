"""Opening the exemplar store's connection (SPEC §10.2).

The driver is an optional extra, so the one thing that must never happen is an
``ImportError`` traceback at import time. A missing psycopg has to say what to
install, in the same shape the engine adapters use for theirs.
"""

from __future__ import annotations

from types import ModuleType, SimpleNamespace

import pytest

from agentdb.core.memory.postgres import connect
from agentdb.core.memory.store import MemoryStoreError


def fake_driver(recorder: list[str]) -> ModuleType:
    def fake_connect(dsn: str) -> SimpleNamespace:
        recorder.append(dsn)
        return SimpleNamespace(dsn=dsn)

    module = ModuleType("psycopg")
    module.connect = fake_connect  # type: ignore[attr-defined]  # a stand-in driver module
    return module


def test_a_missing_driver_names_the_command_that_installs_it() -> None:
    def missing(name: str) -> ModuleType:
        raise ImportError(name)

    with pytest.raises(MemoryStoreError, match="uv sync --extra memory"):
        connect(importer=missing)


def test_the_configured_dsn_is_the_default() -> None:
    seen: list[str] = []

    connect(importer=lambda _: fake_driver(seen))

    assert seen == ["postgresql://agentdb:agentdb@localhost:55432/agentdb"]


def test_an_explicit_dsn_wins() -> None:
    seen: list[str] = []

    connect("postgresql://elsewhere/agentdb", importer=lambda _: fake_driver(seen))

    assert seen == ["postgresql://elsewhere/agentdb"]
