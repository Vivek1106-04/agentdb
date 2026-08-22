"""Connecting the exemplar store to its Postgres (SPEC §10.2).

One function, kept apart from :mod:`agentdb.core.memory.store` so the store
itself never imports a driver: that is what lets the store's logic be covered at
100% against a fake connection while the real driver stays an optional extra.

The import is injected rather than taken at module scope, the same way the
engine adapters take theirs — a missing driver has to fail with the command that
installs it, not with a traceback at import time.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from types import ModuleType
from typing import cast

from agentdb.config import Config
from agentdb.core.memory.store import Connection, MemoryStoreError

DRIVER = "psycopg"
"""psycopg 3. The store speaks plain DB-API, so the version is about the
connection string and the array/JSONB adaptation, not about any pgvector type
registration — embeddings cross as pgvector's own text form."""

Importer = Callable[[str], ModuleType]


def connect(dsn: str | None = None, *, importer: Importer = importlib.import_module) -> Connection:
    """Open a connection to the exemplar store, defaulting to the configured DSN."""
    try:
        module = importer(DRIVER)
    except ImportError as exc:
        raise MemoryStoreError(
            "the exemplar store needs the 'psycopg' driver; "
            "install the optional extra with: uv sync --extra memory"
        ) from exc

    return cast(Connection, module.connect(dsn or Config().memory_dsn))
