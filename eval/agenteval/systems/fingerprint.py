"""Config fingerprints (SPEC §11.3).

Every Family S row carries the pinned version and the full effective config of
the system it measured, because a benchmark of a moving beta product means
nothing without saying exactly what was measured. The fingerprint is the short
form that goes in the table; the config itself is committed alongside.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def config_fingerprint(config: Mapping[str, Any]) -> str:
    """A stable ``sha256:`` digest of an effective config.

    Sorted keys and a canonical separator, so two machines that ran the same
    config agree on the fingerprint and a diff in the report is a real diff.
    """
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"
