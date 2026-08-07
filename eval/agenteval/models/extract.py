"""Pulling one SQL statement out of a model's reply.

Every arm is graded on the query a system emitted, so extraction is part of the
measurement, not a convenience. Two rules follow from that:

* Extraction is **generous** — a model that wraps its answer in prose is
  answering the question, and scoring it as "no query" would measure formatting
  compliance instead of SQL ability.
* Extraction is **identical for every system**, so no arm can gain accuracy by
  being parsed more sympathetically than another.
"""

from __future__ import annotations

import re

_FENCED_SQL = re.compile(r"```[ \t]*sql[ \t]*\r?\n(.*?)```", re.DOTALL | re.IGNORECASE)
_FENCED_ANY = re.compile(r"```[a-zA-Z]*[ \t]*\r?\n(.*?)```", re.DOTALL)
_BARE_START = re.compile(r"^[ \t]*(?:WITH|SELECT)\b", re.IGNORECASE | re.MULTILINE)


def extract_sql(text: str) -> str | None:
    """The single SQL statement ``text`` offers, or ``None`` if it offers none.

    The **last** fenced block wins: a model that reconsiders mid-reply meant the
    query it finished with, and grading the one it abandoned would be scoring a
    draft.
    """
    for pattern in (_FENCED_SQL, _FENCED_ANY):
        blocks = pattern.findall(text)
        if blocks:
            return _tidy(blocks[-1])

    match = _BARE_START.search(text)
    if match is None:
        return None
    return _tidy(text[match.start() :])


def _tidy(sql: str) -> str | None:
    """Trim whitespace and one trailing terminator; report emptiness as ``None``."""
    trimmed = sql.strip()
    while trimmed.endswith(";"):
        trimmed = trimmed[:-1].rstrip()
    return trimmed or None
