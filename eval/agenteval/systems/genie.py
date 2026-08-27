"""Databricks AI/BI Genie, through the Conversation API (SPEC §11.5.2).

The client half of the ``S4a_genie_minimal`` and ``S4b_genie_curated`` arms;
everything about how a Genie row is scored lives in
:mod:`agenteval.systems.managed`, which this module knows nothing about beyond
the answer type it returns.

Genie replies with a list of attachments: some carry prose, and at most one
carries the SQL it wrote plus the id of the statement it ran. The SQL is what
the harness takes — it re-executes it through the harness's own read-only
connection so that a Genie row is graded by exactly the comparison every other
row is graded by. Genie's own result set is deliberately not used: it arrives
formatted for a chat window, and grading formatted output against a gold result
set would be inventing a comparison.

The response is read through :func:`getattr` rather than against imported SDK
models, for the same reason the rest of this package does: the harness must
import, and its unit tests must run, on a machine with no vendor SDK installed.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from agenteval.engines.connect import DatabricksTarget, Importer, build_workspace
from agenteval.systems.managed import ManagedAnswer, ManagedError

STATEMENT_NOTE = "genie statement_id: {statement_id}"
"""Genie's own id for the statement it ran, kept so a reader can join this
attempt to the warehouse's record of it without any string matching."""

CONVERSATION_STATUS_NOTE = "genie message status: {status}"


@dataclass(frozen=True, slots=True)
class GenieConversation:
    """One Genie workspace's conversation surface.

    ``api`` is the SDK's ``WorkspaceClient.genie``, injected rather than built
    here so a unit test can hand over a fake and never reach a workspace.
    """

    api: Any

    async def ask(self, target_id: str, question: str) -> ManagedAnswer:
        """Start a conversation in space ``target_id`` and wait for the answer.

        One question, one fresh conversation. Genie carries context between
        turns of the same conversation, so reusing one across tasks would let
        task N's answer be informed by task N-1 — a leak between cells that no
        other arm has and that the seed structure could not undo.
        """
        try:
            message = self.api.start_conversation_and_wait(space_id=target_id, content=question)
        except Exception as exc:
            raise ManagedError(
                f"Genie space {target_id!r} did not answer: {type(exc).__name__}: {exc}"
            ) from exc
        return read_message(message)


def read_message(message: Any) -> ManagedAnswer:
    """Reduce one Genie message to the SQL, the prose, and where it came from.

    A message with no query attachment is a decline — Genie answering in prose
    alone, which §11.5.2 scores as a failure with its own error class rather
    than as an absent cell.
    """
    sql: str | None = None
    statement_id: str | None = None
    prose: list[str] = []

    for attachment in getattr(message, "attachments", None) or ():
        query = getattr(attachment, "query", None)
        text = getattr(attachment, "text", None)
        candidate = str(getattr(query, "query", "") or "").strip() if query is not None else ""
        if candidate and sql is None:
            sql = candidate
            statement_id = _optional(getattr(query, "statement_id", None))
        content = str(getattr(text, "content", "") or "").strip() if text is not None else ""
        if content:
            prose.append(content)

    notes = [CONVERSATION_STATUS_NOTE.format(status=_status(message))]
    if statement_id:
        notes.append(STATEMENT_NOTE.format(statement_id=statement_id))
    return ManagedAnswer(sql=sql, text="\n".join(prose), notes=tuple(notes))


def build_genie_conversation(
    target: DatabricksTarget, *, importer: Importer = importlib.import_module
) -> GenieConversation:
    """The live client, reached as the same principal the executor uses."""
    return GenieConversation(api=build_workspace(target, importer=importer).genie)


def _status(message: Any) -> str:
    """The message's status as a bare name.

    The SDK hands back a ``MessageStatus`` enum whose ``str()`` carries the
    class name; its ``value`` is what a reader of the trace wants to see.
    """
    raw = getattr(message, "status", None)
    return str(getattr(raw, "value", raw)) if raw is not None else "unknown"


def _optional(raw: object) -> str | None:
    text = str(raw or "").strip()
    return text or None
