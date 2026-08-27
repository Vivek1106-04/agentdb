"""The Genie Conversation API client (SPEC §11.5.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from agenteval.engines.connect import DatabricksTarget, EngineConnectionError
from agenteval.systems.genie import (
    GenieConversation,
    build_genie_conversation,
    read_message,
)
from agenteval.systems.managed import ManagedError

TARGET = DatabricksTarget(host="https://dbc-1.cloud.databricks.com", warehouse_id="w1", token="t")


def attachment(
    *, sql: str | None = None, statement_id: str | None = None, text: str | None = None
) -> Any:
    """One Genie attachment in the shape the SDK returns it."""
    return SimpleNamespace(
        query=SimpleNamespace(query=sql, statement_id=statement_id) if sql is not None else None,
        text=SimpleNamespace(content=text) if text is not None else None,
    )


def message(*attachments: Any, status: str = "COMPLETED") -> Any:
    return SimpleNamespace(attachments=list(attachments), status=SimpleNamespace(value=status))


@dataclass
class FakeGenieApi:
    """The SDK's ``WorkspaceClient.genie``, scripted."""

    reply: Any = None
    error: Exception | None = None
    calls: list[tuple[str, str]] = field(default_factory=list)

    def start_conversation_and_wait(self, *, space_id: str, content: str) -> Any:
        self.calls.append((space_id, content))
        if self.error is not None:
            raise self.error
        return self.reply


# --------------------------------------------------------------------------
# reading a message
# --------------------------------------------------------------------------


def test_the_sql_attachment_is_what_gets_graded() -> None:
    answer = read_message(
        message(
            attachment(text="Here is the revenue by region."),
            attachment(sql="SELECT r_name FROM region", statement_id="01ef"),
        )
    )

    assert answer.sql == "SELECT r_name FROM region"
    assert answer.text == "Here is the revenue by region."
    assert "genie statement_id: 01ef" in answer.notes
    assert "genie message status: COMPLETED" in answer.notes


def test_only_the_first_query_attachment_is_taken() -> None:
    answer = read_message(message(attachment(sql="SELECT 1"), attachment(sql="SELECT 2")))

    assert answer.sql == "SELECT 1"


def test_prose_alone_is_a_decline() -> None:
    answer = read_message(message(attachment(text="I need more detail to answer that.")))

    assert answer.sql is None
    assert answer.text == "I need more detail to answer that."
    assert not any(note.startswith("genie statement_id") for note in answer.notes)


def test_an_empty_query_attachment_is_a_decline_too() -> None:
    answer = read_message(message(attachment(sql="   ")))

    assert answer.sql is None


def test_a_message_with_no_attachments_is_a_decline() -> None:
    answer = read_message(SimpleNamespace(attachments=None, status=None))

    assert answer.sql is None
    assert answer.text == ""
    assert answer.notes == ("genie message status: unknown",)


def test_a_query_attachment_without_a_statement_id_still_reads() -> None:
    answer = read_message(message(attachment(sql="SELECT 1", statement_id=" ")))

    assert answer.sql == "SELECT 1"
    assert answer.notes == ("genie message status: COMPLETED",)


# --------------------------------------------------------------------------
# asking
# --------------------------------------------------------------------------


async def test_each_question_starts_its_own_conversation() -> None:
    api = FakeGenieApi(reply=message(attachment(sql="SELECT 1")))
    conversation = GenieConversation(api=api)

    await conversation.ask("space-1", "How many regions are there?")
    await conversation.ask("space-1", "And how many nations?")

    assert api.calls == [
        ("space-1", "How many regions are there?"),
        ("space-1", "And how many nations?"),
    ]


async def test_a_workspace_failure_is_reported_as_a_managed_error() -> None:
    conversation = GenieConversation(api=FakeGenieApi(error=TimeoutError("gave up")))

    with pytest.raises(ManagedError, match=r"space-1'.*TimeoutError: gave up"):
        await conversation.ask("space-1", "How many regions are there?")


# --------------------------------------------------------------------------
# building the client
# --------------------------------------------------------------------------


def fake_sdk(genie: object) -> ModuleType:
    module = ModuleType("databricks.sdk")
    module.WorkspaceClient = lambda **_: SimpleNamespace(genie=genie)  # type: ignore[attr-defined]
    return module


def test_the_client_is_built_from_the_same_workspace_the_executor_uses() -> None:
    api = FakeGenieApi()

    conversation = build_genie_conversation(TARGET, importer=lambda _: fake_sdk(api))

    assert conversation.api is api


def test_a_missing_sdk_says_what_to_install() -> None:
    def importer(name: str) -> ModuleType:
        raise ImportError(name)

    with pytest.raises(EngineConnectionError, match="databricks-sdk"):
        build_genie_conversation(TARGET, importer=importer)
