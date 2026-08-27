"""The ClickHouse Agents conversation client (SPEC §11.5.1)."""

from __future__ import annotations

import json
import urllib.error
from base64 import b64decode
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pytest

from agenteval.systems.clickhouse_agents import (
    ClickHouseAgentsConfigError,
    ClickHouseAgentsConversation,
    ClickHouseAgentsTarget,
    build_conversation,
    post_json,
    read_payload,
)
from agenteval.systems.managed import ManagedError

ENV = {
    "AGENTEVAL_CH_AGENTS_HOST": "https://api.clickhouse.cloud/",
    "AGENTEVAL_CH_AGENTS_KEY_ID": "key",
    "AGENTEVAL_CH_AGENTS_KEY_SECRET": "secret",
}


@dataclass
class RecordingTransport:
    """A JSON POST that answers from a script and records what it was sent."""

    reply: Mapping[str, Any] = field(default_factory=dict)
    error: Exception | None = None
    calls: list[tuple[str, Mapping[str, Any], Mapping[str, str]]] = field(default_factory=list)

    async def __call__(
        self, url: str, body: Mapping[str, Any], headers: Mapping[str, str]
    ) -> Mapping[str, Any]:
        self.calls.append((url, body, headers))
        if self.error is not None:
            raise self.error
        return self.reply


# --------------------------------------------------------------------------
# the target
# --------------------------------------------------------------------------


def test_the_host_is_read_from_the_environment_without_its_trailing_slash() -> None:
    target = ClickHouseAgentsTarget.from_env(ENV)

    assert target.host == "https://api.clickhouse.cloud"


def test_a_missing_host_refuses_to_start() -> None:
    with pytest.raises(ClickHouseAgentsConfigError, match="AGENTEVAL_CH_AGENTS_HOST is unset"):
        ClickHouseAgentsTarget.from_env({})


def test_the_key_pair_becomes_basic_auth() -> None:
    headers = ClickHouseAgentsTarget.from_env(ENV).headers

    scheme, _, encoded = headers["Authorization"].partition(" ")
    assert scheme == "Basic"
    assert b64decode(encoded).decode() == "key:secret"


def test_an_unauthenticated_target_sends_no_authorization_header() -> None:
    headers = ClickHouseAgentsTarget(host="https://localhost").headers

    assert "Authorization" not in headers
    assert headers["Content-Type"] == "application/json"


# --------------------------------------------------------------------------
# reading a reply
# --------------------------------------------------------------------------


def test_the_configured_paths_decide_what_is_read_as_sql() -> None:
    answer = read_payload(
        {"attachments": [{"statement": "SELECT 1", "prose": "one row"}]},
        sql_path="attachments.0.statement",
        text_path="attachments.0.prose",
    )

    assert answer.sql == "SELECT 1"
    assert answer.text == "one row"


def test_a_reply_with_no_sql_is_a_decline() -> None:
    answer = read_payload({"text": "I cannot answer that."}, sql_path="sql", text_path="text")

    assert answer.sql is None
    assert answer.text == "I cannot answer that."


@pytest.mark.parametrize(
    "payload",
    [
        {"attachments": []},
        {"attachments": [{"statement": None}]},
        {"attachments": "not a list"},
        {"sql": {"nested": "object"}},
        {},
    ],
)
def test_a_path_that_leads_nowhere_reads_as_a_decline(payload: Mapping[str, Any]) -> None:
    answer = read_payload(payload, sql_path="attachments.0.statement", text_path="text")

    assert answer.sql is None


def test_a_non_numeric_step_into_a_list_reads_as_nothing() -> None:
    answer = read_payload(
        {"attachments": [{"statement": "SELECT 1"}]}, sql_path="attachments.first", text_path="text"
    )

    assert answer.sql is None


# --------------------------------------------------------------------------
# asking
# --------------------------------------------------------------------------


def build(transport: RecordingTransport, **response: str) -> ClickHouseAgentsConversation:
    return build_conversation(
        response, target=ClickHouseAgentsTarget.from_env(ENV), transport=transport
    )


async def test_the_question_is_posted_where_the_config_says() -> None:
    transport = RecordingTransport(reply={"sql": "SELECT 1", "text": "one"})
    conversation = build(transport, path="/v2/agents/{target_id}/ask", question_field="prompt")

    answer = await conversation.ask("agent-7", "How many rows are there?")

    url, body, headers = transport.calls[0]
    assert url == "https://api.clickhouse.cloud/v2/agents/agent-7/ask"
    assert body == {"prompt": "How many rows are there?"}
    assert "Authorization" in headers
    assert answer.sql == "SELECT 1"


async def test_defaults_apply_when_the_config_says_nothing() -> None:
    transport = RecordingTransport(reply={"sql": "SELECT 1"})
    conversation = build(transport)

    await conversation.ask("agent-7", "How many rows are there?")

    url, body, _ = transport.calls[0]
    assert url == "https://api.clickhouse.cloud/v1/agents/agent-7/conversations"
    assert body == {"message": "How many rows are there?"}


async def test_a_transport_failure_is_reported_as_a_managed_error() -> None:
    conversation = build(RecordingTransport(error=TimeoutError("gave up")))

    with pytest.raises(ManagedError, match=r"agent-7'.*TimeoutError: gave up"):
        await conversation.ask("agent-7", "How many rows are there?")


async def test_a_managed_error_from_the_transport_passes_through_unwrapped() -> None:
    conversation = build(RecordingTransport(error=ManagedError("502: Bad Gateway")))

    with pytest.raises(ManagedError, match=r"^502: Bad Gateway$"):
        await conversation.ask("agent-7", "How many rows are there?")


def test_an_unknown_response_key_is_refused() -> None:
    with pytest.raises(ClickHouseAgentsConfigError, match="unknown key\\(s\\): prose_path"):
        build_conversation(
            {"prose_path": "text"}, target=ClickHouseAgentsTarget.from_env(ENV), transport=None
        )


def test_the_live_transport_is_the_default() -> None:
    conversation = build_conversation({}, target=ClickHouseAgentsTarget.from_env(ENV))

    assert conversation.transport is post_json


# --------------------------------------------------------------------------
# the transport itself
# --------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None


async def test_a_json_object_comes_back_decoded(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def urlopen(request: Any, timeout: int) -> FakeResponse:
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse(b'{"sql": "SELECT 1"}')

    monkeypatch.setattr("urllib.request.urlopen", urlopen)

    payload = await post_json("https://host/ask", {"message": "hi"}, {"Accept": "application/json"})

    assert payload == {"sql": "SELECT 1"}
    assert captured["body"] == {"message": "hi"}
    assert captured["timeout"] == 120


async def test_an_http_error_names_the_status(monkeypatch: pytest.MonkeyPatch) -> None:
    def urlopen(request: Any, timeout: int) -> FakeResponse:
        raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr("urllib.request.urlopen", urlopen)

    with pytest.raises(ManagedError, match="answered 403: Forbidden"):
        await post_json("https://host/ask", {}, {})


async def test_an_unreachable_host_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    def urlopen(request: Any, timeout: int) -> FakeResponse:
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", urlopen)

    with pytest.raises(ManagedError, match="could not be read: OSError"):
        await post_json("https://host/ask", {}, {})


async def test_a_reply_that_is_not_an_object_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda request, timeout: FakeResponse(b'["not", "an", "object"]')
    )

    with pytest.raises(ManagedError, match="answered with list, not a JSON object"):
        await post_json("https://host/ask", {}, {})
