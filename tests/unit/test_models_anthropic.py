"""The Anthropic adapter, exercised without a network call or an API key."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from types import ModuleType

import pytest

from agenteval.models.anthropic import API_KEY_ENV, AnthropicClient, Message, Usage, build_create
from agenteval.models.base import DEFAULT_MAX_OUTPUT_TOKENS, ModelError, Turn
from agenteval.systems.base import ModelSpec

MODEL = ModelSpec(provider="anthropic", name="claude-opus-5")


@dataclass
class FakeUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class FakeTextBlock:
    text: str


@dataclass
class FakeThinkingBlock:
    """A content block with no ``text`` — the adapter must skip it, not crash."""

    thinking: str = "..."


@dataclass
class FakeMessage:
    content: Sequence[object]
    usage: Usage = field(default_factory=FakeUsage)
    stop_reason: str | None = "end_turn"


@dataclass
class RecordingCreate:
    """Stands in for ``client.messages.create``."""

    message: Message
    calls: list[dict[str, object]] = field(default_factory=list)

    async def __call__(
        self,
        *,
        model: str,
        max_tokens: int,
        temperature: float,
        system: str,
        messages: list[dict[str, str]],
    ) -> Message:
        self.calls.append(
            {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system,
                "messages": messages,
            }
        )
        return self.message


def _client(*blocks: object, usage: FakeUsage | None = None) -> AnthropicClient:
    message = FakeMessage(content=list(blocks), usage=usage or FakeUsage())
    return AnthropicClient(create=RecordingCreate(message=message))


async def test_a_completion_returns_the_concatenated_text_and_usage() -> None:
    # Arrange
    client = _client(FakeTextBlock("SELECT 1"), usage=FakeUsage(input_tokens=120, output_tokens=8))

    # Act
    response = await client.complete(
        system="be terse", turns=(Turn(role="user", content="how many?"),), model=MODEL, seed=3
    )

    # Assert
    assert response.text == "SELECT 1"
    assert response.tokens.input_tokens == 120
    assert response.tokens.output_tokens == 8
    assert response.stop_reason == "end_turn"


async def test_non_text_blocks_are_skipped() -> None:
    client = _client(FakeThinkingBlock(), FakeTextBlock("SELECT 1"), FakeTextBlock("-- done"))

    response = await client.complete(system="s", turns=(), model=MODEL, seed=0)

    assert response.text == "SELECT 1\n-- done"


async def test_turns_are_passed_through_in_order() -> None:
    create = RecordingCreate(message=FakeMessage(content=[FakeTextBlock("ok")]))
    client = AnthropicClient(create=create)

    await client.complete(
        system="be terse",
        turns=(Turn(role="user", content="q"), Turn(role="assistant", content="a")),
        model=MODEL,
        seed=0,
    )

    assert create.calls[0]["messages"] == [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]
    assert create.calls[0]["system"] == "be terse"
    assert create.calls[0]["model"] == "claude-opus-5"


async def test_max_output_tokens_defaults_and_can_be_overridden() -> None:
    create = RecordingCreate(message=FakeMessage(content=[FakeTextBlock("ok")]))
    client = AnthropicClient(create=create)

    await client.complete(system="s", turns=(), model=MODEL, seed=0)
    await client.complete(
        system="s", turns=(), model=ModelSpec("anthropic", "claude-sonnet-5", 0.0, 512), seed=0
    )

    assert create.calls[0]["max_tokens"] == DEFAULT_MAX_OUTPUT_TOKENS
    assert create.calls[1]["max_tokens"] == 512


async def test_a_provider_failure_becomes_a_model_error_naming_the_seed() -> None:
    async def explode(
        *,
        model: str,
        max_tokens: int,
        temperature: float,
        system: str,
        messages: list[dict[str, str]],
    ) -> Message:
        raise TimeoutError("upstream timed out")

    client = AnthropicClient(create=explode)

    with pytest.raises(ModelError, match="at seed 4: upstream timed out"):
        await client.complete(system="s", turns=(), model=MODEL, seed=4)


class FakeAnthropicModule(ModuleType):
    """Stands in for the real ``anthropic`` module, so no SDK install is needed."""

    def __init__(self, create: RecordingCreate) -> None:
        super().__init__("anthropic")
        self._create = create
        self.api_keys: list[str] = []

    def AsyncAnthropic(self, *, api_key: str) -> object:  # noqa: N802 - mirrors the SDK name
        self.api_keys.append(api_key)
        messages = type("Messages", (), {"create": self._create})
        return type("Client", (), {"messages": messages})


def test_build_create_binds_the_sdk_entry_point() -> None:
    create = RecordingCreate(message=FakeMessage(content=[]))
    module = FakeAnthropicModule(create)

    bound = build_create(api_key="sk-test", importer=lambda _: module)

    assert bound is create
    assert module.api_keys == ["sk-test"]


def test_build_create_reads_the_key_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(API_KEY_ENV, "sk-env")
    module = FakeAnthropicModule(RecordingCreate(message=FakeMessage(content=[])))

    build_create(importer=lambda _: module)

    assert module.api_keys == ["sk-env"]


def test_a_missing_key_is_an_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(API_KEY_ENV, raising=False)

    with pytest.raises(ModelError, match=f"{API_KEY_ENV} is not set"):
        build_create()


def test_a_missing_sdk_is_an_actionable_error() -> None:
    def missing(name: str) -> ModuleType:
        raise ImportError(f"no module named {name}")

    with pytest.raises(ModelError, match="uv sync --extra anthropic"):
        build_create(api_key="sk-test", importer=missing)
