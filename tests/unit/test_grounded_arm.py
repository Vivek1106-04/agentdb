"""The grounded arms and the seam that reaches a provider without importing it.

The arm is A0 with a different payload, and these tests hold it to that: the same
system prompt, the same loop, the same accounting. Anything else would make the
A1/A2 deltas a measurement of the arm rather than of the grounding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from agenteval.models.base import ModelError
from agenteval.systems.base import SystemUnderTest
from agenteval.systems.grounded import ContextProvider, GroundedSystem
from agenteval.systems.providers import (
    ProviderConfig,
    ProviderError,
    load_provider,
    load_provider_configs,
)
from agenteval.systems.raw_schema import RawSchemaSystem, build_system_prompt
from tests.harness_fakes import (
    MODEL,
    SYNTAX_ERROR,
    FakeExecutor,
    ScriptedModelClient,
    sample_task,
)

GROUNDED_PAYLOAD = (
    "CREATE TABLE hits (...)\n\nPhysical layout of agentdb.hits:\n- engine: MergeTree"
)


@dataclass
class FakeProvider:
    """A provider that returns a canned payload and records what it was asked."""

    payload: str = GROUNDED_PAYLOAD
    name: str = "agentdb/layout"
    version: str = "1.0"
    fingerprint: str = "sha256:fake"
    asked: list[tuple[str, str]] = field(default_factory=list)

    async def context(self, *, namespace: str, question: str) -> str:
        self.asked.append((namespace, question))
        return self.payload


def _system(arm: str = "A2_layout", **kwargs: object) -> tuple[GroundedSystem, FakeProvider]:
    provider = FakeProvider()
    system = GroundedSystem.create(
        arm=arm,
        provider=provider,
        executor=FakeExecutor(),
        client=ScriptedModelClient(replies=["```sql\nSELECT count() FROM hits\n```"]),
        **kwargs,  # type: ignore[arg-type]
    )
    return system, provider


# --------------------------------------------------------------------------
# the arm
# --------------------------------------------------------------------------


async def test_the_arm_puts_the_provider_payload_where_a0_puts_the_ddl() -> None:
    system, provider = _system()

    attempt = await system.answer(sample_task(), MODEL, seed=0)

    assert provider.asked == [("agentdb", "How many rows are in the hits table?")]
    assert attempt.prompt is not None
    assert GROUNDED_PAYLOAD in attempt.prompt
    assert attempt.context_bytes == len(GROUNDED_PAYLOAD.encode("utf-8"))
    assert attempt.system == "A2_layout"
    assert attempt.queries[0].succeeded


async def test_a_grounded_arm_shares_a0s_system_prompt_so_the_arms_stay_comparable() -> None:
    system, _ = _system()
    client = system.client

    await system.answer(sample_task(), MODEL, seed=0)

    assert isinstance(client, ScriptedModelClient)
    assert client.calls[0][0] == build_system_prompt("clickhouse")


def test_the_arm_satisfies_the_system_under_test_protocol() -> None:
    system, _ = _system(arm="A1_stats")

    assert isinstance(system, SystemUnderTest)
    assert system.name == "A1_stats"
    assert system.controls_model is True


def test_two_arms_over_different_grounding_have_different_fingerprints() -> None:
    stats, _ = _system(arm="A1_stats")
    layout, _ = _system(arm="A2_layout")

    assert stats.config_fingerprint != layout.config_fingerprint


def test_the_fingerprint_covers_the_providers_own_configuration() -> None:
    baseline, _ = _system()
    other = GroundedSystem.create(
        arm="A2_layout",
        provider=FakeProvider(fingerprint="sha256:different"),
        executor=FakeExecutor(),
        client=ScriptedModelClient(),
    )

    assert baseline.config_fingerprint != other.config_fingerprint


def test_a_negative_retry_budget_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="max_retries"):
        _system(max_retries=-1)


async def test_an_arm_that_controls_its_model_refuses_to_run_without_one() -> None:
    system, _ = _system()

    with pytest.raises(ModelError):
        await system.answer(sample_task(), None, seed=0)


async def test_the_grounded_arm_retries_on_the_same_loop_a0_uses() -> None:
    provider = FakeProvider()
    executor = FakeExecutor(outcomes=[SYNTAX_ERROR])
    system = GroundedSystem.create(
        arm="A2_layout",
        provider=provider,
        executor=executor,
        client=ScriptedModelClient(replies=["```sql\nSELEC 1\n```", "```sql\nSELECT 1\n```"]),
    )

    attempt = await system.answer(sample_task(), MODEL, seed=0)

    assert len(attempt.queries) == 2
    assert attempt.queries[-1].succeeded
    assert provider.asked == [("agentdb", "How many rows are in the hits table?")]


def test_the_grounded_arm_and_the_baseline_arm_agree_on_the_prompt_they_share() -> None:
    baseline = RawSchemaSystem.create(executor=FakeExecutor(), client=ScriptedModelClient())
    grounded, _ = _system()

    assert build_system_prompt(baseline.executor.engine) == build_system_prompt(
        grounded.executor.engine
    )


# --------------------------------------------------------------------------
# loading a provider the harness must not import
# --------------------------------------------------------------------------


def _module_with(**attributes: object) -> ModuleType:
    module = ModuleType("fake_provider_module")
    for name, value in attributes.items():
        setattr(module, name, value)
    return module


async def test_a_provider_is_resolved_by_dotted_path_and_given_its_options() -> None:
    seen: dict[str, object] = {}

    def factory(**options: object) -> FakeProvider:
        seen.update(options)
        return FakeProvider(name="third_party/grounding")

    module = _module_with(factory=factory)

    provider = await load_provider(
        "somebody.elses:factory", {"level": "layout"}, importer=lambda _: module
    )

    assert isinstance(provider, ContextProvider)
    assert provider.name == "third_party/grounding"
    assert seen == {"level": "layout"}


async def test_a_factory_that_has_to_connect_may_be_a_coroutine() -> None:
    async def factory() -> FakeProvider:
        return FakeProvider()

    module = _module_with(factory=factory)

    provider = await load_provider("somebody.elses:factory", importer=lambda _: module)

    assert provider.name == "agentdb/layout"


@pytest.mark.parametrize("spec", ["", "no_separator", ":factory", "module:"])
async def test_a_malformed_spec_is_rejected_before_anything_is_imported(spec: str) -> None:
    with pytest.raises(ProviderError, match="module:factory"):
        await load_provider(spec, importer=lambda _: _module_with())


async def test_a_missing_module_says_which_module_was_missing() -> None:
    def importer(name: str) -> ModuleType:
        raise ImportError(f"no module named {name!r}")

    with pytest.raises(ProviderError, match="cannot import provider module"):
        await load_provider("nowhere:factory", importer=importer)


async def test_a_missing_factory_says_which_attribute_was_missing() -> None:
    with pytest.raises(ProviderError, match="has no attribute 'factory'"):
        await load_provider("somebody.elses:factory", importer=lambda _: _module_with())


async def test_something_that_is_not_a_provider_is_refused_with_the_shape_it_needed() -> None:
    module = _module_with(factory=lambda: SimpleNamespace(name="not a provider"))

    with pytest.raises(ProviderError, match="not a context provider"):
        await load_provider("somebody.elses:factory", importer=lambda _: module)


# --------------------------------------------------------------------------
# the arm catalogue on disk
# --------------------------------------------------------------------------


def test_the_shipped_catalogue_names_the_family_a_arms(tmp_path: Path) -> None:
    configs = load_provider_configs(Path("eval/providers.yaml"))

    assert [config.arm for config in configs] == [
        "A1_stats",
        "A2_layout",
        "A3_plan",
        "A4_memory",
        "A5_negmemory",
    ]
    assert all(config.provider.startswith("agentdb.bench.") for config in configs)
    assert configs[0].options["level"] == "stats"
    assert [config.plan_review for config in configs] == [False, False, True, True, True]


def test_a_config_needs_an_arm_and_a_provider() -> None:
    with pytest.raises(ProviderError, match="needs an arm name"):
        ProviderConfig(arm="", provider="module:factory")
    with pytest.raises(ProviderError, match="needs a provider spec"):
        ProviderConfig(arm="A1_stats", provider="")


def test_a_missing_catalogue_is_reported_by_path(tmp_path: Path) -> None:
    with pytest.raises(ProviderError, match="no provider config at"):
        load_provider_configs(tmp_path / "absent.yaml")


def test_a_catalogue_that_is_not_a_list_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "providers.yaml"
    path.write_text("arm: A1_stats\n", encoding="utf-8")

    with pytest.raises(ProviderError, match="must contain a list"):
        load_provider_configs(path)


def test_an_unknown_field_is_refused_rather_than_silently_ignored(tmp_path: Path) -> None:
    path = tmp_path / "providers.yaml"
    path.write_text("- arm: A1_stats\n  provider: mod:factory\n  levl: stats\n", encoding="utf-8")

    with pytest.raises(ProviderError, match="unknown field\\(s\\): levl"):
        load_provider_configs(path)


def test_the_same_arm_twice_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "providers.yaml"
    path.write_text(
        "- arm: A1_stats\n  provider: mod:factory\n- arm: A1_stats\n  provider: mod:other\n",
        encoding="utf-8",
    )

    with pytest.raises(ProviderError, match="A1_stats more than once"):
        load_provider_configs(path)
