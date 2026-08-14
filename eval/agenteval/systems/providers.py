"""Loading a context provider by dotted path (SPEC §4.1.6, §18.3).

The harness must be able to score a grounding layer it does not import — that is
the property that makes it a scoreboard rather than a self-report, and it is
enforced in CI by an import-linter contract. So providers are named in
configuration and resolved at runtime:

``agentdb.bench.provider:clickhouse_layout_provider``

The same mechanism a reader uses to plug in their own grounding service is the
one agentdb's own arms go through. Nothing here knows what agentdb is.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

from agenteval.systems.grounded import ContextProvider

Importer = Callable[[str], ModuleType]

SEPARATOR = ":"


class ProviderError(RuntimeError):
    """A configured context provider could not be loaded or is not one."""


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    """One Family A arm, described by data rather than by code.

    The arm name and the dotted path are committed with the run, so a reader can
    see which grounding produced a row without reading the harness.
    """

    arm: str
    provider: str
    options: Mapping[str, Any] = field(default_factory=dict)
    plan_review: bool = False
    """Whether the arm shows the model its plan before the query runs (``A3``).

    Declared here rather than inferred from the provider: an arm that reviews
    plans is a different arm, and the report has to be able to say which."""

    def __post_init__(self) -> None:
        if not self.arm:
            raise ProviderError("a provider config needs an arm name")
        if not self.provider:
            raise ProviderError(
                f"arm {self.arm!r} needs a provider spec 'module{SEPARATOR}factory'"
            )


def load_provider_configs(path: Path) -> tuple[ProviderConfig, ...]:
    """Load a YAML list of Family A arms, e.g. ``eval/providers.yaml``."""
    if not path.is_file():
        raise ProviderError(f"no provider config at {path}")

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, Sequence) or isinstance(document, str):
        raise ProviderError(f"{path} must contain a list of provider configs")

    configs = tuple(_parse(entry, path) for entry in document)
    arms = [config.arm for config in configs]
    duplicates = sorted({arm for arm in arms if arms.count(arm) > 1})
    if duplicates:
        raise ProviderError(f"{path} defines {', '.join(duplicates)} more than once")
    return configs


def _parse(payload: Mapping[str, Any], path: Path) -> ProviderConfig:
    allowed = {"arm", "provider", "options", "plan_review"}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ProviderError(f"{path}: provider config has unknown field(s): {', '.join(unknown)}")
    return ProviderConfig(
        arm=str(payload.get("arm", "")),
        provider=str(payload.get("provider", "")),
        options=dict(payload.get("options", {})),
        plan_review=bool(payload.get("plan_review", False)),
    )


async def load_provider(
    spec: str,
    options: Mapping[str, Any] | None = None,
    *,
    importer: Importer = importlib.import_module,
) -> ContextProvider:
    """Resolve ``module:factory`` and call it with ``options``.

    The factory is called rather than imported as an instance so a provider can
    open a connection of its own — the harness supplies configuration, not
    plumbing. A factory that needs to connect may be a coroutine function, and is
    awaited here.
    """
    module_name, _, attribute = spec.partition(SEPARATOR)
    if not module_name or not attribute:
        raise ProviderError(f"provider spec must be 'module{SEPARATOR}factory', got {spec!r}")

    try:
        module = importer(module_name)
    except ImportError as exc:
        raise ProviderError(f"cannot import provider module {module_name!r}: {exc}") from exc

    try:
        factory = getattr(module, attribute)
    except AttributeError as exc:
        raise ProviderError(f"{module_name!r} has no attribute {attribute!r}") from exc

    provider = factory(**dict(options or {}))
    if inspect.isawaitable(provider):
        provider = await provider
    if not isinstance(provider, ContextProvider):
        raise ProviderError(
            f"{spec} returned {type(provider).__name__}, which is not a context provider: "
            "it needs name, version, fingerprint and an async context(namespace, question)"
        )
    return provider
