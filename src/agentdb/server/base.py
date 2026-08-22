"""What a tool is, and the argument handling every tool shares (SPEC §13).

A tool here is data — name, description, two schemas and an async handler — not
a subclass and not a decorator registry. That keeps the catalog inspectable
without starting a server, which is what the contract tests need and what makes
``tools/list`` reviewable in a diff.

Argument errors are raised, not returned as plausible defaults. A tool that
quietly substitutes a default namespace for a missing one answers a question the
agent did not ask, and the agent has no way to notice.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field

from agentdb.adapters import Adapter, Capability, RelationRef
from agentdb.config import Config
from agentdb.core import ContextBuilder, PlanExplainer
from agentdb.core.memory.store import ExemplarStore
from agentdb.server.schemas import JsonValue

Handler = Callable[[Mapping[str, JsonValue]], Awaitable[dict[str, JsonValue]]]
"""One tool's implementation: arguments in, ``structuredContent`` out."""


class ToolError(RuntimeError):
    """A tool could not do what was asked.

    Carries a ``suggestion`` wherever one exists, because the agent on the other
    end can act on "qualify the name as catalog.schema.table" and cannot act on a
    traceback (SPEC §12).
    """

    def __init__(self, message: str, *, suggestion: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.suggestion = suggestion

    def as_dict(self) -> dict[str, JsonValue]:
        """The structured error shape a client receives."""
        return {"error": self.message, "suggestion": self.suggestion}


@dataclass(frozen=True, slots=True)
class ToolDef:
    """One tool as the server advertises and dispatches it."""

    name: str
    title: str
    description: str
    input_schema: dict[str, JsonValue]
    output_schema: dict[str, JsonValue]
    handler: Handler = field(compare=False)


@dataclass(frozen=True, slots=True)
class ServerContext:
    """Everything the tools are allowed to reach.

    One adapter per server process: a tool that could switch engines mid-session
    would make every plan warning ambiguous about which engine produced it.
    """

    adapter: Adapter
    config: Config = field(default_factory=Config)
    store: ExemplarStore | None = None
    """The exemplar store, when one is configured (SPEC §10).

    Optional because the memory tools are only advertised where they can work: a
    server run without Postgres serves the rest of the catalog rather than three
    tools that fail on call.
    """

    @property
    def builder(self) -> ContextBuilder:
        return ContextBuilder(adapter=self.adapter, config=self.config)

    @property
    def explainer(self) -> PlanExplainer:
        return PlanExplainer(adapter=self.adapter, config=self.config)

    def parse_relation(self, name: str) -> RelationRef:
        """Turn a written name into a reference, refusing an ambiguous one.

        Under Unity Catalog a two-part name resolves against session ``USE``
        state that a stateless server does not have, so a short name is rejected
        rather than guessed at. That is the ``UNQUALIFIED_RELATION`` hazard of
        SPEC §7 caught one layer earlier, before it can silently read the wrong
        table.
        """
        parts = name.split(".")
        three_level = self.adapter.supports(Capability.THREE_LEVEL_NAMESPACE)
        if any(not part for part in parts):
            raise ToolError(
                f"{name!r} has an empty name part",
                suggestion="write the name without a leading, trailing or doubled dot",
            )
        if len(parts) == 3:
            return RelationRef(catalog=parts[0], namespace=parts[1], name=parts[2])
        if len(parts) == 2 and not three_level:
            return RelationRef(namespace=parts[0], name=parts[1])
        expected = "catalog.schema.table" if three_level else "database.table"
        raise ToolError(
            f"{name!r} is not a fully qualified relation name on {self.adapter.engine}",
            suggestion=f"write it as {expected}",
        )


def require_str(args: Mapping[str, JsonValue], key: str) -> str:
    """A required string argument, or a ``ToolError`` naming what is missing."""
    value = args.get(key)
    if not isinstance(value, str) or not value:
        raise ToolError(f"argument {key!r} is required and must be a non-empty string")
    return value


def optional_str(args: Mapping[str, JsonValue], key: str) -> str | None:
    value = args.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ToolError(f"argument {key!r} must be a string")
    return value


def optional_int(args: Mapping[str, JsonValue], key: str, *, minimum: int = 1) -> int | None:
    value = args.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ToolError(f"argument {key!r} must be an integer >= {minimum}")
    return value


def require_str_list(args: Mapping[str, JsonValue], key: str) -> list[str]:
    """A required non-empty array of strings."""
    value = args.get(key)
    if not isinstance(value, list) or not value:
        raise ToolError(f"argument {key!r} is required and must be a non-empty array of strings")
    if not all(isinstance(item, str) for item in value):
        raise ToolError(f"argument {key!r} must contain only strings")
    return [str(item) for item in value]
