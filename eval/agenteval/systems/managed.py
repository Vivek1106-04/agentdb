"""Managed analytics services, scored (SPEC §11.5.1, §11.5.2) — ``S3``, ``S4a``, ``S4b``.

ClickHouse Agents and Databricks AI/BI Genie are the same shape from the
harness's side: a question goes in as natural language, and what comes back is
prose plus, when the service chose to write one, a SQL query. Neither exposes a
model to control, neither reports tokens, and neither can be asked to explain
itself the way a Family A arm can. So one system class serves both, and the two
vendor modules beside this one hold only the conversation client.

Three rules from §11.5.2 are structural here rather than promised in prose:

* **The result set is graded, never the prose.** The SQL the service produced is
  re-executed through the harness's own read-only connection, exactly as
  :mod:`agenteval.systems.mcp_generic` does, so grading is byte-identical across
  every arm in the table.
* **A decline is a measurement.** A service that answers without writing a query
  scores incorrect with ``error_class="declined"`` rather than vanishing into
  ``no_query``, and the report gives declines their own column.
* **Curated examples may not overlap gold.** :func:`check_example_overlap` runs
  before a curated arm is built, and a hit fails the run. Scoring a space that
  had been handed the answers would be the single most likely way to publish a
  fraudulent number.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Protocol, cast, runtime_checkable

import yaml

from agenteval.execution import QueryExecutor
from agenteval.systems.base import DECLINED, Attempt, EmittedQuery, ModelSpec, TokenUsage
from agenteval.systems.fingerprint import config_fingerprint
from agenteval.tasks import Engine, Task

ManagedKind = Literal["genie", "clickhouse_agents"]
"""Which vendor's conversation API an arm speaks to."""

ENGINE_FOR_KIND: Mapping[ManagedKind, Engine] = {
    "genie": "databricks",
    "clickhouse_agents": "clickhouse",
}
"""A managed service is bound to its own engine. Naming a Genie arm in a
ClickHouse run is a configuration error, not a cross-engine measurement."""

DECLINE_NOTE = "the service answered without producing a query"
DECLINE_TEXT = "the service returned no SQL"
PROSE_NOTE = "service prose: {text}"

PARAPHRASE_SIMILARITY = 0.8
"""Token overlap at which a curated example counts as a paraphrase of a task.

Deliberately strict in the direction that costs a run rather than the direction
that costs the benchmark's credibility: a false positive here is an author
rewriting one example, and a false negative is a published number taken from a
space that had been shown the answers.
"""

_WORD = re.compile(r"[a-z0-9_]+")


class ManagedConfigError(ValueError):
    """A managed-service arm that cannot be trusted or launched."""


class ManagedError(RuntimeError):
    """The service could not be asked, or was asked for something it cannot do."""


@dataclass(frozen=True, slots=True)
class CuratedExample:
    """One question/SQL pair a curated space was given before any run."""

    question: str
    sql: str

    def as_record(self) -> dict[str, str]:
        return {"question": self.question, "sql": self.sql}


@dataclass(frozen=True, slots=True)
class ManagedConfig:
    """One measured managed service, in full.

    Everything the service was given is here and is committed with the run:
    table scope, instruction text, and every curated example. §11.5.2 exists
    because Genie's accuracy is a function of exactly these fields, so a row
    that did not publish them would say nothing at all.
    """

    name: str
    """The arm name, e.g. ``S4a_genie_minimal``."""

    kind: ManagedKind
    version: str
    """Pinned and printed in the report. These are moving beta products."""

    target_id: str
    """The Genie space id, or the ClickHouse Agents agent id."""

    tables: tuple[str, ...] = ()
    instructions: str = ""
    examples: tuple[CuratedExample, ...] = ()
    notes: str = ""
    """Why this configuration exists — the sentence a reader needs to compare it
    against the other one."""

    response: Mapping[str, str] = field(default_factory=dict)
    """Vendor-specific reading instructions, where the client needs them. Held
    here rather than in code so the trace records exactly what was read out of
    the service's reply (see :mod:`agenteval.systems.clickhouse_agents`)."""

    def __post_init__(self) -> None:
        if not self.name:
            raise ManagedConfigError("a managed service config needs a name")
        if self.kind not in ENGINE_FOR_KIND:
            raise ManagedConfigError(
                f"managed arm {self.name!r} has unknown kind {self.kind!r}; "
                f"expected one of {sorted(ENGINE_FOR_KIND)}"
            )
        if not self.version:
            raise ManagedConfigError(f"managed arm {self.name!r} needs a pinned version")
        if not self.target_id:
            raise ManagedConfigError(
                f"managed arm {self.name!r} needs the id of the space or agent it measures"
            )

    @property
    def engine(self) -> Engine:
        return ENGINE_FOR_KIND[self.kind]

    @property
    def curated(self) -> bool:
        """Whether this configuration was given anything beyond table scope."""
        return bool(self.instructions or self.examples)

    def as_record(self) -> dict[str, Any]:
        """The committed description — the whole of what the service was given."""
        return {
            "name": self.name,
            "kind": self.kind,
            "version": self.version,
            "target_id": self.target_id,
            "tables": list(self.tables),
            "instructions": self.instructions,
            "examples": [example.as_record() for example in self.examples],
            "notes": self.notes,
            "response": dict(self.response),
        }

    @property
    def fingerprint(self) -> str:
        return config_fingerprint(self.as_record())


def parse_managed(payload: Mapping[str, Any]) -> ManagedConfig:
    """Build one config from a mapping, rejecting anything unrecognised."""
    allowed = {
        "name",
        "kind",
        "version",
        "target_id",
        "tables",
        "instructions",
        "examples",
        "notes",
        "response",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ManagedConfigError(
            f"managed service config has unknown field(s): {', '.join(unknown)}"
        )

    missing = [key for key in ("name", "kind", "version", "target_id") if key not in payload]
    if missing:
        raise ManagedConfigError(f"managed service config is missing: {', '.join(missing)}")

    return ManagedConfig(
        name=str(payload["name"]),
        kind=cast(ManagedKind, str(payload["kind"])),
        version=str(payload["version"]),
        target_id=str(payload["target_id"]),
        tables=_strings(payload.get("tables", ())),
        instructions=str(payload.get("instructions", "")),
        examples=_examples(payload.get("examples", ())),
        notes=str(payload.get("notes", "")),
        response={str(k): str(v) for k, v in dict(payload.get("response", {})).items()},
    )


def load_managed_configs(path: Path) -> tuple[ManagedConfig, ...]:
    """Load a YAML list of managed-service descriptions, e.g. ``eval/managed.yaml``."""
    if not path.is_file():
        raise ManagedConfigError(f"no managed service config at {path}")

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, Sequence) or isinstance(document, str):
        raise ManagedConfigError(f"{path} must contain a list of managed service configs")

    configs = tuple(parse_managed(entry) for entry in document)
    names = [config.name for config in configs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ManagedConfigError(f"{path} defines {', '.join(duplicates)} more than once")
    return configs


def write_config_record(configs: Sequence[ManagedConfig], path: Path) -> Path:
    """Commit the measured configurations beside the traces (SPEC §11.5.2).

    The fingerprint in each trace record says two runs used the same space; only
    this file says what that space actually was.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [config.as_record() for config in configs]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def check_example_overlap(config: ManagedConfig, tasks: Sequence[Task]) -> None:
    """Refuse a curated space that was handed any suite's answer.

    Exact repetition and paraphrase both count, on the question as well as on
    the SQL: a space given "which counter had the most hits, phrased slightly
    differently" has been given the task, and scoring it would publish a number
    about a leak rather than about the product.
    """
    violations = [
        f"example {index} ({reason}) matches task {task.id}"
        for index, example in enumerate(config.examples)
        for task in tasks
        for reason in _overlap_reasons(example, task)
    ]
    if violations:
        raise ManagedConfigError(
            f"curated arm {config.name!r} was given {len(violations)} example(s) drawn from the "
            f"suite: {'; '.join(violations)}. Rewrite them against tables the suite does not "
            "ask about, or against questions it does not ask."
        )


def _overlap_reasons(example: CuratedExample, task: Task) -> tuple[str, ...]:
    """Every way one curated example collides with one task."""
    reasons = []
    if _normalize_sql(example.sql) == _normalize_sql(task.gold_sql):
        reasons.append("identical SQL")
    elif _similarity(_tokens(example.sql), _tokens(task.gold_sql)) >= PARAPHRASE_SIMILARITY:
        reasons.append("near-identical SQL")
    if _similarity(_tokens(example.question), _tokens(task.question)) >= PARAPHRASE_SIMILARITY:
        reasons.append("near-identical question")
    return tuple(reasons)


@dataclass(frozen=True, slots=True)
class ManagedAnswer:
    """What a managed service said, reduced to the parts the harness grades."""

    sql: str | None
    """``None`` when the service declined — answered without writing a query."""

    text: str = ""
    """The conversational reply. Committed to the trace, never graded."""

    notes: tuple[str, ...] = ()
    """Vendor-side provenance, e.g. the service's own statement id."""


@runtime_checkable
class Conversation(Protocol):
    """One managed service's ask-a-question surface."""

    async def ask(self, target_id: str, question: str) -> ManagedAnswer:
        """Put ``question`` to the service and return what it answered."""
        ...


@dataclass(frozen=True, slots=True)
class ManagedSystem:
    """One managed analytics service under test."""

    config: ManagedConfig
    conversation: Conversation
    executor: QueryExecutor

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def version(self) -> str:
        return self.config.version

    @property
    def controls_model(self) -> bool:
        """Always ``False``: the service selects its own model (SPEC §11.5.2).

        Every row it produces carries the footnote saying so, and the report
        must not present it and an agentdb-on-Opus-5 row as a controlled model
        comparison. They are not one.
        """
        return False

    @property
    def config_fingerprint(self) -> str:
        return self.config.fingerprint

    @classmethod
    def create(
        cls,
        *,
        config: ManagedConfig,
        conversation: Conversation,
        executor: QueryExecutor,
        tasks: Sequence[Task],
    ) -> ManagedSystem:
        """Build the arm, refusing a curated space that overlaps the suite.

        ``tasks`` is required rather than optional: a guard that can be skipped
        by omitting an argument is not a guard.
        """
        if executor.engine != config.engine:
            raise ManagedConfigError(
                f"managed arm {config.name!r} measures {config.engine} but this run is on "
                f"{executor.engine}; it cannot be scored here"
            )
        check_example_overlap(config, tasks)
        return cls(config=config, conversation=conversation, executor=executor)

    async def answer(self, task: Task, model: ModelSpec | None, seed: int) -> Attempt:
        if model is not None:
            raise ManagedError(
                f"{self.name} selects its own model; it cannot be run against {model}"
            )

        started = perf_counter()
        answer = await self.conversation.ask(self.config.target_id, task.question)
        notes = list(answer.notes)
        if answer.text:
            notes.append(PROSE_NOTE.format(text=" ".join(answer.text.split())))

        if answer.sql is None:
            notes.append(DECLINE_NOTE)
            queries = (
                EmittedQuery(
                    sql="",
                    succeeded=False,
                    error_class=DECLINED,
                    error_text=answer.text or DECLINE_TEXT,
                ),
            )
        else:
            queries = (await self.executor.run(answer.sql, task.namespace),)

        return Attempt(
            system=self.name,
            task_id=task.id,
            seed=seed,
            model=None,
            prompt=task.question,
            queries=queries,
            # Tokens stay zero: the service reports none, and an estimate would
            # put a fabricated number in a cost column beside measured ones.
            tokens=TokenUsage(),
            context_bytes=0,
            wall_clock_ms=round((perf_counter() - started) * 1000),
            notes=tuple(notes),
        )


def _normalize_sql(sql: str) -> str:
    return " ".join(sql.split()).lower().rstrip(";").strip()


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_WORD.findall(text.lower()))


def _similarity(left: frozenset[str], right: frozenset[str]) -> float:
    """Jaccard overlap, and zero when either side has nothing to compare."""
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _strings(raw: object) -> tuple[str, ...]:
    if isinstance(raw, str) or not isinstance(raw, Sequence):
        raise ManagedConfigError(f"expected a list of strings, got {raw!r}")
    return tuple(str(item) for item in raw)


def _examples(raw: object) -> tuple[CuratedExample, ...]:
    if isinstance(raw, str) or not isinstance(raw, Sequence):
        raise ManagedConfigError(f"expected a list of curated examples, got {raw!r}")

    examples = []
    for entry in raw:
        if not isinstance(entry, Mapping) or not {"question", "sql"} <= set(entry):
            raise ManagedConfigError(f"a curated example needs a question and a sql, got {entry!r}")
        examples.append(CuratedExample(question=str(entry["question"]), sql=str(entry["sql"])))
    return tuple(examples)
