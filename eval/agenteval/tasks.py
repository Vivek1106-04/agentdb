"""Task and suite definitions (SPEC §11.2).

A task is a natural-language question plus the SQL that is known to answer it.
Tasks are data, not code: they live in YAML under ``suites/`` so that authoring
them leaves a git history separate from the systems being measured — the
evidence behind SPEC §11.5.1's "do not tune against a system under test".

Loading validates strictly. A malformed task is a benchmark defect, and a
benchmark that silently skips malformed tasks is reporting on a task set nobody
can enumerate.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import yaml

Engine = Literal["clickhouse", "databricks"]
"""Engines a task can target. Declared here, not imported: agenteval depends on
no part of agentdb (SPEC §4.1.6)."""

Difficulty = Literal["easy", "medium", "hard"]

_REQUIRED_FIELDS = ("id", "suite", "engine", "question", "gold_sql")
_ALLOWED_FIELDS = frozenset(
    {*_REQUIRED_FIELDS, "difficulty", "tags", "gold_result_hash", "notes", "namespace"}
)
GOLD_SQL_KEY_SUFFIX = "_sql"
"""Suffix marking a lock-entry key as a gold-SQL fingerprint for one engine
(``clickhouse_sql``) rather than the name of an engine."""

GOLD_LOCK_NAME = "gold.lock.yaml"
"""Committed gold hashes live in one sidecar per suite, not scattered through the
task files. A reviewer auditing gold drift then reads a single diff, and
authoring a task never touches the same lines as verifying one."""

_ENGINES: frozenset[str] = frozenset({"clickhouse", "databricks"})
_DIFFICULTIES: frozenset[str] = frozenset({"easy", "medium", "hard"})


class TaskLoadError(ValueError):
    """Raised when a task file cannot be trusted. Never skipped, never inferred."""


@dataclass(frozen=True, slots=True)
class Task:
    """One benchmark question and its gold answer."""

    id: str
    suite: str
    engines: tuple[Engine, ...]
    question: str
    gold_sql: str
    namespace: str = "agentdb"
    difficulty: Difficulty = "medium"
    tags: tuple[str, ...] = ()
    gold_result_hashes: tuple[tuple[Engine, str], ...] = ()
    """Hash of the canonical gold result **per engine**, so drift is detectable.

    Per engine, not per task, and that is not a nicety. The same question over
    the same scale factor still hashes differently on two engines, because the
    drivers render values differently — the Databricks Statement Execution API
    returns every cell as text where ClickHouse returns typed numbers. One
    shared hash would make a cross-engine suite fail on whichever engine did not
    freeze it, which is exactly the suite that matters (SPEC §18.6).
    """

    notes: str | None = None
    """Why the task is interesting — usually the schema semantics it probes."""

    source_path: str | None = None

    def targets(self, engine: Engine) -> bool:
        """Whether this task is meant to run against ``engine``."""
        return engine in self.engines

    def gold_hash_for(self, engine: Engine) -> str | None:
        """The committed hash for ``engine``, if one was frozen."""
        return next((digest for name, digest in self.gold_result_hashes if name == engine), None)


@dataclass(frozen=True, slots=True)
class TaskSuite:
    """An ordered, immutable collection of tasks sharing a name."""

    name: str
    tasks: tuple[Task, ...]

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self) -> Iterator[Task]:
        return iter(self.tasks)

    def for_engine(self, engine: Engine) -> TaskSuite:
        """The subset of this suite that targets ``engine``."""
        return TaskSuite(name=self.name, tasks=tuple(t for t in self.tasks if t.targets(engine)))

    def subset(self, size: int) -> TaskSuite:
        """The first ``size`` tasks — the ``make bench-quick`` path (SPEC §14.1)."""
        if size <= 0:
            raise ValueError(f"subset size must be > 0, got {size}")
        return TaskSuite(name=self.name, tasks=self.tasks[:size])

    def by_id(self, task_id: str) -> Task:
        for task in self.tasks:
            if task.id == task_id:
                return task
        raise KeyError(f"no task {task_id!r} in suite {self.name!r}")


def parse_task(payload: Mapping[str, Any], *, source_path: str | None = None) -> Task:
    """Validate one task mapping and build a :class:`Task`.

    Unknown keys are an error rather than a shrug: a typo in ``gold_result_hash``
    that silently disables gold-drift detection is exactly the failure a
    benchmark cannot afford.
    """
    missing = [name for name in _REQUIRED_FIELDS if name not in payload]
    if missing:
        raise TaskLoadError(f"task is missing required field(s): {', '.join(missing)}")

    unknown = sorted(set(payload) - _ALLOWED_FIELDS)
    if unknown:
        raise TaskLoadError(f"task {payload['id']!r} has unknown field(s): {', '.join(unknown)}")

    engines = _parse_engines(payload["id"], payload["engine"])
    difficulty = payload.get("difficulty", "medium")
    if difficulty not in _DIFFICULTIES:
        raise TaskLoadError(
            f"task {payload['id']!r} has unknown difficulty {difficulty!r}; "
            f"expected one of {sorted(_DIFFICULTIES)}"
        )

    gold_sql = str(payload["gold_sql"]).strip()
    if not gold_sql:
        raise TaskLoadError(f"task {payload['id']!r} has an empty gold_sql")

    question = str(payload["question"]).strip()
    if not question:
        raise TaskLoadError(f"task {payload['id']!r} has an empty question")

    return Task(
        id=str(payload["id"]),
        suite=str(payload["suite"]),
        engines=engines,
        question=question,
        gold_sql=gold_sql,
        namespace=str(payload.get("namespace", "agentdb")),
        difficulty=difficulty,
        tags=tuple(str(tag) for tag in payload.get("tags", ())),
        gold_result_hashes=_parse_hashes(payload["id"], payload.get("gold_result_hash")),
        notes=_optional_str(payload.get("notes")),
        source_path=source_path,
    )


def load_suite(directory: Path) -> TaskSuite:
    """Load every ``*.yaml`` task under ``directory`` into one suite.

    Tasks are sorted by id so a run's task order is a property of the data, not
    of the filesystem — otherwise two machines produce different seeds-to-task
    pairings and the seeds stop meaning anything.
    """
    if not directory.is_dir():
        raise TaskLoadError(f"task directory does not exist: {directory}")

    paths = [path for path in sorted(directory.glob("*.yaml")) if path.name != GOLD_LOCK_NAME]
    tasks = [task for path in paths for task in _load_file(path)]
    if not tasks:
        raise TaskLoadError(f"no tasks found in {directory}")

    _assert_unique_ids(tasks)
    tasks = _apply_gold_lock(tasks, directory / GOLD_LOCK_NAME)
    suites = {task.suite for task in tasks}
    if len(suites) != 1:
        raise TaskLoadError(
            f"{directory} mixes suites {sorted(suites)}; one directory holds one suite"
        )

    return TaskSuite(name=next(iter(suites)), tasks=tuple(sorted(tasks, key=lambda t: t.id)))


def _load_file(path: Path) -> list[Task]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise TaskLoadError(f"{path} is not valid YAML: {exc}") from exc

    if document is None:
        raise TaskLoadError(f"{path} is empty")

    payloads: Sequence[Any] = document if isinstance(document, list) else [document]
    tasks: list[Task] = []
    for payload in payloads:
        if not isinstance(payload, Mapping):
            raise TaskLoadError(f"{path} contains a {type(payload).__name__}, expected a mapping")
        tasks.append(parse_task(payload, source_path=str(path)))
    return tasks


def _apply_gold_lock(tasks: Sequence[Task], lock_path: Path) -> list[Task]:
    """Attach committed gold hashes from the suite's sidecar, if it has one."""
    if not lock_path.is_file():
        return list(tasks)

    document = yaml.safe_load(lock_path.read_text(encoding="utf-8")) or {}
    if not isinstance(document, Mapping):
        raise TaskLoadError(f"{lock_path} must be a mapping of task id to hash")

    known = {task.id for task in tasks}
    unknown = sorted(set(document) - known)
    if unknown:
        raise TaskLoadError(f"{lock_path} names task(s) that do not exist: {', '.join(unknown)}")

    locked = []
    for task in tasks:
        entry = document.get(task.id)
        if entry is None:
            locked.append(task)
            continue
        if task.gold_result_hashes:
            raise TaskLoadError(
                f"task {task.id!r} has a gold_result_hash in both its file and "
                f"{lock_path.name}; the lock file is the one source of truth"
            )
        hashes = _parse_hashes(task.id, entry)
        locked.append(replace(task, gold_result_hashes=_fresh_hashes(task, entry, hashes)))
    return locked


def gold_sql_fingerprint(gold_sql: str) -> str:
    """Identify the question a committed hash was taken against.

    Without this the lock records only "task id produced this hash", so editing a
    task's ``gold_sql`` makes the next freeze report drift — the hash of the old
    question against the result of the new one — and the only way past it is to
    hand-delete the entry. That conflates the two things the lock exists to tell
    apart: data that changed underneath a fixed question, which must fail loudly,
    and a question the author deliberately rewrote, which must not.
    """
    return "sha256:" + sha256(gold_sql.strip().encode("utf-8")).hexdigest()


def gold_sql_key(engine: str) -> str:
    """Where one engine's gold-SQL fingerprint lives in a lock entry.

    Per engine, not per task, because a cross-engine suite is frozen once per
    engine and those freezes happen at different times. A single shared
    fingerprint would be rewritten by whichever engine was frozen last, silently
    re-validating the other engine's stale hash — which is exactly the drift the
    lock exists to catch.
    """
    return f"{engine}{GOLD_SQL_KEY_SUFFIX}"


def _fresh_hashes(
    task: Task, entry: object, hashes: tuple[tuple[Engine, str], ...]
) -> tuple[tuple[Engine, str], ...]:
    """Drop any engine's hash that was frozen against different SQL than the task now has.

    An engine with no committed fingerprint predates them and is trusted, so
    older lock files keep working exactly as before.
    """
    if not isinstance(entry, Mapping):
        return hashes
    current = gold_sql_fingerprint(task.gold_sql)
    return tuple(
        (engine, digest)
        for engine, digest in hashes
        if str(entry.get(gold_sql_key(engine), current)) == current
    )


def _parse_hashes(task_id: object, raw: object) -> tuple[tuple[Engine, str], ...]:
    """Read a committed hash, in either the flat or the per-engine form.

    A bare string means "this hash holds on every engine the task targets",
    which is what a single-engine suite writes and what an author writes by
    hand. A mapping names the engine each hash was frozen on, which is what a
    cross-engine suite needs: the same question over the same data still hashes
    differently on two engines, because their drivers render values differently.
    """
    if raw is None:
        return ()
    if isinstance(raw, Mapping):
        hashes: list[tuple[Engine, str]] = []
        for name, digest in raw.items():
            engine = str(name)
            if engine.endswith(GOLD_SQL_KEY_SUFFIX):
                continue
            if engine not in _ENGINES:
                raise TaskLoadError(
                    f"task {task_id!r} commits a gold hash for unknown engine {engine!r}; "
                    f"expected one of {sorted(_ENGINES)}"
                )
            text = _optional_str(digest)
            if text is None:
                raise TaskLoadError(f"task {task_id!r} has an empty gold hash for {engine!r}")
            hashes.append((engine, text))  # type: ignore[arg-type]
        return tuple(sorted(hashes))
    text = _optional_str(raw)
    if text is None:
        return ()
    return tuple((engine, text) for engine in sorted(_ENGINES))  # type: ignore[misc]


def _parse_engines(task_id: object, raw: object) -> tuple[Engine, ...]:
    values = raw if isinstance(raw, list) else [raw]
    if not values:
        raise TaskLoadError(f"task {task_id!r} declares no engine")

    engines: list[Engine] = []
    for value in values:
        name = str(value)
        if name not in _ENGINES:
            raise TaskLoadError(
                f"task {task_id!r} targets unknown engine {name!r}; "
                f"expected one of {sorted(_ENGINES)}"
            )
        if name not in engines:
            engines.append(name)  # type: ignore[arg-type]
    return tuple(engines)


def _assert_unique_ids(tasks: Sequence[Task]) -> None:
    seen: dict[str, str | None] = {}
    for task in tasks:
        if task.id in seen:
            raise TaskLoadError(
                f"duplicate task id {task.id!r} in {task.source_path} and {seen[task.id]}"
            )
        seen[task.id] = task.source_path


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
