"""Managed analytics services under test (SPEC §11.5.1, §11.5.2)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from agenteval.suites import load_builtin
from agenteval.systems.managed import (
    DECLINE_NOTE,
    CuratedExample,
    ManagedAnswer,
    ManagedConfig,
    ManagedConfigError,
    ManagedError,
    ManagedSystem,
    check_example_overlap,
    load_managed_configs,
    parse_managed,
    write_config_record,
)
from agenteval.tasks import Task
from tests.harness_fakes import MODEL, FakeExecutor, sample_task

MANAGED_YAML = Path("eval/managed.yaml")


@dataclass
class ScriptedConversation:
    """A managed service that replays canned answers in order."""

    answers: list[ManagedAnswer] = field(default_factory=list)
    asked: list[tuple[str, str]] = field(default_factory=list)
    error: Exception | None = None

    async def ask(self, target_id: str, question: str) -> ManagedAnswer:
        self.asked.append((target_id, question))
        if self.error is not None:
            raise self.error
        return self.answers.pop(0)


def genie_config(**overrides: object) -> ManagedConfig:
    defaults: dict[str, object] = {
        "name": "S4b_genie_curated",
        "kind": "genie",
        "version": "2026-08",
        "target_id": "space-1",
    }
    return ManagedConfig(**{**defaults, **overrides})  # type: ignore[arg-type]


def agents_config(**overrides: object) -> ManagedConfig:
    defaults: dict[str, object] = {
        "name": "S3_clickhouse_agents",
        "kind": "clickhouse_agents",
        "version": "beta-2026-05",
        "target_id": "agent-1",
    }
    return ManagedConfig(**{**defaults, **overrides})  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


def test_a_config_names_the_engine_its_kind_belongs_to() -> None:
    assert genie_config().engine == "databricks"
    assert agents_config().engine == "clickhouse"


def test_a_config_without_curation_says_so() -> None:
    assert not agents_config().curated
    assert genie_config(instructions="answer with one query").curated


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("name", "", "needs a name"),
        ("kind", "vertex", "unknown kind"),
        ("version", "", "pinned version"),
        ("target_id", "", "id of the space or agent"),
    ],
)
def test_an_unusable_config_is_refused(field_name: str, value: str, message: str) -> None:
    with pytest.raises(ManagedConfigError, match=message):
        genie_config(**{field_name: value})


def test_the_committed_record_holds_everything_the_service_was_given() -> None:
    config = genie_config(
        tables=("samples.tpch.region",),
        instructions="one query only",
        examples=(CuratedExample(question="how many regions?", sql="SELECT count(*) FROM region"),),
        notes="the curated configuration",
        response={"sql_path": "sql"},
    )

    record = config.as_record()

    assert record["instructions"] == "one query only"
    assert record["examples"] == [
        {"question": "how many regions?", "sql": "SELECT count(*) FROM region"}
    ]
    assert record["tables"] == ["samples.tpch.region"]
    assert config.fingerprint.startswith("sha256:")


def test_two_configurations_that_differ_only_in_curation_fingerprint_differently() -> None:
    bare = genie_config(name="S4a_genie_minimal")
    curated = genie_config(
        name="S4a_genie_minimal", instructions="revenue is price times 1-discount"
    )

    assert bare.fingerprint != curated.fingerprint


def test_parsing_rejects_an_unknown_field() -> None:
    with pytest.raises(ManagedConfigError, match="unknown field"):
        parse_managed(
            {"name": "S4a", "kind": "genie", "version": "1", "target_id": "s", "curated": True}
        )


def test_parsing_rejects_a_missing_field() -> None:
    with pytest.raises(ManagedConfigError, match="missing: target_id"):
        parse_managed({"name": "S4a", "kind": "genie", "version": "1"})


def test_parsing_reads_examples_and_tables() -> None:
    config = parse_managed(
        {
            "name": "S4b_genie_curated",
            "kind": "genie",
            "version": "2026-08",
            "target_id": "space-1",
            "tables": ["samples.tpch.region"],
            "examples": [{"question": "how many regions?", "sql": "SELECT count(*) FROM region"}],
            "notes": "the curated configuration",
        }
    )

    assert config.tables == ("samples.tpch.region",)
    assert config.examples[0].question == "how many regions?"
    assert config.notes == "the curated configuration"


@pytest.mark.parametrize("raw", ["region", 7])
def test_parsing_rejects_tables_that_are_not_a_list(raw: object) -> None:
    with pytest.raises(ManagedConfigError, match="list of strings"):
        parse_managed(
            {"name": "S", "kind": "genie", "version": "1", "target_id": "s", "tables": raw}
        )


@pytest.mark.parametrize("raw", ["example", 7])
def test_parsing_rejects_examples_that_are_not_a_list(raw: object) -> None:
    with pytest.raises(ManagedConfigError, match="list of curated examples"):
        parse_managed(
            {"name": "S", "kind": "genie", "version": "1", "target_id": "s", "examples": raw}
        )


def test_parsing_rejects_an_example_missing_its_sql() -> None:
    with pytest.raises(ManagedConfigError, match="needs a question and a sql"):
        parse_managed(
            {
                "name": "S",
                "kind": "genie",
                "version": "1",
                "target_id": "s",
                "examples": [{"question": "how many regions?"}],
            }
        )


def test_loading_a_missing_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ManagedConfigError, match="no managed service config"):
        load_managed_configs(tmp_path / "absent.yaml")


def test_loading_rejects_a_document_that_is_not_a_list(tmp_path: Path) -> None:
    path = tmp_path / "managed.yaml"
    path.write_text("name: S4a\n", encoding="utf-8")

    with pytest.raises(ManagedConfigError, match="must contain a list"):
        load_managed_configs(path)


def test_loading_rejects_a_duplicated_arm(tmp_path: Path) -> None:
    path = tmp_path / "managed.yaml"
    entry = "- {name: S4a, kind: genie, version: '1', target_id: s}\n"
    path.write_text(entry * 2, encoding="utf-8")

    with pytest.raises(ManagedConfigError, match="defines S4a more than once"):
        load_managed_configs(path)


def test_the_configurations_are_committed_beside_the_traces(tmp_path: Path) -> None:
    path = write_config_record([genie_config(), agents_config()], tmp_path / "raw" / "run.json")

    committed = json.loads(path.read_text(encoding="utf-8"))
    assert [entry["name"] for entry in committed] == [
        "S4b_genie_curated",
        "S3_clickhouse_agents",
    ]


# --------------------------------------------------------------------------
# the overlap guard — SPEC 11.5.2
# --------------------------------------------------------------------------


def test_a_curated_example_repeating_a_gold_query_fails_the_run() -> None:
    task = sample_task()
    config = genie_config(
        examples=(
            CuratedExample(question="how big is the table?", sql=" select COUNT() from hits "),
        )
    )

    with pytest.raises(ManagedConfigError, match="identical SQL"):
        check_example_overlap(config, [task])


def test_a_curated_example_paraphrasing_a_question_fails_the_run() -> None:
    task = sample_task()
    config = genie_config(
        examples=(
            CuratedExample(
                question="How many rows are in the hits table today?",
                sql="SELECT uniq(UserID) FROM visits",
            ),
        )
    )

    with pytest.raises(ManagedConfigError, match="near-identical question"):
        check_example_overlap(config, [task])


def test_a_curated_example_paraphrasing_gold_sql_fails_the_run() -> None:
    task = sample_task()
    config = genie_config(
        examples=(
            CuratedExample(
                question="how many counters are there?", sql="SELECT COUNT(*) FROM hits"
            ),
        )
    )

    with pytest.raises(ManagedConfigError, match="near-identical SQL"):
        check_example_overlap(config, [task])


def test_an_example_with_no_words_to_compare_matches_nothing() -> None:
    """An empty side compares as zero overlap rather than as a perfect match.

    Jaccard of two empty sets is undefined; treating it as 1.0 would fail every
    run whose config carried a blank example.
    """
    config = genie_config(examples=(CuratedExample(question="", sql="???"),))

    check_example_overlap(config, [sample_task()])


def test_an_example_about_other_tables_is_allowed() -> None:
    config = genie_config(
        examples=(
            CuratedExample(
                question="List each region with the number of nations it contains.",
                sql="SELECT r_name, count(*) FROM region JOIN nation ON n_regionkey = r_regionkey",
            ),
        )
    )

    check_example_overlap(config, [sample_task()])


def test_the_committed_genie_space_shares_nothing_with_any_suite() -> None:
    """The test SPEC §11.5.2 asks for by name: no curated example is a gold query.

    Run against every shipped suite rather than the one the arm targets, so an
    example borrowed from the other engine's questions is caught too.
    """
    tasks: list[Task] = [
        task for suite in ("clickbench_nl", "tpch_nl") for task in load_builtin(suite)
    ]

    for config in load_managed_configs(MANAGED_YAML):
        check_example_overlap(config, tasks)


# --------------------------------------------------------------------------
# answering
# --------------------------------------------------------------------------


def build_system(
    conversation: ScriptedConversation,
    *,
    config: ManagedConfig | None = None,
    engine: str = "databricks",
) -> ManagedSystem:
    executor = FakeExecutor(engine=engine)  # type: ignore[arg-type]
    return ManagedSystem.create(
        config=config or genie_config(),
        conversation=conversation,
        executor=executor,
        tasks=[sample_task()],
    )


async def test_the_sql_the_service_wrote_is_re_executed_by_the_harness() -> None:
    conversation = ScriptedConversation(
        answers=[ManagedAnswer(sql="SELECT count() FROM hits", text="There are 100M rows.")]
    )
    system = build_system(conversation)

    attempt = await system.answer(sample_task(), None, seed=0)

    assert conversation.asked == [("space-1", "How many rows are in the hits table?")]
    assert attempt.queries[0].sql == "SELECT count() FROM hits"
    assert attempt.queries[0].succeeded
    assert any("There are 100M rows." in note for note in attempt.notes)


async def test_a_decline_is_scored_with_its_own_error_class() -> None:
    conversation = ScriptedConversation(
        answers=[ManagedAnswer(sql=None, text="I cannot answer that with these tables.")]
    )
    system = build_system(conversation)

    attempt = await system.answer(sample_task(), None, seed=0)

    assert attempt.queries[0].error_class == "declined"
    assert not attempt.queries[0].succeeded
    assert DECLINE_NOTE in attempt.notes


async def test_a_silent_decline_still_carries_an_explanation() -> None:
    system = build_system(ScriptedConversation(answers=[ManagedAnswer(sql=None)]))

    attempt = await system.answer(sample_task(), None, seed=0)

    assert attempt.queries[0].error_text == "the service returned no SQL"


async def test_vendor_notes_reach_the_trace() -> None:
    conversation = ScriptedConversation(
        answers=[ManagedAnswer(sql="SELECT 1", notes=("genie statement_id: abc",))]
    )
    system = build_system(conversation)

    attempt = await system.answer(sample_task(), None, seed=0)

    assert "genie statement_id: abc" in attempt.notes


async def test_the_arm_reports_no_tokens_of_its_own() -> None:
    system = build_system(ScriptedConversation(answers=[ManagedAnswer(sql="SELECT 1")]))

    attempt = await system.answer(sample_task(), None, seed=1)

    assert attempt.tokens.total == 0
    assert attempt.context_bytes == 0
    assert attempt.model is None


async def test_the_service_cannot_be_run_against_a_chosen_model() -> None:
    system = build_system(ScriptedConversation(answers=[ManagedAnswer(sql="SELECT 1")]))

    with pytest.raises(ManagedError, match="selects its own model"):
        await system.answer(sample_task(), MODEL, seed=0)


def test_the_arm_never_claims_to_control_the_model() -> None:
    system = build_system(ScriptedConversation())

    assert system.controls_model is False
    assert system.name == "S4b_genie_curated"
    assert system.version == "2026-08"
    assert system.config_fingerprint == genie_config().fingerprint


def test_an_arm_bound_to_the_other_engine_is_refused() -> None:
    with pytest.raises(ManagedConfigError, match="measures databricks but this run is on"):
        build_system(ScriptedConversation(), engine="clickhouse")


def test_a_curated_arm_is_never_built_against_a_suite_it_was_shown() -> None:
    config = genie_config(
        examples=(CuratedExample(question="how big is the table?", sql="SELECT count() FROM hits"),)
    )

    with pytest.raises(ManagedConfigError, match="drawn from the suite"):
        ManagedSystem.create(
            config=config,
            conversation=ScriptedConversation(),
            executor=FakeExecutor(engine="databricks"),
            tasks=[sample_task()],
        )
