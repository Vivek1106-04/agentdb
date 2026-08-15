"""Claude Code as a measured system, driven through its CLI (SPEC §11.5).

The subprocess path is exercised for real against a stub executable rather than
mocked: the failure modes that matter here — a non-zero exit, a hang, a binary
that is not installed, a JSON document that reports its own error — all live in
the plumbing, and a mock would assert that the mock works.

What these tests pin above all is that the arm reports what it cannot control.
Claude Code carries its own scaffolding and the operator's instruction files
into every call, and a number that hid that would be measuring something nobody
could name.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from agenteval.models.base import ModelError, Turn
from agenteval.models.claude_cli import (
    DENIED_TOOLS,
    PROVIDER,
    ClaudeCliClient,
    environment_report,
    parse_result,
    render_prompt,
)
from agenteval.systems.base import ModelSpec

MODEL = ModelSpec(provider=PROVIDER, name="sonnet")

# Shaped like a real `claude -p --output-format json` document, trimmed to the
# fields this adapter reads. Captured from Claude Code 2.1.233.
LIVE_DOCUMENT = {
    "is_error": False,
    "stop_reason": "end_turn",
    "session_id": "5e761523-f416-43e6-9422-2b594809757f",
    "total_cost_usd": 0.113466,
    "usage": {
        "input_tokens": 2,
        "cache_creation_input_tokens": 18830,
        "cache_read_input_tokens": 0,
        "output_tokens": 32,
    },
    "result": "```sql\nSELECT COUNT(*) FROM samples.tpch.nation;\n```",
    "type": "result",
}


def _stub(directory: Path, body: str) -> Path:
    """A fake ``claude`` binary that behaves however a test needs."""
    path = directory / "claude-stub"
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def _client(executable: Path, **overrides: object) -> ClaudeCliClient:
    return ClaudeCliClient(executable=str(executable), **overrides)  # type: ignore[arg-type]


# -- reading the CLI's answer -----------------------------------------------


def test_a_real_document_yields_the_answer_and_its_accounting() -> None:
    result = parse_result(json.dumps(LIVE_DOCUMENT))

    assert "SELECT COUNT(*)" in result.text
    assert result.input_tokens == 2
    assert result.output_tokens == 32
    # the operator wrote 2 tokens of input; the product added 18,830
    assert result.scaffolding_tokens == 18_830
    assert result.stop_reason == "end_turn"
    assert result.cost_usd == pytest.approx(0.113466)


def test_scaffolding_counts_cache_creation_and_cache_reads_together() -> None:
    document = {
        **LIVE_DOCUMENT,
        "usage": {
            "input_tokens": 5,
            "cache_creation_input_tokens": 1_000,
            "cache_read_input_tokens": 20_000,
            "output_tokens": 7,
        },
    }

    assert parse_result(json.dumps(document)).scaffolding_tokens == 21_000


def test_a_document_reporting_its_own_error_is_a_failed_call_not_an_empty_answer() -> None:
    document = {"is_error": True, "result": "Not logged in - Please run /login"}

    with pytest.raises(ModelError, match="Not logged in"):
        parse_result(json.dumps(document))


def test_output_that_is_not_json_names_what_came_back() -> None:
    with pytest.raises(ModelError, match="did not return JSON"):
        parse_result("Not logged in - Please run /login")


def test_output_that_is_json_but_not_an_object_is_refused() -> None:
    with pytest.raises(ModelError, match="expected an object"):
        parse_result("[1, 2, 3]")


def test_a_document_with_no_result_text_is_refused() -> None:
    with pytest.raises(ModelError, match="no result text"):
        parse_result(json.dumps({"is_error": False, "usage": {}}))


def test_absent_accounting_reads_as_zero_rather_than_crashing() -> None:
    result = parse_result(json.dumps({"result": "hi"}))

    assert (result.input_tokens, result.output_tokens, result.scaffolding_tokens) == (0, 0, 0)
    assert result.cost_usd is None
    assert result.session_id is None


def test_a_boolean_masquerading_as_a_count_is_not_counted() -> None:
    document = {"result": "hi", "usage": {"input_tokens": True}, "total_cost_usd": True}

    result = parse_result(json.dumps(document))

    assert result.input_tokens == 0
    assert result.cost_usd is None


# -- rendering the conversation ---------------------------------------------


def test_a_single_turn_is_sent_verbatim() -> None:
    assert render_prompt((Turn(role="user", content="Question: how many?"),)) == (
        "Question: how many?"
    )


def test_a_retry_conversation_is_flattened_with_its_roles_named() -> None:
    # print mode takes one prompt, so a self-correction turn is represented to
    # the product rather than replayed through it
    turns = (
        Turn(role="user", content="Question: how many?"),
        Turn(role="assistant", content="SELECT bad"),
        Turn(role="user", content="That failed: syntax error"),
    )

    rendered = render_prompt(turns)

    assert rendered.startswith("[user]\nQuestion: how many?")
    assert "[assistant]\nSELECT bad" in rendered
    assert rendered.endswith("[user]\nThat failed: syntax error")


def test_the_environment_report_says_which_instruction_files_are_in_scope() -> None:
    report = environment_report()

    # a number from this arm is only auditable if the run records these
    assert set(report) == {
        "user_memory_present",
        "user_memory_bytes",
        "project_memory_present",
    }
    assert isinstance(report["user_memory_present"], bool)


# -- driving the binary ------------------------------------------------------


async def test_a_completion_runs_the_cli_and_returns_its_answer(tmp_path: Path) -> None:
    executable = _stub(tmp_path, f"cat <<'EOF'\n{json.dumps(LIVE_DOCUMENT)}\nEOF")
    client = _client(executable)

    response = await client.complete(
        system="You write SQL.",
        turns=(Turn(role="user", content="how many?"),),
        model=MODEL,
        seed=0,
    )

    assert "SELECT COUNT(*)" in response.text
    assert response.tokens.input_tokens == 2
    assert response.tokens.output_tokens == 32
    # kept off TokenUsage: the product's context is not the arm's context
    assert client.scaffolding_tokens == 18_830


async def test_the_call_denies_every_tool_and_replaces_the_system_prompt(tmp_path: Path) -> None:
    executable = _stub(
        tmp_path,
        'echo "$@" > "$(dirname "$0")/argv.txt"\n'
        + f"cat <<'EOF'\n{json.dumps(LIVE_DOCUMENT)}\nEOF",
    )

    await _client(executable).complete(
        system="You write SQL.",
        turns=(Turn(role="user", content="how many?"),),
        model=MODEL,
        seed=3,
    )

    argv = (tmp_path / "argv.txt").read_text(encoding="utf-8")
    assert "--print" in argv
    assert "--output-format json" in argv
    assert "--system-prompt You write SQL." in argv
    assert DENIED_TOOLS in argv
    assert "--strict-mcp-config" in argv
    assert "--model sonnet" in argv


async def test_the_subscription_channel_is_not_allowed_to_fall_back_to_an_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a trace that named the subscription while a key answered would be wrong
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-be-visible")
    executable = _stub(
        tmp_path,
        'printf \'{"result":"%s","usage":{}}\' "${ANTHROPIC_API_KEY:-absent}"',
    )

    response = await _client(executable).complete(
        system="s", turns=(Turn(role="user", content="q"),), model=MODEL, seed=0
    )

    assert response.text == "absent"


async def test_a_non_zero_exit_is_a_failed_call(tmp_path: Path) -> None:
    executable = _stub(tmp_path, "echo 'credit balance too low' >&2\nexit 1")

    with pytest.raises(ModelError, match="exited 1: credit balance too low"):
        await _client(executable).complete(
            system="s", turns=(Turn(role="user", content="q"),), model=MODEL, seed=0
        )


async def test_a_silent_non_zero_exit_still_reports_something(tmp_path: Path) -> None:
    executable = _stub(tmp_path, "exit 3")

    with pytest.raises(ModelError, match="exited 3: no output"):
        await _client(executable).complete(
            system="s", turns=(Turn(role="user", content="q"),), model=MODEL, seed=0
        )


async def test_a_hang_is_a_hole_in_the_run_not_a_wrong_answer(tmp_path: Path) -> None:
    executable = _stub(tmp_path, "sleep 5")

    with pytest.raises(ModelError, match="did not answer within 1s"):
        await _client(executable, timeout_s=1).complete(
            system="s", turns=(Turn(role="user", content="q"),), model=MODEL, seed=0
        )


async def test_a_missing_binary_says_what_to_install(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="is not on PATH"):
        await _client(tmp_path / "does-not-exist").complete(
            system="s", turns=(Turn(role="user", content="q"),), model=MODEL, seed=0
        )


async def test_the_working_directory_is_honoured_so_project_memory_can_be_excluded(
    tmp_path: Path,
) -> None:
    executable = _stub(tmp_path, 'printf \'{"result":"%s","usage":{}}\' "$(pwd)"')
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    response = await _client(executable, cwd=scratch).complete(
        system="s", turns=(Turn(role="user", content="q"),), model=MODEL, seed=0
    )

    assert response.text.endswith(scratch.name)


def test_the_client_reports_the_provider_recorded_against_every_attempt() -> None:
    assert ClaudeCliClient().provider == "claude-cli"
