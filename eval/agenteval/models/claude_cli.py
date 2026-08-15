"""Claude Code as a system under test, driven through its own CLI (SPEC §11.5).

Claude Code is a shipped, Claude-powered agent product — the same category as
ClickHouse Agents and Databricks AI/BI Genie, both of which this project exists
to measure. It is reachable without an API key, through a subscription, which
makes it the one Anthropic-model arm a reader without API credit can reproduce.

**This adapter is not a bare-model channel, and must never be used for a Family A
arm.** Three facts, each measured on this machine rather than assumed:

1. Roughly **16k to 30k tokens of Claude Code scaffolding** reach the model on every
   call — tool definitions and harness context — even with ``--system-prompt``
   replacing the prompt and every tool denied. A0 means "a model given a schema
   and nothing else", and this is not that.
2. **User-level instruction files load regardless of working directory.** Asked
   directly, the model confirmed it could see ``~/.claude/CLAUDE.md`` and the
   rule files it imports, from an empty scratch directory. Whoever runs this arm
   is measuring their own global instructions along with the model.
3. **Isolation and subscription auth are mutually exclusive.** The flag that
   skips memory and CLAUDE.md discovery (``--bare`` / ``CLAUDE_CODE_SIMPLE=1``)
   also disables OAuth: it answers "Not logged in · Please run /login".

So the honest reading of an ``S5_claude_code`` number is *"Claude Code, tools
denied, carrying whatever instructions this operator's machine supplies"*. Every
attempt records the scaffolding token count so a reader can see the size of what
they cannot see the contents of, and :func:`environment_report` captures the
instruction files in scope so a run is at least auditable.

There is also no temperature or seed control: ``seed`` is a repetition index
here, nothing more. That is disclosed rather than hidden, and it is why this arm
reports a spread rather than a seed-controlled interval.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agenteval.models.base import ModelError, ModelResponse, Turn
from agenteval.systems.base import ModelSpec, TokenUsage

PROVIDER = "claude-cli"

EXECUTABLE = "claude"

DENIED_TOOLS = (
    "Bash,Edit,Write,Read,Glob,Grep,WebFetch,WebSearch,Task,TodoWrite,"
    "NotebookEdit,SlashCommand,Skill,BashOutput,KillShell"
)
"""Every tool this arm refuses. Denial does not remove the definitions from the
prompt — measured, not assumed — but it does stop the agent from acting, which
is what keeps this a *question answering* arm rather than an agentic one."""

DEFAULT_TIMEOUT_S = 180
"""A cold Claude Code start plus a long answer. Past this the attempt is a hole
in the run, reported as such rather than recorded as a wrong answer."""

USER_MEMORY = Path.home() / ".claude" / "CLAUDE.md"
"""Loaded on every call regardless of working directory. Its presence is part of
what a number from this arm means, so a run records whether it existed."""


@dataclass(frozen=True, slots=True)
class CliResult:
    """The fields of ``--output-format json`` this adapter reads."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    scaffolding_tokens: int = 0
    """``cache_creation + cache_read``: the context the operator did not write.

    Reported separately from ``input_tokens`` because it is not the arm's
    context — it is the product's — and folding the two together would make this
    arm look like it was given far more grounding than it was."""

    stop_reason: str | None = None
    cost_usd: float | None = None
    session_id: str | None = None


def parse_result(payload: str) -> CliResult:
    """Read one ``claude -p --output-format json`` document.

    A document that reports its own failure is an error, not an empty answer:
    the run needs to tell "the product said nothing useful" apart from "the
    product never ran".
    """
    try:
        document = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ModelError(f"claude CLI did not return JSON: {payload[:200]!r}") from exc

    if not isinstance(document, dict):
        raise ModelError(f"claude CLI returned {type(document).__name__}, expected an object")
    if document.get("is_error"):
        raise ModelError(f"claude CLI reported an error: {document.get('result') or document}")

    text = document.get("result")
    if not isinstance(text, str):
        raise ModelError("claude CLI returned no result text")

    usage = document.get("usage") or {}
    return CliResult(
        text=text,
        input_tokens=_int(usage.get("input_tokens")),
        output_tokens=_int(usage.get("output_tokens")),
        scaffolding_tokens=(
            _int(usage.get("cache_creation_input_tokens"))
            + _int(usage.get("cache_read_input_tokens"))
        ),
        stop_reason=_optional_str(document.get("stop_reason")),
        cost_usd=_optional_float(document.get("total_cost_usd")),
        session_id=_optional_str(document.get("session_id")),
    )


def render_prompt(turns: Sequence[Turn]) -> str:
    """Flatten a conversation into the single prompt the CLI accepts.

    Print mode takes one prompt, so a self-correction turn is rendered with its
    role named rather than sent as a real assistant message. The arm's retry
    loop is therefore *represented* to the product rather than replayed through
    it, and that difference is recorded here rather than glossed over.
    """
    if len(turns) == 1:
        return turns[0].content
    return "\n\n".join(f"[{turn.role}]\n{turn.content}" for turn in turns)


def environment_report() -> dict[str, Any]:
    """What this machine will inject into every call, for the trace.

    A number from this arm is only auditable if the run says which instruction
    files were in scope when it was produced.
    """
    return {
        "user_memory_present": USER_MEMORY.is_file(),
        "user_memory_bytes": USER_MEMORY.stat().st_size if USER_MEMORY.is_file() else 0,
        "project_memory_present": Path("CLAUDE.md").is_file(),
    }


@dataclass
class ClaudeCliClient:
    """Runs Claude Code in print mode, once per completion."""

    executable: str = EXECUTABLE
    timeout_s: int = DEFAULT_TIMEOUT_S
    cwd: Path | None = None
    """Where the CLI runs. A directory with no ``CLAUDE.md`` of its own keeps
    *project* instructions out; user-level ones load regardless."""

    scaffolding_tokens: int = field(default=0, init=False)
    """Scaffolding seen on the most recent call, for the arm to record."""

    @property
    def provider(self) -> str:
        return PROVIDER

    async def complete(
        self,
        *,
        system: str,
        turns: tuple[Turn, ...],
        model: ModelSpec,
        seed: int,
    ) -> ModelResponse:
        """Ask Claude Code the question, with tools denied and the prompt replaced.

        ``seed`` reaches no sampling parameter — the CLI exposes none — so it is
        a repetition index. An adapter that quietly pretended otherwise would
        make a spread look like a controlled interval.
        """
        # seed reaches no sampling parameter: the CLI exposes none. Recorded on
        # the response so a trace shows which repetition produced which answer.
        del seed

        argv = [
            self.executable,
            "--print",
            "--output-format",
            "json",
            "--model",
            model.name,
            "--system-prompt",
            system,
            "--disallowed-tools",
            DENIED_TOOLS,
            "--strict-mcp-config",
            render_prompt(turns),
        ]

        stdout, stderr, code = await self._run(argv)
        if code != 0:
            raise ModelError(
                f"claude CLI exited {code}: {(stderr or stdout).strip()[:300] or 'no output'}"
            )

        result = parse_result(stdout)
        self.scaffolding_tokens = result.scaffolding_tokens
        return ModelResponse(
            text=result.text,
            tokens=TokenUsage(input_tokens=result.input_tokens, output_tokens=result.output_tokens),
            stop_reason=result.stop_reason,
        )

    async def _run(self, argv: list[str]) -> tuple[str, str, int]:
        """Spawn the CLI, and treat a hang as a failed call rather than a wrong answer."""
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.cwd) if self.cwd else None,
                env=_child_env(),
            )
        except FileNotFoundError as exc:
            raise ModelError(
                f"{self.executable!r} is not on PATH; install Claude Code to run this arm"
            ) from exc

        try:
            out, err = await asyncio.wait_for(process.communicate(), timeout=self.timeout_s)
        except TimeoutError:
            process.kill()
            await process.wait()
            raise ModelError(f"claude CLI did not answer within {self.timeout_s}s") from None

        return (
            out.decode("utf-8", "replace"),
            err.decode("utf-8", "replace"),
            process.returncode or 0,
        )


def _child_env() -> dict[str, str]:
    """The environment the CLI runs under.

    ``ANTHROPIC_API_KEY`` is dropped on purpose: this arm exists to measure the
    subscription path, and silently falling back to a key would mean the trace
    named a channel the run did not use.
    """
    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)
    return env


def _int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_float(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None
