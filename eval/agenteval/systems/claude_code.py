"""``S5_claude_code`` — Claude Code as a system under test (SPEC §11.5).

Family S measures agent stacks this project did not write: MCP servers,
ClickHouse Agents, Databricks AI/BI Genie. Claude Code belongs in that list. It
is a shipped, Claude-powered agent product, and it is the one Anthropic-model
system a reader can reproduce with a subscription instead of API credit.

**Why this is not A0 with a different client.** The A0 arm means "a model, a
schema dump, and nothing else", and this channel cannot deliver that: measured
on the authoring machine, every call carries 16k-30k tokens of the product's own
scaffolding and whatever instruction files the operator's ``~/.claude`` supplies,
and the flag that would strip them also disables subscription auth. Scoring that
as A0 would put a number in the ablation floor that was produced by a different
system. So it is a Family S row, named after the product, with the contamination
recorded on every attempt rather than argued about later.

The task payload is identical to A0's — same prompt, same schema dump, same
retry loop — because the comparison only means something if the question is the
same. What differs is the channel, and the channel is in the name.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from agenteval.execution import QueryExecutor
from agenteval.models.base import ModelClient, ModelError
from agenteval.models.claude_cli import PROVIDER as CLI_PROVIDER
from agenteval.models.claude_cli import environment_report
from agenteval.systems.base import Attempt, ModelSpec
from agenteval.systems.fingerprint import config_fingerprint
from agenteval.systems.loop import answer_with_model
from agenteval.systems.raw_schema import (
    DEFAULT_MAX_RETRIES,
    build_question_turn,
    build_system_prompt,
)
from agenteval.tasks import Task

ARM_NAME = "S5_claude_code"

SCAFFOLDING_NOTE = "claude_code_scaffolding_tokens"
"""Note key carrying the product context this arm could not remove."""

VERSION = "1.0"
"""Bumped whenever the prompt or the loop changes, because both change the number."""


@dataclass(frozen=True, slots=True)
class ClaudeCodeSystem:
    """Claude Code, tools denied, answering the A0 payload through its own CLI."""

    executor: QueryExecutor
    client: ModelClient
    config_fingerprint: str
    max_retries: int = DEFAULT_MAX_RETRIES
    name: str = ARM_NAME
    version: str = VERSION
    controls_model: bool = True

    @classmethod
    def create(
        cls,
        *,
        executor: QueryExecutor,
        client: ModelClient,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> ClaudeCodeSystem:
        """Build the arm, refusing a client that is not the product this arm names.

        A run that pointed this arm at the API would publish an ``S5`` row for a
        bare model — the exact confusion the arm exists to prevent.
        """
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")
        if client.provider != CLI_PROVIDER:
            raise ModelError(
                f"{ARM_NAME} measures the Claude Code product and needs the "
                f"{CLI_PROVIDER!r} channel; got {client.provider!r}"
            )

        return cls(
            executor=executor,
            client=client,
            max_retries=max_retries,
            config_fingerprint=config_fingerprint(
                {
                    "arm": ARM_NAME,
                    "version": VERSION,
                    "engine": executor.engine,
                    "max_retries": max_retries,
                    "provider": client.provider,
                    "system_prompt": build_system_prompt(executor.engine),
                    # The operator's instruction files are part of what produced
                    # the number, so they are part of what identifies the config.
                    "environment": environment_report(),
                }
            ),
        )

    async def answer(self, task: Task, model: ModelSpec | None, seed: int) -> Attempt:
        """Answer ``task`` once, through the product rather than through the API."""
        if model is None:
            raise ModelError(f"{ARM_NAME} chooses no model of its own; pass a ModelSpec")

        schema = await self.executor.schema_text(task.namespace)
        attempt = await answer_with_model(
            system_name=self.name,
            task=task,
            model=model,
            seed=seed,
            executor=self.executor,
            client=self.client,
            system_prompt=build_system_prompt(self.executor.engine),
            first_turn=build_question_turn(task, schema),
            context_bytes=len(schema.encode("utf-8")),
            max_retries=self.max_retries,
        )
        return replace(attempt, notes=(*attempt.notes, self._scaffolding_note()))

    def _scaffolding_note(self) -> str:
        """What the product added to the prompt, recorded on the attempt.

        The arm's own context is the schema dump in ``context_bytes``. This is
        everything else the product sent — tool definitions and the operator's
        instruction files — and it dwarfs the arm's context. A row that reported
        only the arm's tokens would read as though this system had been given
        far less than it was.
        """
        seen = getattr(self.client, "scaffolding_tokens", 0)
        return f"{SCAFFOLDING_NOTE}={seen}"
