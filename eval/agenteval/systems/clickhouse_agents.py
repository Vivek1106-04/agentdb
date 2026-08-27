"""ClickHouse Agents, through its Cloud conversation endpoint (SPEC §11.5.1).

The client half of the ``S3_clickhouse_agents`` arm; how the row is scored lives
in :mod:`agenteval.systems.managed`.

**Why the reply is read by configured path rather than by hardcoded field
names.** ClickHouse Agents is a public beta, and §11.5.1 turns on measuring a
named build honestly: an adapter that guessed at field names would either break
silently against the build actually measured, or — worse — read prose where it
meant to read SQL and score the arm at zero. So the endpoint, the request field,
and the two response paths come from the arm's committed config, which means the
trace records exactly what was asked for and exactly what was read out. When the
beta's payload shape changes, the config changes and the diff is visible in the
run that used it.

Transport is stdlib ``urllib`` on a worker thread. The harness already refuses
to make a vendor SDK a hard dependency, and one JSON POST does not justify a new
one.
"""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from base64 import b64encode
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from agenteval.systems.managed import ManagedAnswer, ManagedConfigError, ManagedError

DEFAULT_PATH = "/v1/agents/{target_id}/conversations"
"""Where the question is posted, relative to the host. Overridable per arm."""

DEFAULT_QUESTION_FIELD = "message"
DEFAULT_SQL_PATH = "sql"
DEFAULT_TEXT_PATH = "text"
DEFAULT_TIMEOUT_S = 120

RESPONSE_KEYS = frozenset({"path", "question_field", "sql_path", "text_path"})
"""What an ``S3`` arm may put in its config's ``response`` mapping."""

Transport = Callable[[str, Mapping[str, Any], Mapping[str, str]], Awaitable[Mapping[str, Any]]]


class ClickHouseAgentsConfigError(ManagedConfigError):
    """The arm names a reply shape this client cannot read."""


@dataclass(frozen=True, slots=True)
class ClickHouseAgentsTarget:
    """Which ClickHouse Cloud endpoint to ask, and as whom.

    No credential has a default, for the same reason the Databricks target has
    none: a benchmark that silently reached an organisation the operator did not
    choose would be worse than one that refused to start.
    """

    host: str
    key_id: str = ""
    key_secret: str = ""

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ClickHouseAgentsTarget:
        """Read ``AGENTEVAL_CH_AGENTS_*``. Credentials come from the environment only."""
        source = os.environ if env is None else env
        host = source.get("AGENTEVAL_CH_AGENTS_HOST", "").strip().rstrip("/")
        if not host:
            raise ClickHouseAgentsConfigError(
                "AGENTEVAL_CH_AGENTS_HOST is unset; export the ClickHouse Cloud host, "
                "plus AGENTEVAL_CH_AGENTS_KEY_ID and AGENTEVAL_CH_AGENTS_KEY_SECRET"
            )
        return cls(
            host=host,
            key_id=source.get("AGENTEVAL_CH_AGENTS_KEY_ID", ""),
            key_secret=source.get("AGENTEVAL_CH_AGENTS_KEY_SECRET", ""),
        )

    @property
    def headers(self) -> dict[str, str]:
        """Key-pair auth, as ClickHouse Cloud's API takes it."""
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.key_id or self.key_secret:
            pair = b64encode(f"{self.key_id}:{self.key_secret}".encode()).decode("ascii")
            headers["Authorization"] = f"Basic {pair}"
        return headers


@dataclass(frozen=True, slots=True)
class ClickHouseAgentsConversation:
    """One ClickHouse Agents deployment's ask-a-question surface."""

    target: ClickHouseAgentsTarget
    transport: Transport
    path: str = DEFAULT_PATH
    question_field: str = DEFAULT_QUESTION_FIELD
    sql_path: str = DEFAULT_SQL_PATH
    text_path: str = DEFAULT_TEXT_PATH

    async def ask(self, target_id: str, question: str) -> ManagedAnswer:
        """Put one question to agent ``target_id`` and read its reply."""
        url = f"{self.target.host}{self.path.format(target_id=target_id)}"
        try:
            payload = await self.transport(
                url, {self.question_field: question}, self.target.headers
            )
        except ManagedError:
            raise
        except Exception as exc:
            raise ManagedError(
                f"ClickHouse Agents agent {target_id!r} did not answer: {type(exc).__name__}: {exc}"
            ) from exc
        return read_payload(payload, sql_path=self.sql_path, text_path=self.text_path)


def build_conversation(
    response: Mapping[str, str],
    *,
    target: ClickHouseAgentsTarget,
    transport: Transport | None = None,
) -> ClickHouseAgentsConversation:
    """Build the client from an arm's committed ``response`` mapping."""
    unknown = sorted(set(response) - RESPONSE_KEYS)
    if unknown:
        raise ClickHouseAgentsConfigError(
            f"a ClickHouse Agents arm's response config has unknown key(s): {', '.join(unknown)}; "
            f"expected some of {sorted(RESPONSE_KEYS)}"
        )
    return ClickHouseAgentsConversation(
        target=target,
        transport=transport or post_json,
        path=response.get("path", DEFAULT_PATH),
        question_field=response.get("question_field", DEFAULT_QUESTION_FIELD),
        sql_path=response.get("sql_path", DEFAULT_SQL_PATH),
        text_path=response.get("text_path", DEFAULT_TEXT_PATH),
    )


def read_payload(payload: Mapping[str, Any], *, sql_path: str, text_path: str) -> ManagedAnswer:
    """Pull the SQL and the prose out of one reply.

    A reply with nothing at ``sql_path`` is a decline: the service answered
    without writing a query, which is measured rather than treated as an error.
    """
    sql = _text_at(payload, sql_path)
    return ManagedAnswer(sql=sql or None, text=_text_at(payload, text_path) or "")


def _text_at(payload: Mapping[str, Any], path: str) -> str:
    """Follow a dotted path — ``attachments.0.sql`` — to a string, or ``''``."""
    current: Any = payload
    for step in path.split("."):
        if isinstance(current, Mapping):
            current = current.get(step)
        elif isinstance(current, Sequence) and not isinstance(current, str) and step.isdigit():
            index = int(step)
            current = current[index] if index < len(current) else None
        else:
            return ""
        if current is None:
            return ""
    return str(current).strip()


async def post_json(
    url: str, body: Mapping[str, Any], headers: Mapping[str, str]
) -> Mapping[str, Any]:
    """One JSON POST, off the event loop.

    ``urllib`` is blocking, and a benchmark that stalled its loop for the length
    of an agent's turn would make every concurrent timing in the run a lie.
    """
    return await asyncio.to_thread(_post_json_blocking, url, body, headers)


def _post_json_blocking(
    url: str, body: Mapping[str, Any], headers: Mapping[str, str]
) -> Mapping[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(dict(body)).encode("utf-8"),
        headers=dict(headers),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=DEFAULT_TIMEOUT_S) as response:
            decoded = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise ManagedError(f"{url} answered {exc.code}: {exc.reason}") from exc
    except (OSError, ValueError) as exc:
        raise ManagedError(f"{url} could not be read: {type(exc).__name__}: {exc}") from exc

    if not isinstance(decoded, Mapping):
        raise ManagedError(f"{url} answered with {type(decoded).__name__}, not a JSON object")
    return decoded
