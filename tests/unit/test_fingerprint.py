"""A fingerprint that is not stable across machines is not evidence."""

from __future__ import annotations

from agenteval.systems.fingerprint import config_fingerprint


def test_fingerprint_is_prefixed_and_hex() -> None:
    digest = config_fingerprint({"arm": "A0_baseline"})

    assert digest.startswith("sha256:")
    assert len(digest) == len("sha256:") + 64


def test_key_order_does_not_change_the_fingerprint() -> None:
    assert config_fingerprint({"a": 1, "b": 2}) == config_fingerprint({"b": 2, "a": 1})


def test_any_config_change_changes_the_fingerprint() -> None:
    assert config_fingerprint({"max_retries": 2}) != config_fingerprint({"max_retries": 3})


def test_unserializable_values_are_stringified_rather_than_crashing() -> None:
    # Arrange — configs carry objects (paths, enums); a fingerprint must never
    # be the thing that fails a run
    class Opaque:
        def __str__(self) -> str:
            return "opaque"

    assert config_fingerprint({"client": Opaque()}) == config_fingerprint({"client": "opaque"})
