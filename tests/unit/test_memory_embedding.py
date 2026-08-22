"""The default embedder, and the properties the store depends on (SPEC §10.2).

Determinism is the load-bearing one. An embedding written today is compared
against a question embedded months later, so an embedder whose output moved
between processes — as anything built on :func:`hash` would, with its per-process
salt — would silently degrade every retrieval without any fingerprint noticing.
"""

from __future__ import annotations

import math
import subprocess
import sys

import pytest

from agentdb.core.memory import EMBEDDING_DIMENSIONS, Embedder, HashingEmbedder


def test_the_default_width_is_the_width_the_column_declares() -> None:
    assert HashingEmbedder().dimensions == EMBEDDING_DIMENSIONS
    assert len(HashingEmbedder().embed("revenue by nation")) == EMBEDDING_DIMENSIONS


def test_a_hashing_embedder_satisfies_the_protocol() -> None:
    assert isinstance(HashingEmbedder(), Embedder)


def test_a_width_no_vector_could_have_is_refused() -> None:
    with pytest.raises(ValueError, match="dimensions must be positive"):
        HashingEmbedder(dimensions=0)


def test_embeddings_are_unit_norm() -> None:
    vector = HashingEmbedder(dimensions=64).embed("total revenue per nation in 1994")

    assert math.isclose(math.sqrt(sum(value * value for value in vector)), 1.0)


def test_an_empty_question_embeds_to_zeros_rather_than_failing_a_write() -> None:
    assert set(HashingEmbedder(dimensions=8).embed("   ")) == {0.0}


def test_the_same_text_embeds_identically_in_a_separate_process() -> None:
    """The property :func:`hash` would break, and the store would never notice."""
    text = "orders per nation"
    here = HashingEmbedder(dimensions=32).embed(text)

    there = subprocess.run(
        [
            sys.executable,
            "-c",
            "from agentdb.core.memory import HashingEmbedder;"
            f"print(list(HashingEmbedder(dimensions=32).embed({text!r})))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert [float(value) for value in eval(there.stdout)] == list(here)


def test_word_order_is_not_invisible_to_the_bigram_features() -> None:
    embedder = HashingEmbedder(dimensions=256)

    forward = embedder.embed("revenue by customer")
    reversed_ = embedder.embed("customer by revenue")

    assert forward != reversed_


def test_a_shared_vocabulary_scores_closer_than_a_disjoint_one() -> None:
    embedder = HashingEmbedder(dimensions=512)

    def similarity(left: str, right: str) -> float:
        return sum(a * b for a, b in zip(embedder.embed(left), embedder.embed(right), strict=True))

    near = similarity("orders per nation", "orders per nation in 1994")
    far = similarity("orders per nation", "top browsers by unique visitor")

    assert near > far
