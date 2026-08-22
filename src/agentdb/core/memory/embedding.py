"""How a question becomes a vector (SPEC §10.2, §10.4).

The store's semantic term needs an embedding, and the project's reproduction
promise — clone, ``docker compose up``, ``make bench-quick``, numbers in ten
minutes — will not survive a required embeddings API key. So the default
embedder is deterministic, local, and offline: feature hashing over word
unigrams and bigrams into the 1536 dimensions the ``VECTOR(1536)`` column
declares.

**Call it what it is.** This is a lexical baseline, not a semantic model. It
scores "orders per nation" close to "orders by nation" and nowhere near
"customers by region", which is a real signal and a weak one. That is why
:class:`Embedder` is a protocol: a reader with an embeddings budget swaps in a
model, re-runs, and the ``w_sem`` ablation row says what the upgrade bought.
Publishing the weak default alongside the arm that measures it is honest;
hiding a lexical trick behind the word "semantic" would not be.

Hashing is ``blake2b``, never :func:`hash`, whose per-process salt would make
today's embeddings unrankable against yesterday's stored ones.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Sequence
from itertools import pairwise
from typing import Protocol, runtime_checkable

EMBEDDING_DIMENSIONS = 1536
"""Width of the ``agentdb_exemplar.embedding`` column (SPEC §10.2).

Not tunable: it is a DDL fact. A different embedder must project to this width
or the schema migrates with it.
"""

_TOKEN = re.compile(r"[a-z0-9_]+")


@runtime_checkable
class Embedder(Protocol):
    """Turns text into a fixed-width vector.

    Implementations must be deterministic across processes: an embedding stored
    today is compared against a question embedded months later, and a model that
    silently changes its output invalidates the whole store without any schema
    fingerprint noticing.
    """

    @property
    def dimensions(self) -> int:
        """Vector width. Must equal :data:`EMBEDDING_DIMENSIONS` to be storable."""

    def embed(self, text: str) -> tuple[float, ...]:
        """Return the unit-norm embedding of ``text``."""


class HashingEmbedder:
    """Deterministic offline feature-hashing embedder — the default.

    Signed hashing keeps the expected collision contribution at zero rather than
    additive, which is what makes a 1536-wide bag of hashed features usable at
    all: unsigned buckets would inflate similarity between any two long texts.
    """

    __slots__ = ("_dimensions",)

    def __init__(self, dimensions: int = EMBEDDING_DIMENSIONS) -> None:
        if dimensions <= 0:
            raise ValueError(f"dimensions must be positive, got {dimensions}")
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, text: str) -> tuple[float, ...]:
        """Hash ``text`` into a unit-norm vector; empty text yields all zeros.

        An all-zero vector is a deliberate outcome rather than an error: the
        ranking's cosine term treats it as zero similarity, so an exemplar with
        an empty question falls back on its other four terms instead of failing
        a write.
        """
        vector = [0.0] * self._dimensions
        for feature in _features(text):
            index, sign = _bucket(feature, self._dimensions)
            vector[index] += sign
        return _unit(vector)


def _features(text: str) -> Iterable[str]:
    """Word unigrams plus adjacent bigrams.

    Bigrams are what let the vector distinguish "revenue by customer" from
    "customer by revenue" at all. Without them the representation is a pure bag
    of words, and two questions built from the same vocabulary are identical.
    """
    tokens = _TOKEN.findall(text.casefold())
    yield from tokens
    yield from (f"{left}_{right}" for left, right in pairwise(tokens))


def _bucket(feature: str, dimensions: int) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    index = int.from_bytes(digest[:4], "big") % dimensions
    sign = 1.0 if digest[4] & 1 else -1.0
    return index, sign


def _unit(vector: Sequence[float]) -> tuple[float, ...]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return tuple(vector)
    return tuple(value / norm for value in vector)
