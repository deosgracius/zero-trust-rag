"""Embedding backends.

The differentiator of this project is authorization correctness, not embedding
quality. So the default backend is a deterministic, dependency-free hashing
embedder: it makes the whole authorization eval runnable and reproducible with
no model download, no GPU and no API key.

This is a real tradeoff, stated plainly rather than hidden: hashing embeddings
have no semantic structure, so *retrieval quality* numbers produced against
them are meaningless. They are used only for the authorization suite, where
what matters is which chunks are reachable, not how well they rank. Recall@k
comparisons must be run against a real model — see ``SentenceTransformerEmbedder``.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, Sequence

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    """Anything that turns text into a fixed-length vector."""

    @property
    def dimensions(self) -> int: ...

    def embed(self, text: str) -> tuple[float, ...]: ...


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder.

    Each token is hashed to a bucket and accumulated with a sign derived from
    the same hash, which keeps unrelated tokens from systematically reinforcing
    each other. The result is L2-normalised so cosine similarity is a plain dot
    product.

    Properties that make it useful for testing: identical text always produces
    an identical vector, texts sharing vocabulary score higher than texts that
    do not, and no state is carried between calls.
    """

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, text: str) -> tuple[float, ...]:
        vec = [0.0] * self._dimensions
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            bucket = value % self._dimensions
            sign = 1.0 if (value >> 63) & 1 else -1.0
            vec[bucket] += sign

        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return tuple(vec)
        return tuple(v / norm for v in vec)


class SentenceTransformerEmbedder:
    """Real embeddings via sentence-transformers.

    Not a dependency of the test suite. Import is deferred to the constructor
    so the package remains installable, and the authorization eval remains
    runnable, without torch present.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "SentenceTransformerEmbedder requires the 'real-embeddings' extra: "
                "pip install -e '.[real-embeddings]'"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self._dimensions = int(self._model.get_sentence_embedding_dimension())

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, text: str) -> tuple[float, ...]:  # pragma: no cover - optional
        vec = self._model.encode(text, normalize_embeddings=True)
        return tuple(float(x) for x in vec)
