"""Vector store with authorization applied *during* the search.

This module exists to make one distinction concrete, because it is the
difference between a RAG demo and a system an enterprise can deploy:

**Post-filtering** retrieves the global top-k, then removes chunks the caller
may not see. It is both slow and leaky:

  - *Leaky.* If the caller asks for k=10 and receives 4, they have learned that
    6 documents exist which match their query and which they cannot see. Result
    counts are a side channel; so are the gaps in a ranked list.
  - *Slow and wrong.* A user with access to 1% of the corpus gets almost
    nothing back, because the global top-k is dominated by documents they
    cannot read. Recovering their true top-k requires retrieving a far larger
    candidate set, with no principled bound on how much larger.

**Pre-filtering** applies the authorization predicate as part of the scan, so
the candidate set only ever contains chunks the caller may read. The caller
gets a true top-k over their own visible corpus, and result counts reveal
nothing about documents outside it.

Production systems express this as a payload filter (Qdrant), a `WHERE` clause
alongside the ANN operator (pgvector), or a filtered index. This implementation
is an exact brute-force scan, which is the right choice here for two reasons:
the eval corpus is small, and an exact scan has no recall degradation to
confound the authorization measurements. Swapping in a real ANN index is a
change of `search` only — and it introduces a recall/filter interaction that
ADR 0002 covers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable unit of text carrying the ACL of its source document."""

    id: str          # "doc:design-spec#3"
    doc_id: str      # "doc:design-spec"
    text: str
    embedding: tuple[float, ...]
    # Denormalised for display/audit only. Authorization is resolved through
    # the AuthorizationStore, never from this field -- a copy of an ACL is a
    # second source of truth, and it will drift.
    source: str = ""


@dataclass(frozen=True, slots=True)
class SearchResult:
    chunk: Chunk
    score: float


@dataclass
class VectorStore:
    """An exact-scan vector store with a mandatory authorization predicate."""

    _chunks: list[Chunk] = field(default_factory=list)
    _by_id: dict[str, Chunk] = field(default_factory=dict)

    def add(self, *chunks: Chunk) -> None:
        for c in chunks:
            if c.id in self._by_id:
                raise ValueError(f"duplicate chunk id {c.id!r}")
            self._by_id[c.id] = c
            self._chunks.append(c)

    def __len__(self) -> int:
        return len(self._chunks)

    def get(self, chunk_id: str) -> Chunk | None:
        return self._by_id.get(chunk_id)

    def all_chunks(self) -> Sequence[Chunk]:
        return tuple(self._chunks)

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        authorized: Callable[[str], bool],
        k: int = 5,
    ) -> list[SearchResult]:
        """Return the top ``k`` chunks the caller is authorized to read.

        ``authorized`` is applied *before* scoring, not after ranking. The
        parameter is required rather than optional and has no default: an
        unauthenticated search is not a meaningful operation in this system,
        and a default of "allow everything" is the kind of convenience that
        becomes a production incident.
        """
        scored: list[SearchResult] = []
        for chunk in self._chunks:
            # The authorization gate. Everything downstream -- scoring,
            # ranking, reranking, context assembly -- only ever sees chunks
            # that passed here.
            if not authorized(chunk.id):
                continue
            scored.append(SearchResult(chunk, cosine_similarity(query_embedding, chunk.embedding)))

        scored.sort(key=lambda r: (-r.score, r.chunk.id))
        return scored[:k]

    def search_postfilter_UNSAFE(
        self,
        query_embedding: Sequence[float],
        *,
        authorized: Callable[[str], bool],
        k: int = 5,
    ) -> list[SearchResult]:
        """Post-filtering implementation, kept only to demonstrate the leak.

        Retrieves the global top-k and *then* removes unauthorized chunks. It
        never returns content the caller may not read, so a naive test suite
        calls it secure. The eval suite uses it to show the existence-inference
        channel it opens: the number of results, and which ranks are missing,
        both vary with the content of documents the caller cannot see.

        Never call this from production code paths.
        """
        scored = [
            SearchResult(c, cosine_similarity(query_embedding, c.embedding))
            for c in self._chunks
        ]
        scored.sort(key=lambda r: (-r.score, r.chunk.id))
        top_k = scored[:k]
        return [r for r in top_k if authorized(r.chunk.id)]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two vectors.

    Returns 0.0 for a zero-magnitude vector rather than raising: an empty
    chunk should rank last, not crash a query.
    """
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
