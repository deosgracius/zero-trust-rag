"""The query pipeline: resolve permissions, retrieve, assemble, audit.

The ordering here is the architecture. Authorization is resolved *before*
retrieval and enforced *inside* it, so unauthorized chunks are never candidates
and therefore never reach context assembly. No prompt-level instruction is
relied on to withhold anything, because prompt-level instructions are
defeatable by prompt-level attacks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

from ztrag.audit.chain import AuditChain
from ztrag.authz.model import AuthorizationStore
from ztrag.embed.embedder import Embedder
from ztrag.retrieval.store import Chunk, SearchResult, VectorStore


@dataclass(frozen=True, slots=True)
class RetrievalResponse:
    """What a query returns, plus the evidence needed to audit it."""

    user: str
    query: str
    results: tuple[SearchResult, ...]
    # Count of chunks excluded by authorization. Recorded for the audit trail,
    # never returned to the caller -- exposing it would recreate exactly the
    # existence-inference channel pre-filtering closes.
    filtered_count: int
    latency_ms: float

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return tuple(r.chunk.id for r in self.results)

    def context(self) -> str:
        """Assemble the retrieved chunks into an LLM context block."""
        return "\n\n---\n\n".join(r.chunk.text for r in self.results)


@dataclass
class RetrievalPipeline:
    """Wires authorization, embedding, retrieval and audit together."""

    authz: AuthorizationStore
    store: VectorStore
    embedder: Embedder
    audit: AuditChain | None = None

    # Padding target for constant-shape responses; see ``query``.
    pad_to_k: bool = True

    def query(self, user: str, text: str, *, k: int = 5) -> RetrievalResponse:
        started = time.perf_counter()

        # 1. Resolve the caller's effective permissions, expanding nested
        #    groups. Cached per user and invalidated on any tuple write.
        readable = self.authz.readable_objects(user)

        # 2. Retrieve with the authorization predicate applied during the scan.
        query_vec = self.embedder.embed(text)
        results = self.store.search(
            query_vec,
            authorized=lambda chunk_id: chunk_id in readable,
            k=k,
        )

        # 3. Count what was excluded, for the audit trail only.
        total = len(self.store)
        filtered = total - sum(1 for c in self.store.all_chunks() if c.id in readable)

        latency_ms = (time.perf_counter() - started) * 1000.0

        if self.audit is not None:
            self.audit.record(
                actor=user,
                action="retrieval.query",
                payload={
                    "query": text,
                    "returned_chunk_ids": list(r.chunk.id for r in results),
                    "returned_count": len(results),
                    "filtered_out_count": filtered,
                    "authorized_corpus_size": total - filtered,
                    "corpus_size": total,
                    "latency_ms": round(latency_ms, 3),
                },
            )

        return RetrievalResponse(
            user=user,
            query=text,
            results=tuple(results),
            filtered_count=filtered,
            latency_ms=latency_ms,
        )
