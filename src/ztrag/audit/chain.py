"""Hash-chained audit log.

Same construction as the LLM gateway's Go implementation (project 01,
ADR 0003), reimplemented here rather than shared because the two systems ship
independently and a shared library across language boundaries would be more
coupling than the ~120 lines are worth.

For this project the audit trail carries a specific burden: it is the evidence
that no unauthorized retrieval happened. "We filter correctly" is a claim;
"here is a tamper-evident record of every chunk returned to every user, and
the decision basis for each" is a demonstration.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

GENESIS_HASH = "0" * 64


@dataclass(frozen=True, slots=True)
class Entry:
    seq: int
    timestamp: str
    actor: str
    action: str
    payload: dict[str, Any]
    prev_hash: str
    hash: str


def _canonical(
    seq: int, timestamp: str, actor: str, action: str, payload: dict[str, Any], prev_hash: str
) -> bytes:
    """Length-prefixed encoding of the hashed fields.

    Length-prefixing makes the encoding injective: no two distinct field sets
    can produce the same bytes by shifting a delimiter into a value. JSON with
    sorted keys is used for the payload specifically because Python dict
    ordering is insertion-ordered, and an audit hash must not depend on the
    order in which a caller happened to build a dict.
    """
    parts = [
        str(seq),
        timestamp,
        actor,
        action,
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str),
        prev_hash,
    ]
    return b"".join(f"{len(p)}:".encode() + p.encode("utf-8") for p in parts)


def compute_hash(
    seq: int, timestamp: str, actor: str, action: str, payload: dict[str, Any], prev_hash: str
) -> str:
    return hashlib.sha256(_canonical(seq, timestamp, actor, action, payload, prev_hash)).hexdigest()


class VerificationError(Exception):
    """Raised by ``verify`` when the chain does not hold."""


@dataclass
class AuditChain:
    """Append-only, hash-chained audit log held in memory."""

    entries: list[Entry] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, actor: str, action: str, payload: dict[str, Any]) -> Entry:
        with self._lock:
            prev = self.entries[-1].hash if self.entries else GENESIS_HASH
            seq = len(self.entries) + 1
            ts = datetime.now(timezone.utc).isoformat()
            entry = Entry(
                seq=seq,
                timestamp=ts,
                actor=actor,
                action=action,
                payload=payload,
                prev_hash=prev,
                hash=compute_hash(seq, ts, actor, action, payload, prev),
            )
            self.entries.append(entry)
            return entry

    def verify(self) -> None:
        """Re-walk the chain, raising ``VerificationError`` on any break.

        Checks sequence contiguity, predecessor linkage and content hashes --
        all three, because any one alone misses an attack the others catch.
        See project 01 ADR 0003 for the reasoning and for the documented
        limitation: tail truncation is not detectable by the chain alone.
        """
        prev_hash = GENESIS_HASH
        for i, e in enumerate(self.entries, start=1):
            if e.seq != i:
                raise VerificationError(f"sequence gap: expected {i}, got {e.seq}")
            if e.prev_hash != prev_hash:
                raise VerificationError(f"prev_hash mismatch at seq {e.seq}")
            recomputed = compute_hash(e.seq, e.timestamp, e.actor, e.action, e.payload, e.prev_hash)
            if recomputed != e.hash:
                raise VerificationError(f"content hash mismatch at seq {e.seq}")
            prev_hash = e.hash

    def head(self) -> tuple[str, int]:
        with self._lock:
            if not self.entries:
                return GENESIS_HASH, 0
            return self.entries[-1].hash, self.entries[-1].seq

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(asdict(e), sort_keys=True, default=str) for e in self.entries)
