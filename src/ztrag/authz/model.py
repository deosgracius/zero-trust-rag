"""Relationship-based authorization, Zanzibar-style.

The central claim of this project is that an unauthorized chunk never enters
the model's context window. That claim is only as good as the permission
resolution behind it, so this module is deliberately the most carefully tested
part of the codebase.

The model follows Google Zanzibar (and its open-source descendant OpenFGA):
authorization is expressed as *relationship tuples* rather than as a permission
matrix. A tuple says "user U has relation R on object O", and permissions are
derived by walking those relationships transitively:

    user:alice          member    group:engineering
    group:engineering   viewer    doc:design-spec
    doc:design-spec#3   parent    doc:design-spec

Chunk ids are "<doc_id>#<n>". Alice can view chunk 3 because the
relationships compose. Nothing enumerates "alice can read chunk 3" directly, which is what makes the model survive
nested groups, org restructures and inherited folder permissions.

This is an in-process implementation, not a Zanzibar clone. It has no
distributed consistency story (no zookies, no snapshot reads). What it does
have is the resolution semantics the authorization eval depends on, so the
correctness properties can be tested without standing up a service.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Iterator


@dataclass(frozen=True, slots=True)
class Tuple_:
    """One relationship: ``subject`` has ``relation`` on ``obj``.

    Frozen so tuples can live in sets, which the expansion walk relies on for
    cycle detection.
    """

    subject: str   # "user:alice" or "group:engineering"
    relation: str  # "member", "viewer", "editor", "owner", "parent"
    obj: str       # "group:engineering", "doc:design-spec"

    def __str__(self) -> str:
        return f"{self.subject} {self.relation} {self.obj}"


class CycleError(Exception):
    """Raised when the relationship graph contains a cycle.

    Groups containing each other is a real thing that happens in directory
    systems after enough reorganisations. Left unhandled it is an infinite
    loop in the permission resolver, i.e. a denial of service reachable by
    anyone who can create a group.
    """


# Relations that grant read access to a document, in increasing order of power.
# "viewer" is the weakest; an owner can also read. Keeping this explicit rather
# than implicit means adding a relation later forces a decision about whether
# it implies read.
READ_RELATIONS: frozenset[str] = frozenset({"viewer", "editor", "owner"})


@dataclass
class AuthorizationStore:
    """Holds relationship tuples and resolves effective permissions.

    Thread-safe for concurrent reads and writes. The lock is coarse; at the
    scale where that matters this should be OpenFGA rather than a dataclass.
    """

    _tuples: set[Tuple_] = field(default_factory=set)
    # Index: obj -> tuples about that object. Permission checks walk *upward*
    # from an object to its principals far more often than downward, so this
    # is the index that matters.
    _by_object: dict[str, set[Tuple_]] = field(default_factory=dict)
    # Index: subject -> tuples where it is the subject. Used to expand a user
    # into the groups they belong to.
    _by_subject: dict[str, set[Tuple_]] = field(default_factory=dict)

    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    # Cache of user -> set of readable object ids. Invalidated wholesale on
    # any write; see the note in ``write``.
    _cache: dict[str, frozenset[str]] = field(default_factory=dict, repr=False)
    _cache_hits: int = 0
    _cache_misses: int = 0

    # -- mutation ---------------------------------------------------------

    def write(self, *tuples: Tuple_) -> None:
        """Add relationship tuples.

        Any write invalidates the whole permission cache. That is blunt, and
        deliberately so: computing the precise invalidation set for a change
        deep in a nested group hierarchy is itself a graph walk, and getting it
        subtly wrong means serving a *stale permission* — the exact failure
        this system exists to prevent. Over-invalidating costs latency;
        under-invalidating costs a leak.
        """
        with self._lock:
            for t in tuples:
                if t in self._tuples:
                    continue
                self._tuples.add(t)
                self._by_object.setdefault(t.obj, set()).add(t)
                self._by_subject.setdefault(t.subject, set()).add(t)
            self._cache.clear()

    def delete(self, *tuples: Tuple_) -> None:
        """Remove relationship tuples (a revocation)."""
        with self._lock:
            for t in tuples:
                self._tuples.discard(t)
                if t.obj in self._by_object:
                    self._by_object[t.obj].discard(t)
                if t.subject in self._by_subject:
                    self._by_subject[t.subject].discard(t)
            self._cache.clear()

    # -- resolution -------------------------------------------------------

    def expand_user(self, user: str, *, max_depth: int = 32) -> frozenset[str]:
        """Return ``user`` plus every group they belong to, transitively.

        Breadth-first with a visited set, so a cycle terminates instead of
        hanging. ``max_depth`` is a second backstop against a pathologically
        deep hierarchy.
        """
        seen: set[str] = {user}
        frontier: deque[tuple[str, int]] = deque([(user, 0)])

        with self._lock:
            while frontier:
                current, depth = frontier.popleft()
                if depth >= max_depth:
                    raise CycleError(
                        f"group nesting exceeded {max_depth} levels from {user!r}; "
                        "this is almost certainly a cycle"
                    )
                for t in self._by_subject.get(current, ()):
                    if t.relation != "member":
                        continue
                    if t.obj not in seen:
                        seen.add(t.obj)
                        frontier.append((t.obj, depth + 1))
        return frozenset(seen)

    def readable_objects(self, user: str) -> frozenset[str]:
        """Return every object id ``user`` may read.

        This is the set the retrieval layer filters against. Returning the
        *set* rather than answering one check at a time is what makes
        pre-filtered vector search possible: the filter has to be expressible
        as a predicate the index can apply during its scan, not as a callback
        into a permission service per candidate.
        """
        with self._lock:
            cached = self._cache.get(user)
            if cached is not None:
                self._cache_hits += 1
                return cached
            self._cache_misses += 1

            principals = self.expand_user(user)
            readable: set[str] = set()

            for principal in principals:
                for t in self._by_subject.get(principal, ()):
                    if t.relation in READ_RELATIONS:
                        readable.add(t.obj)

            # Propagate document permissions down to child objects (chunks).
            # A chunk inherits exactly its parent document's ACL; it never has
            # an ACL of its own. Chunks are derived artifacts, and letting them
            # carry independent permissions creates a second source of truth
            # that will drift from the first.
            readable |= self._descendants_of(readable)

            frozen = frozenset(readable)
            self._cache[user] = frozen
            return frozen

    def _descendants_of(self, parents: Iterable[str], *, max_depth: int = 16) -> set[str]:
        """Return every object reachable by ``parent`` edges from ``parents``."""
        out: set[str] = set()
        frontier: deque[tuple[str, int]] = deque((p, 0) for p in parents)
        seen: set[str] = set(parents)

        while frontier:
            current, depth = frontier.popleft()
            if depth >= max_depth:
                raise CycleError(f"object hierarchy exceeded {max_depth} levels at {current!r}")
            for t in self._by_object.get(current, ()):
                if t.relation != "parent":
                    continue
                # tuple reads: "<child> parent <parent>"
                child = t.subject
                if child not in seen:
                    seen.add(child)
                    out.add(child)
                    frontier.append((child, depth + 1))
        return out

    def check(self, user: str, obj: str) -> bool:
        """Return whether ``user`` may read ``obj``."""
        return obj in self.readable_objects(user)

    # -- introspection ----------------------------------------------------

    def explain(self, user: str, obj: str) -> list[str]:
        """Return a human-readable justification for a read decision.

        The audit trail records *why* a chunk was allowed or denied, not just
        the verdict. A decision that cannot be explained cannot be reviewed,
        and "the system said no" is not an answer an enterprise customer
        accepts during an access review.
        """
        if not self.check(user, obj):
            return [f"DENY {user} -> {obj}: no relationship path grants read"]

        principals = self.expand_user(user)
        reasons: list[str] = []
        with self._lock:
            for principal in sorted(principals):
                for t in self._by_subject.get(principal, ()):
                    if t.relation in READ_RELATIONS and t.obj == obj:
                        via = "" if principal == user else f" (via {principal})"
                        reasons.append(f"ALLOW {user} -> {obj}: {t}{via}")
                    elif t.relation in READ_RELATIONS:
                        # Maybe obj is a descendant of t.obj.
                        if obj in self._descendants_of({t.obj}):
                            via = "" if principal == user else f" (via {principal})"
                            reasons.append(
                                f"ALLOW {user} -> {obj}: inherited from {t.obj} by {t}{via}"
                            )
        return reasons or [f"ALLOW {user} -> {obj}: (path not reconstructed)"]

    def cache_stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "entries": len(self._cache),
            }

    def __iter__(self) -> Iterator[Tuple_]:
        with self._lock:
            return iter(sorted(self._tuples, key=str))

    def __len__(self) -> int:
        return len(self._tuples)
