# ADR 0002 — Relationship-based permissions and blunt cache invalidation

**Status:** Accepted · **Date:** 2026-08-25

## Context

Enterprise document permissions are not a matrix. They are a graph: users
belong to groups, groups belong to other groups, folders grant access to their
contents, and a document may be reachable through several unrelated paths at
once. Any model that enumerates `(user, document)` pairs has to recompute a
combinatorial set on every organisational change, and will be stale in between.

Google Zanzibar solved this with *relationship tuples* — `subject relation
object` — and derived permissions by walking the graph. OpenFGA is the
open-source implementation.

## Decision

Model permissions as relationship tuples and resolve them by graph walk.

```
user:alice          member   group:platform-team
group:platform-team member   group:engineering
group:engineering   viewer   doc:security-review
doc:security-review#0 parent doc:security-review
```

Alice reads chunk 0 because those four tuples compose. Nothing anywhere states
that fact directly.

Resolution is exposed as `readable_objects(user) -> frozenset[str]` rather than
as a per-object `check`, because the retrieval layer needs a predicate it can
apply during an index scan (ADR 0001).

This is an in-process implementation with Zanzibar's *resolution semantics* but
none of its distributed machinery — no zookies, no snapshot reads, no
consistency tokens. The reason is that the properties under test are
resolution-correctness properties, and they are testable without a service.
Swapping in OpenFGA replaces this module, not its callers.

## Consequences

**Direction matters, and it is the thing people get backwards.** Membership
flows *upward*: a member of `platform-team` is transitively a member of
`engineering` and inherits what engineering can read. It must not flow
*downward*: a member of `engineering` is not a member of `platform-team` and
must not see platform-team documents. The fixture encodes exactly this case
(alice vs bob) and a test asserts the negative direction, because a resolver
that walks the graph without minding direction passes the positive test and
fails silently on the negative one.

**Cycles must terminate.** Mutually-nested groups occur in real directories
after enough reorganisations. An unguarded resolver loops forever — a denial of
service reachable by anyone who can create a group. Expansion uses a visited
set, with a depth limit as an independent backstop in case the visited-set
logic is ever broken by a refactor.

**Cache invalidation is deliberately blunt: any write clears everything.**

This is the decision most worth defending, because it looks lazy.

Computing the precise invalidation set for a tuple change is itself a graph
walk — adding one member to a deeply nested group can change the effective
permissions of thousands of users, through paths that are not locally visible
at the write site. Getting that walk subtly wrong produces a *stale permission*:
a user who no longer has access continuing to retrieve documents. That is the
exact failure this entire project exists to prevent.

The asymmetry is what settles it:

- Over-invalidating costs recomputation. Bounded, measurable, and it degrades
  latency.
- Under-invalidating costs a leak with a time window, and produces no symptom.

At the scale where full invalidation is too expensive, the answer is OpenFGA
with its own consistency model, not a hand-rolled invalidation graph.

Two tests pin the behaviour from both sides: a revocation must invalidate a
cached positive, and a new grant must not be masked by a cached negative. The
second is easy to forget — a cache that only invalidates on delete looks
correct until someone is granted access and cannot use it.

**Chunk inheritance uses the same mechanism as everything else.** A chunk has a
`parent` tuple pointing at its document. No special case, no denormalised ACL
field, one source of truth.

**Decisions are explainable.** `explain()` reconstructs the path that granted
or denied access. An access review that cannot answer "why does this person
have access to this?" is not a review.

## Alternatives considered

**ACL list per document.** Simple, and it collapses on nested groups: every
group change requires rewriting every document's list.

**Role-based access control.** Roles are a flat namespace. Real permission
structures are hierarchical and overlapping, and RBAC models them by
multiplying roles until the role list is itself unmanageable.

**Materialised `(user, object)` permission table.** Fast reads, and it makes
the invalidation problem worse rather than better: the same graph walk is
needed to keep it fresh, and now the stale window is durable instead of
in-memory.

## Follow-ups

- Permission sync from real sources (Google Drive, Confluence) is not built.
  That introduces the *stale-permission window* between an upstream ACL change
  and its propagation, which is the genuinely hard version of this problem and
  is untouched here. Webhook-driven incremental sync plus a reconciliation job
  is the intended shape.
- Cache is unbounded and per-process. Needs an eviction policy and, for
  multiple replicas, either shared state or per-replica invalidation
  broadcasts.
- `readable_objects` materialises the full readable set. For a user who can
  read millions of chunks this is the wrong shape, and the filter should be
  pushed into the index as a predicate over document ids rather than chunk ids.
