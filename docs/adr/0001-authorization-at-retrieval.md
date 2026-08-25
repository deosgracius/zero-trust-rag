# ADR 0001 — Authorization at retrieval, not at generation

**Status:** Accepted · **Date:** 2026-08-25

## Context

A RAG system serving multiple users over a shared corpus must return different
answers to the same question depending on who asked. There are three places the
permission check could live:

1. **At generation.** Retrieve broadly, place everything in context, and
   instruct the model to withhold restricted content.
2. **After retrieval.** Retrieve the global top-k, then drop unauthorized
   chunks before assembling context.
3. **Inside retrieval.** Apply the authorization predicate as part of the
   vector scan, so unauthorized chunks are never candidates.

Option 1 is what most demos do, usually implicitly, by not thinking about it.

## Decision

Enforce authorization inside retrieval. An unauthorized chunk is never scored,
never ranked, and never assembled into context.

## Consequences

**Prompt injection cannot reach restricted content.** This is the property that
matters. Under option 1, the model is the security boundary — a text predictor
asked to keep a secret it has been handed. Every jailbreak technique is an
attack on that boundary, and the defender has to win every time. Under option 3
there is no boundary to attack, because the restricted text was never in the
process.

The eval reflects this: 6 jailbreak prompts against 3 low-privilege users leak
nothing, and that result is structural rather than fortunate. Worth being
precise about what it does and does not show — it demonstrates that the attack
surface is absent, not that any particular defence is strong.

**Option 2 is not a middle ground; it is a leak.** Post-filtering never returns
forbidden *content*, which is why it survives naive testing. It leaks
*metadata*: ask for k=10, receive 4, and you have learned that 6 matching
documents exist that you cannot see. Repeated with crafted queries, that is an
oracle for the contents of a corpus you have no access to.

It is also functionally poor. A user with access to 1% of the corpus gets
almost nothing back, because the global top-k is dominated by documents they
cannot read. Recovering their true top-k means over-retrieving by an unbounded
factor.

The post-filter implementation is kept in the codebase as
`search_postfilter_UNSAFE`, used only by the eval to demonstrate the channel.
Keeping it makes the argument concrete rather than theoretical, and the eval
asserts that it *does* leak — a control which proves the probe is capable of
detecting what it claims to test.

**The authorization predicate must be expressible to the index.** This is the
real constraint option 3 imposes. The filter cannot be a callback into a
permission service per candidate; that is a network round trip per vector
comparison. It has to be a set membership or a payload predicate the index can
apply during its own scan. This is why `readable_objects` returns the *whole
set* of readable ids rather than answering one check at a time, and why the
effective-permission cache exists.

**Chunks inherit ACLs, they do not own them.** A chunk carries no independent
permission. It reaches its ACL through a `parent` relationship to its source
document. Copying the ACL onto each chunk at ingest would create a second
source of truth, and the window during which the copy disagrees with the
original is precisely a leak window.

## Alternatives considered

**Per-user index partitions.** Give each user their own vector index containing
only what they can read. Perfect isolation, and it does not survive contact
with reality: storage multiplies by the number of users, and every ACL change
requires re-indexing.

**Encrypt chunks under per-document keys.** Retrieve everything, decrypt only
what the caller holds keys for. Moves the problem to key distribution, and
similarity search over ciphertext does not work, so embeddings would have to
stay in the clear — which is where the leakage is anyway.

**Trust the model, verify the output.** Retrieve broadly, generate, then scan
the response for restricted content. Requires knowing what is restricted well
enough to detect paraphrase, which is a harder problem than the one being
solved.

## Follow-ups

- The current store is an exact brute-force scan. The interesting version of
  this decision is filtered ANN, where the filter and the index interact:
  as the filter gets more selective, graph-based indexes like HNSW lose recall
  because the reachable subgraph fragments. Quantifying that degradation is the
  main open engineering question — see ADR 0002.
- No generation step exists yet. The security claim concerns what enters the
  context window, which is fully testable without it, but the end-to-end
  product is incomplete.
