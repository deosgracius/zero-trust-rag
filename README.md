# Zero-Trust RAG

A retrieval system where every chunk carries the access-control list inherited
from its source document, and authorization is enforced **at query time, inside
the vector search**. Two users asking the identical question get different
answers based on what each is allowed to see, and an unauthorized chunk never
becomes a retrieval candidate — so no prompt-level attack can extract it.

Most RAG demos ignore permissions entirely. This one treats authorization
correctness as the product, and the evaluation suite as the proof.

## Status

| Area | State |
|---|---|
| Zanzibar-style relationship model, nested groups, revocation | Working, tested |
| Chunk-level ACL inheritance from source documents | Working, tested |
| Pre-filtered vector search | Working, tested |
| Authorization eval (containment, adversarial, side channels) | 21/21 passing |
| Hash-chained audit trail with decision basis | Working, tested |
| Unit suite | 29 tests passing |
| Real embeddings (bge/e5) | Optional extra, not exercised in eval |
| OpenFGA backend, Qdrant/pgvector, live connectors | **Not started** |
| Reranking, hybrid BM25+vector, LLM generation | **Not started** |

## Run it

```bash
py -m pytest -q
```

```bash
py eval/authorization_eval.py
```

No services, no model downloads, no API keys. See "Honest limitations" for what
that costs.

## The core architectural claim

**Authorization happens at retrieval, not at generation.**

The tempting design is to retrieve broadly and instruct the model to withhold
restricted content. That fails to a jailbreak, because it makes the model the
security boundary — and the model is a text predictor being asked to keep a
secret it has been handed.

Here the filter is applied inside the vector scan. An unauthorized chunk is
never scored, never ranked, never assembled into context. There is no prompt
that extracts it, because there is nothing to extract from.

The eval confirms this: 6 jailbreak prompts against 3 low-privilege users leak
nothing. That result is *structural* rather than lucky, which is the point
worth making — the attack surface it would exploit does not exist.

## Pre-filter vs post-filter

This distinction is the difference between a demo and a deployable system.

**Post-filtering** takes the global top-k, then drops what the caller may not
see. It never returns forbidden content, so a naive test suite calls it secure.
It leaks anyway:

- Ask for k=10, receive 4, and you have learned that 6 documents exist which
  match your query and which you cannot read. Result counts are a side channel.
- A user with access to 1% of the corpus gets almost nothing, because the
  global top-k is dominated by documents they cannot read.

**Pre-filtering** applies the predicate during the scan, so the candidate set
only ever contains readable chunks. The caller gets a true top-k over their own
visible corpus, and result counts reveal nothing.

The eval measures this directly, with a control:

| Query | Pre-filter | Post-filter |
|---|---|---|
| Matches a hidden document | 2 results | **0 results** |
| Matches nothing at all | 2 results | 1 result |

Post-filtering's counts move with content the user cannot see. Pre-filtering's
do not. The control assertion matters: if the post-filter probe *didn't* show a
difference, the pre-filter result would be meaningless — it would only prove
the probe was too weak to detect anything.

## Measured results

From `py eval/authorization_eval.py`.

### Permission resolution

The fixture is built to break naive resolvers: three levels of group nesting, a
user in two unrelated branches, a document reachable only through a
group-of-groups, a revoked user, and a document nobody can read.

| Property | Result |
|---|---|
| Three-level nesting resolves (alice reaches platform-team docs) | Pass |
| Membership does **not** flow downward (bob stays out of platform-team) | Pass |
| User in two unrelated group branches gets both | Pass |
| Seniority is not a permission (carol is not in `group:everyone`) | Pass |
| Revoked user loses access, cache invalidated | Pass |
| Document with no viewer tuple is readable by nobody | Pass |

### Containment and adversarial

| Property | Result |
|---|---|
| `returned ⊆ authorized` across 36 user × query combinations | **Zero leaks** |
| Suite is non-vacuous (24 pairs actually returned results) | Pass |
| 6 jailbreaks × 3 low-privilege users | **Zero leaks** |
| Result count reveals existence of hidden matches | No (2 vs 2) |
| Post-filter control: does leak | Yes (0 vs 1) |
| Timing difference, hidden-match vs no-match query | 0.4% (< 25% bar) |
| Audit chain verifies / tampering detected | Pass / Pass |

The **non-vacuity check** is there because a suite that returns nothing
trivially satisfies containment. "Zero leaks" only means something alongside
"and 24 of the 36 combinations actually returned results."

## Design notes worth defending

**Chunks never carry their own ACL.** They inherit from the parent document via
a `parent` relationship tuple. Copying the ACL onto each chunk creates a second
source of truth that drifts from the first the moment a permission changes, and
the drift window is exactly a leak window.

**Cache invalidation is deliberately blunt.** Any tuple write clears the entire
permission cache. Computing the precise invalidation set for a change deep in a
nested hierarchy is itself a graph walk, and getting it subtly wrong serves a
*stale permission* — the exact failure this system exists to prevent.
Over-invalidating costs latency; under-invalidating costs a leak. Two tests
pin this: revocation must invalidate, and a new grant must not be masked by a
cached negative.

**Cycles terminate rather than hang.** Mutually-nested groups happen in real
directories after enough reorganisations. An unguarded resolver loops forever —
a denial of service that anyone able to create a group can trigger. Expansion
uses a visited set, with a depth limit as a second backstop.

**`search` has no default authorization predicate.** An unauthenticated search
is not a meaningful operation, and a default of "allow everything" is the kind
of convenience that becomes an incident. A test asserts the call fails without
it.

**Decisions are explainable.** `explain()` reconstructs *why* a read was
allowed or denied. "The system said no" is not an answer that survives an
enterprise access review.

**The audit trail is the evidence.** "We filter correctly" is a claim. A
tamper-evident record of every chunk returned to every user, with the filtered
count and decision basis, is a demonstration. The filtered count is recorded
but never returned to the caller — exposing it would recreate the
existence-inference channel that pre-filtering closes.

## Honest limitations

These matter more than the passing checks, and are stated here rather than
discovered by a reader.

- **The default embedder is a hashing stub, not a semantic model.** It makes
  the authorization eval reproducible with zero dependencies, and it is
  sufficient for that eval, because what is measured is *which chunks are
  reachable*, not how well they rank. It means **no retrieval-quality number
  in this repo is meaningful.** Recall@k with and without ACL filters — the
  "cost of security" measurement the design calls for — requires the
  `real-embeddings` extra and has not been run.
- **Exact brute-force scan, not ANN.** This removes recall degradation as a
  confounder for the authorization measurements, but it also means the hardest
  version of the problem is untouched: how a filtered ANN index degrades recall
  as the filter gets more selective. That interaction is the real engineering
  difficulty at scale, and it is not addressed here.
- **The timing result is weak evidence.** An exact scan touches every chunk
  regardless of authorization, so constant-time behaviour is nearly automatic.
  A real ANN index with a payload filter would have far more timing structure,
  and this test would need to be much more careful to mean anything.
- **In-process authorization store, not OpenFGA.** No distributed consistency
  story — no zookies, no snapshot reads. It has the resolution semantics the
  eval depends on and nothing else.
- **No source connectors.** Permission sync from Google Drive or Confluence,
  the stale-permission window, and reconciliation are not built. The revocation
  path is exercised only through direct tuple deletion.
- **No generation step.** The pipeline assembles context; it does not call an
  LLM. The security claim is about what enters the context window, which is
  fully testable without generation — but the end-to-end product is not there.
- **6 documents, 11 chunks.** The corpus is a fixture designed to expose
  specific resolver bugs, not a scale test.

## Layout

```
src/ztrag/authz/model.py       relationship tuples, nested-group expansion, cache
src/ztrag/retrieval/store.py   pre-filtered vector search (+ post-filter, for contrast)
src/ztrag/retrieval/pipeline.py  resolve -> retrieve -> assemble -> audit
src/ztrag/embed/embedder.py    hashing stub + optional sentence-transformers
src/ztrag/audit/chain.py       hash-chained audit log
src/ztrag/ingest/corpus.py     fixture corpus and ground-truth expectations
eval/authorization_eval.py     the suite that produces every number above
tests/                         unit suite (29 tests)
docs/adr/                      architecture decision records
```
