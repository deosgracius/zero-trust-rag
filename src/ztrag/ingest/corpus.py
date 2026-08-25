"""A synthetic corpus with a deliberately awkward permission structure.

The shape here is chosen to break naive implementations. It includes nested
groups three levels deep, a user who belongs to two unrelated branches, a
document readable only through a group-of-groups, a revoked user who used to
have access, and a document nobody can read. Each of those is a case where a
plausible-but-wrong permission resolver produces the wrong answer.

Chunks never carry their own ACL. They inherit from their parent document via
a ``parent`` tuple, so there is exactly one source of truth for who can read
what.
"""

from __future__ import annotations

from dataclasses import dataclass

from ztrag.authz.model import AuthorizationStore, Tuple_
from ztrag.embed.embedder import Embedder
from ztrag.retrieval.store import Chunk, VectorStore


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    title: str
    source: str
    paragraphs: tuple[str, ...]


DOCUMENTS: tuple[Document, ...] = (
    Document(
        id="doc:onboarding-guide",
        title="Engineering Onboarding Guide",
        source="confluence",
        paragraphs=(
            "New engineers should request access to the deployment tooling on their first day. "
            "The platform team reviews access requests every morning.",
            "Our deployment pipeline runs tests, builds a container image, and promotes through "
            "staging before production. Every promotion requires a green test suite.",
        ),
    ),
    Document(
        id="doc:incident-runbook",
        title="Production Incident Runbook",
        source="confluence",
        paragraphs=(
            "When the error rate alert fires, first check the deployment history for a recent "
            "release. Most incidents correlate with a deployment in the previous hour.",
            "Roll back with the deployment tooling rather than editing configuration by hand. "
            "Manual configuration edits are the most common cause of prolonged incidents.",
        ),
    ),
    Document(
        id="doc:comp-bands",
        title="Compensation Bands FY26",
        source="google-drive",
        paragraphs=(
            "Senior engineer compensation ranges from 180,000 to 240,000 base salary depending "
            "on location and level within the band.",
            "Equity refresh grants are reviewed annually and target the 60th percentile of the "
            "market data set for each band.",
        ),
    ),
    Document(
        id="doc:acquisition-memo",
        title="Project Lighthouse Acquisition Memo",
        source="google-drive",
        paragraphs=(
            "The board has approved an offer for the target company at a valuation of 45 million, "
            "subject to completion of technical due diligence.",
            "Deployment of the combined engineering organisation would consolidate both incident "
            "response rotations into a single on-call schedule.",
        ),
    ),
    Document(
        id="doc:security-review",
        title="Q3 Security Review Findings",
        source="confluence",
        paragraphs=(
            "The review identified three high-severity findings in the deployment pipeline, "
            "including a service account with excessive permissions.",
            "Remediation of the excessive service account permissions is tracked and assigned to "
            "the platform team for the next quarter.",
        ),
    ),
    Document(
        id="doc:orphaned-legal",
        title="Legal Hold Notice",
        source="google-drive",
        paragraphs=(
            "This document is subject to a legal hold and its access list has been cleared "
            "pending review by outside counsel.",
        ),
    ),
)


def build_authorization() -> AuthorizationStore:
    """Return the store wired with the fixture permission graph.

    The hierarchy, and why each piece is here:

      group:everyone
        +- group:engineering            (nested one level)
             +- group:platform-team     (nested two levels)
      group:leadership                  (separate branch)
      group:deal-team                   (separate branch, narrowest access)

      alice   -> platform-team          (inherits engineering AND everyone)
      bob     -> engineering            (inherits everyone, NOT platform-team)
      carol   -> leadership + deal-team (two unrelated branches at once)
      dave    -> everyone only          (weakest real user)
      mallory -> (nothing)              (revoked; tuple deleted below)
    """
    store = AuthorizationStore()

    store.write(
        # -- group nesting ------------------------------------------------
        Tuple_("group:engineering", "member", "group:everyone"),
        Tuple_("group:platform-team", "member", "group:engineering"),
        # -- user memberships ---------------------------------------------
        Tuple_("user:alice", "member", "group:platform-team"),
        Tuple_("user:bob", "member", "group:engineering"),
        Tuple_("user:carol", "member", "group:leadership"),
        Tuple_("user:carol", "member", "group:deal-team"),
        Tuple_("user:dave", "member", "group:everyone"),
        # -- document permissions -----------------------------------------
        Tuple_("group:everyone", "viewer", "doc:onboarding-guide"),
        Tuple_("group:engineering", "viewer", "doc:incident-runbook"),
        # Readable only by reaching platform-team, which only alice does.
        Tuple_("group:platform-team", "viewer", "doc:security-review"),
        Tuple_("group:leadership", "viewer", "doc:comp-bands"),
        Tuple_("group:deal-team", "owner", "doc:acquisition-memo"),
        # doc:orphaned-legal deliberately has NO viewer tuple: nobody reads it.
    )

    # Mallory had access and it was revoked. Writing then deleting exercises
    # the revocation path and the cache invalidation that must follow it --
    # a stale cache here is exactly a stale-permission leak.
    store.write(Tuple_("user:mallory", "member", "group:deal-team"))
    store.delete(Tuple_("user:mallory", "member", "group:deal-team"))

    return store


def build_store(embedder: Embedder, authz: AuthorizationStore) -> VectorStore:
    """Chunk every document, embed it, and register parent tuples."""
    store = VectorStore()
    for doc in DOCUMENTS:
        for i, para in enumerate(doc.paragraphs):
            chunk_id = f"{doc.id}#{i}"
            store.add(
                Chunk(
                    id=chunk_id,
                    doc_id=doc.id,
                    text=para,
                    embedding=embedder.embed(para),
                    source=doc.source,
                )
            )
            # "<chunk> parent <doc>": the chunk inherits the document's ACL.
            authz.write(Tuple_(chunk_id, "parent", doc.id))
    return store


# Users in the fixture, and the documents each should be able to read.
# This is the ground truth the authorization eval asserts against; it is
# written out by hand rather than computed, so a bug in the resolver cannot
# quietly redefine what "correct" means.
EXPECTED_READABLE_DOCS: dict[str, set[str]] = {
    "user:alice": {
        "doc:onboarding-guide",   # via everyone
        "doc:incident-runbook",   # via engineering
        "doc:security-review",    # via platform-team
    },
    "user:bob": {
        "doc:onboarding-guide",   # via everyone
        "doc:incident-runbook",   # via engineering
        # NOT security-review: bob is in engineering, not platform-team.
    },
    "user:carol": {
        "doc:comp-bands",         # via leadership
        "doc:acquisition-memo",   # via deal-team
        # NOT onboarding-guide: carol is not in group:everyone.
    },
    "user:dave": {
        "doc:onboarding-guide",   # via everyone
    },
    "user:mallory": set(),        # revoked
    "user:nobody": set(),         # never existed
}

# Queries chosen so each one has a strong lexical match inside a *restricted*
# document. A pipeline that leaks will leak on these.
PROBE_QUERIES: tuple[str, ...] = (
    "what is the deployment process",
    "senior engineer compensation salary band",
    "acquisition valuation board approved offer",
    "incident response rollback procedure",
    "service account excessive permissions finding",
    "legal hold outside counsel",
)
