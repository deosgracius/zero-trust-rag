"""Unit tests for permission resolution.

The eval suite proves end-to-end containment on a fixed corpus. These cover the
failure modes that a fixed corpus would not surface: cycles, cache staleness,
and the boundary between "member of" and "can read".
"""

from __future__ import annotations

import pytest

from ztrag.authz.model import AuthorizationStore, CycleError, Tuple_


def test_direct_grant():
    s = AuthorizationStore()
    s.write(Tuple_("user:a", "viewer", "doc:x"))
    assert s.check("user:a", "doc:x")
    assert not s.check("user:b", "doc:x")


def test_nested_group_expansion():
    s = AuthorizationStore()
    s.write(
        Tuple_("user:a", "member", "group:inner"),
        Tuple_("group:inner", "member", "group:middle"),
        Tuple_("group:middle", "member", "group:outer"),
        Tuple_("group:outer", "viewer", "doc:x"),
    )
    assert s.check("user:a", "doc:x"), "permission must flow up three levels of nesting"


def test_membership_does_not_flow_downward():
    """Being in a parent group must not grant a child group's permissions.

    This is the direction people get backwards. Alice in group:everyone should
    not inherit what group:platform-team can see, even though platform-team is
    a member of everyone.
    """
    s = AuthorizationStore()
    s.write(
        Tuple_("group:special", "member", "group:general"),
        Tuple_("user:a", "member", "group:general"),
        Tuple_("group:special", "viewer", "doc:secret"),
    )
    assert not s.check("user:a", "doc:secret")


def test_chunks_inherit_document_acl():
    s = AuthorizationStore()
    s.write(
        Tuple_("user:a", "viewer", "doc:x"),
        Tuple_("doc:x#0", "parent", "doc:x"),
        Tuple_("doc:x#1", "parent", "doc:x"),
    )
    assert s.check("user:a", "doc:x#0")
    assert s.check("user:a", "doc:x#1")


def test_chunk_of_unreadable_document_is_unreadable():
    s = AuthorizationStore()
    s.write(
        Tuple_("user:a", "viewer", "doc:x"),
        Tuple_("doc:y#0", "parent", "doc:y"),
    )
    assert not s.check("user:a", "doc:y#0")


def test_group_cycle_terminates():
    """A cycle must raise, not hang.

    Mutually-nested groups occur in real directories after enough
    reorganisations. An unguarded resolver loops forever, which is a denial of
    service anyone able to create a group can trigger.
    """
    s = AuthorizationStore()
    s.write(
        Tuple_("group:a", "member", "group:b"),
        Tuple_("group:b", "member", "group:c"),
        Tuple_("group:c", "member", "group:a"),
        Tuple_("user:x", "member", "group:a"),
    )
    # The visited set makes expansion terminate rather than raise.
    principals = s.expand_user("user:x")
    assert principals == {"user:x", "group:a", "group:b", "group:c"}


def test_deep_nesting_beyond_limit_raises():
    s = AuthorizationStore()
    for i in range(40):
        s.write(Tuple_(f"group:{i}", "member", f"group:{i+1}"))
    s.write(Tuple_("user:x", "member", "group:0"))
    with pytest.raises(CycleError):
        s.expand_user("user:x", max_depth=8)


def test_revocation_invalidates_cache():
    """The stale-permission failure mode, directly.

    Reading first populates the cache; the revocation must clear it. A cache
    that survives a revocation is a leak with a time window.
    """
    s = AuthorizationStore()
    grant = Tuple_("user:a", "viewer", "doc:x")
    s.write(grant)
    assert s.check("user:a", "doc:x")  # populates cache

    s.delete(grant)
    assert not s.check("user:a", "doc:x"), "revocation did not invalidate the cached permission set"


def test_grant_invalidates_cache():
    s = AuthorizationStore()
    assert not s.check("user:a", "doc:x")  # caches the empty set
    s.write(Tuple_("user:a", "viewer", "doc:x"))
    assert s.check("user:a", "doc:x"), "new grant was masked by a cached negative"


def test_group_removal_revokes_transitively():
    s = AuthorizationStore()
    membership = Tuple_("user:a", "member", "group:g")
    s.write(membership, Tuple_("group:g", "viewer", "doc:x"))
    assert s.check("user:a", "doc:x")

    s.delete(membership)
    assert not s.check("user:a", "doc:x")


@pytest.mark.parametrize("relation", ["viewer", "editor", "owner"])
def test_read_granting_relations(relation):
    s = AuthorizationStore()
    s.write(Tuple_("user:a", relation, "doc:x"))
    assert s.check("user:a", "doc:x")


def test_member_relation_alone_does_not_grant_read():
    """"member of a document" is not "can read the document"."""
    s = AuthorizationStore()
    s.write(Tuple_("user:a", "member", "doc:x"))
    assert not s.check("user:a", "doc:x")


def test_explain_produces_a_reason():
    s = AuthorizationStore()
    s.write(
        Tuple_("user:a", "member", "group:g"),
        Tuple_("group:g", "viewer", "doc:x"),
    )
    allow = s.explain("user:a", "doc:x")
    assert any("ALLOW" in line and "group:g" in line for line in allow)

    deny = s.explain("user:a", "doc:y")
    assert any("DENY" in line for line in deny)
