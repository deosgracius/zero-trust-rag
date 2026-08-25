"""Retrieval-layer tests: pre-filtering, containment, and the leak it closes."""

from __future__ import annotations

import pytest

from ztrag.authz.model import AuthorizationStore, Tuple_
from ztrag.embed.embedder import HashingEmbedder
from ztrag.retrieval.store import Chunk, VectorStore, cosine_similarity


@pytest.fixture
def setup():
    emb = HashingEmbedder(dimensions=128)
    store = VectorStore()
    texts = {
        "doc:public#0": "the deployment pipeline runs tests and promotes to production",
        "doc:secret#0": "the deployment of the acquisition will consolidate both teams",
        "doc:secret#1": "compensation bands for senior engineers were revised upward",
    }
    for cid, text in texts.items():
        store.add(Chunk(id=cid, doc_id=cid.split("#")[0], text=text, embedding=emb.embed(text)))

    authz = AuthorizationStore()
    authz.write(Tuple_("user:a", "viewer", "doc:public"))
    for cid in texts:
        authz.write(Tuple_(cid, "parent", cid.split("#")[0]))
    return emb, store, authz


def test_prefilter_returns_only_authorized(setup):
    emb, store, authz = setup
    readable = authz.readable_objects("user:a")
    results = store.search(emb.embed("deployment"), authorized=lambda c: c in readable, k=10)
    assert {r.chunk.id for r in results} == {"doc:public#0"}


def test_prefilter_gives_true_topk_over_visible_corpus(setup):
    """The user's own top-k, not the leftovers of a global top-k.

    "deployment" matches the secret chunk more strongly here, so a post-filter
    with k=1 would return nothing at all while pre-filtering correctly returns
    the best chunk the user can actually see.
    """
    emb, store, authz = setup
    readable = authz.readable_objects("user:a")
    q = emb.embed("the deployment of the acquisition will consolidate both teams")

    pre = store.search(q, authorized=lambda c: c in readable, k=1)
    post = store.search_postfilter_UNSAFE(q, authorized=lambda c: c in readable, k=1)

    assert len(pre) == 1, "pre-filtering must return the best authorized chunk"
    assert len(post) == 0, "post-filtering returns nothing -- the global top-1 was restricted"


def test_postfilter_leaks_existence_via_result_count(setup):
    """The control for the eval's side-channel probe.

    Post-filtering returns a different number of results depending on whether
    restricted documents matched, which is an existence oracle.
    """
    emb, store, authz = setup
    readable = authz.readable_objects("user:a")

    # k must be below the corpus size, or every chunk fits in the top-k and
    # restricted chunks never crowd authorized ones out -- which is the whole
    # mechanism of the leak. A 3-chunk corpus with k=3 shows nothing.
    hidden = store.search_postfilter_UNSAFE(
        emb.embed("compensation bands senior engineers"), authorized=lambda c: c in readable, k=1
    )
    absent = store.search_postfilter_UNSAFE(
        emb.embed("the deployment pipeline runs tests"), authorized=lambda c: c in readable, k=1
    )
    assert len(hidden) != len(absent), (
        "if these are equal the probe cannot detect the channel and the eval's "
        "control assertion is meaningless"
    )


def test_search_requires_an_authorization_predicate(setup):
    """There is no unauthenticated search. The parameter has no default."""
    emb, store, _ = setup
    with pytest.raises(TypeError):
        store.search(emb.embed("anything"), k=5)  # type: ignore[call-arg]


def test_no_authorized_chunks_returns_empty_not_error(setup):
    emb, store, _ = setup
    assert store.search(emb.embed("deployment"), authorized=lambda c: False, k=5) == []


def test_duplicate_chunk_id_rejected(setup):
    emb, store, _ = setup
    with pytest.raises(ValueError):
        store.add(Chunk(id="doc:public#0", doc_id="doc:public", text="dup", embedding=emb.embed("dup")))


class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert cosine_similarity((1.0, 0.0), (1.0, 0.0)) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity((1.0, 0.0), (0.0, 1.0)) == pytest.approx(0.0)

    def test_zero_vector_scores_zero_rather_than_raising(self):
        assert cosine_similarity((0.0, 0.0), (1.0, 1.0)) == 0.0

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError):
            cosine_similarity((1.0, 0.0), (1.0, 0.0, 0.0))


class TestHashingEmbedder:
    def test_deterministic(self):
        e = HashingEmbedder(64)
        assert e.embed("the deployment pipeline") == e.embed("the deployment pipeline")

    def test_shared_vocabulary_scores_higher(self):
        e = HashingEmbedder(256)
        q = e.embed("deployment pipeline production")
        near = e.embed("the deployment pipeline promotes to production")
        far = e.embed("compensation bands equity refresh grants")
        assert cosine_similarity(q, near) > cosine_similarity(q, far)

    def test_normalised(self):
        e = HashingEmbedder(64)
        v = e.embed("some ordinary text here")
        assert sum(x * x for x in v) == pytest.approx(1.0)

    def test_empty_text_does_not_crash(self):
        e = HashingEmbedder(32)
        assert e.embed("") == tuple([0.0] * 32)
