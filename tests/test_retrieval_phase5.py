from ragstate.world.ledger import WorldLedger
from ragstate.world.models import DocumentVersion, Principal, PermissionRule
from ragstate.world.generator import generate_world
from ragstate.retrieval.chunking import chunk_document
from ragstate.retrieval.bm25 import BM25Retriever
from ragstate.retrieval.indexing import build_chunks_from_ledger
from ragstate.retrieval.filtering import IndexedAuthorizer, LiveAuthorizer
from ragstate.retrieval.pipeline import run_retrieval
from ragstate.retrieval.metrics import (
    current_evidence_recall,
    superseded_retrieval_rate,
    unauthorized_context_rate,
    authorized_retrieval_denial,
    mixed_version_context_rate,
    state_consistent_retrieval_success,
)
from ragstate.transitions.engine import TransitionEngine
from ragstate.transitions.events import Transition

def test_chunk_preserves_provenance():
    d = DocumentVersion("d1v2","d1","t1",2,0,content="alpha beta gamma")
    chunks = chunk_document(d, acl_version=9, max_words=2)
    assert chunks[0].document_id == "d1v2"
    assert chunks[0].logical_document_id == "d1"
    assert chunks[0].version == 2
    assert chunks[0].acl_version == 9

def test_bm25_retrieves_matching_document():
    w = WorldLedger()
    w.add_principal(Principal("alice","t1"))
    w.add_document(DocumentVersion("travel-v1","travel","t1",1,0,content="Travel reimbursement maximum is 85 dollars."))
    chunks = build_chunks_from_ledger(w, 0)
    r = BM25Retriever(chunks)
    hits = r.retrieve("travel reimbursement maximum", top_k=5)
    assert hits
    assert hits[0].chunk.document_id == "travel-v1"

def test_live_authorizer_blocks_revoked_doc_while_stale_index_authorizer_allows():
    w = WorldLedger()
    w.add_principal(Principal("alice","t1"))
    w.add_document(DocumentVersion("secret-v1","secret","t1",1,0,content="Project Helios adjustment is 12 percent."))
    w.set_query_gold("q1", {"secret"})
    w.add_permission(PermissionRule("alice","secret", True, 0))
    chunks = build_chunks_from_ledger(w, 0)
    r = BM25Retriever(chunks)

    eng = TransitionEngine(w)
    eng.apply(Transition("rev1", 10, "PERMISSION_REVOKE", "secret", {
        "principal_id":"alice","logical_document_id":"secret"
    }))

    stale = IndexedAuthorizer({"alice":{"secret"}})
    live = LiveAuthorizer(w, 10)

    stale_result = run_retrieval(r, "Helios adjustment", "alice", stale, 5)
    live_result = run_retrieval(r, "Helios adjustment", "alice", live, 5)

    assert len(stale_result.context_hits) == 1
    assert len(live_result.context_hits) == 0
    assert unauthorized_context_rate(stale_result.context_hits, "alice", w, 10) == 1.0

def test_old_version_can_outrank_new_when_both_are_indexed():
    w = WorldLedger()
    w.add_principal(Principal("alice","t1"))
    w.add_document(DocumentVersion("travel-v1","travel","t1",1,0,valid_until=10,
        content="Travel reimbursement maximum is 75 dollars. Travel reimbursement travel reimbursement."))
    w.add_document(DocumentVersion("travel-v2","travel","t1",2,10,
        content="Approved limit is 85 dollars."))
    chunks = build_chunks_from_ledger(w, 10, include_all_versions=True)
    r = BM25Retriever(chunks)
    hits = r.retrieve("travel reimbursement maximum", top_k=2)
    assert hits[0].chunk.document_id == "travel-v1"
    current = {"travel-v2"}
    assert superseded_retrieval_rate(hits, current) > 0

def test_mixed_version_metric():
    w = WorldLedger()
    w.add_principal(Principal("alice","t1"))
    w.add_document(DocumentVersion("d1v1","d1","t1",1,0,valid_until=5,content="alpha beta"))
    w.add_document(DocumentVersion("d1v2","d1","t1",2,5,content="alpha beta new"))
    chunks = build_chunks_from_ledger(w, 5, include_all_versions=True)
    r = BM25Retriever(chunks)
    hits = r.retrieve("alpha beta", top_k=10)
    assert mixed_version_context_rate(hits) == 1.0

def test_state_consistent_success_requires_current_allowed_gold():
    w = WorldLedger()
    w.add_principal(Principal("alice","t1"))
    w.add_document(DocumentVersion("d1v1","d1","t1",1,0,content="alpha current policy"))
    w.set_query_gold("q1", {"d1"})
    chunks = build_chunks_from_ledger(w, 0)
    hits = BM25Retriever(chunks).retrieve("alpha current policy", 5)
    req = w.eligible_evidence("q1","alice",0)
    assert current_evidence_recall(hits, req) == 1.0
    assert state_consistent_retrieval_success(hits, req, "alice", w, 0)

def test_state_consistency_accepts_paragraph_whitespace_but_rejects_corrupt_content():
    w = WorldLedger()
    w.add_principal(Principal("alice", "t1"))
    w.add_document(DocumentVersion("policy-v1", "policy", "t1", 1, 0,
                                   content="The current limit is 85 dollars.\n\nApproval is required."))
    hits = BM25Retriever(build_chunks_from_ledger(w, 0)).retrieve("current limit approval", 1)
    assert state_consistent_retrieval_success(hits, {"policy-v1"}, "alice", w, 0)
    from dataclasses import replace
    from ragstate.retrieval.models import RetrievalHit
    corrupted = [RetrievalHit(replace(hits[0].chunk, text=hits[0].chunk.text.replace("85", "95")),
                              hits[0].score)]
    assert not state_consistent_retrieval_success(corrupted, {"policy-v1"}, "alice", w, 0)

def test_generated_world_is_deterministic():
    a = generate_world(seed=17, docs_per_domain=3)
    b = generate_world(seed=17, docs_per_domain=3)
    assert sorted(a.documents) == sorted(b.documents)
    assert [a.documents[k].content for k in sorted(a.documents)] == [b.documents[k].content for k in sorted(b.documents)]
