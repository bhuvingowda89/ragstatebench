from ragstate.world.ledger import WorldLedger
from ragstate.world.models import DocumentVersion, Principal
from ragstate.transitions.engine import TransitionEngine
from ragstate.transitions.events import Transition

def test_supersede_closes_old_version():
    w = WorldLedger()
    w.add_principal(Principal("alice", "t1"))
    w.add_document(DocumentVersion("d1v1","d1","t1",1,0,content="old"))
    eng = TransitionEngine(w)
    eng.apply(Transition("e2", 10, "DOCUMENT_SUPERSEDE", "d1v2", {
        "document_id":"d1v2","logical_document_id":"d1","tenant_id":"t1",
        "version":2,"valid_from":10,"supersedes":"d1v1","content":"new"
    }))
    assert w.document_at("d1", 9).document_id == "d1v1"
    assert w.document_at("d1", 10).document_id == "d1v2"

def test_delete_removes_from_canonical_view():
    w = WorldLedger()
    w.add_document(DocumentVersion("d1v1","d1","t1",1,0,content="old"))
    eng = TransitionEngine(w)
    eng.apply(Transition("e3", 7, "DOCUMENT_DELETE", "d1v1", {}))
    assert w.document_at("d1", 6).document_id == "d1v1"
    assert w.document_at("d1", 7) is None
