from ragstate.faults.scheduler import FaultSchedule
from ragstate.transitions.events import Transition
from ragstate.replicas.index import ContentIndexReplica
from ragstate.world.ledger import WorldLedger
from ragstate.world.models import DocumentVersion, Principal
from ragstate.oracle.evaluator import QueryOracle

def test_fault_scheduler_delay_and_drop():
    e = Transition("e1", 10, "DOCUMENT_UPDATE", "d1", {"x":1})
    f = FaultSchedule()
    f.schedule_delay("index", "e1", 4)
    assert f.delivery_time("index", e) == 14
    f.schedule_drop("cache", "e1")
    assert f.delivery_time("cache", e) is None

def test_replica_hash_changes_after_event():
    e = Transition("e1", 10, "DOCUMENT_UPDATE", "d1", {"x":1})
    r = ContentIndexReplica("index")
    h0 = r.state_hash()
    r.apply(e)
    assert r.state_hash() != h0
    assert r.version() == 1

def test_oracle_uses_canonical_state():
    w = WorldLedger()
    w.add_principal(Principal("alice", "t1"))
    w.add_document(DocumentVersion("d1v1","d1","t1",1,0,content="x"))
    w.set_query_gold("q1", {"d1"})
    o = QueryOracle(w)
    res = o.evaluate("q1","alice",1)
    assert res.answerable
    assert res.eligible_document_ids == frozenset({"d1v1"})
