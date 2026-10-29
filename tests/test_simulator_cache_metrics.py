from ragstate.transitions.events import Transition
from ragstate.replicas.index import ContentIndexReplica, ACLMetadataReplica, CacheReplica
from ragstate.replicas.simulator import ReplicaSimulator
from ragstate.replicas.cache import DependencyAwareCache, CacheEntry
from ragstate.faults.scheduler import FaultSchedule
from ragstate.oracle.metrics import convergence_time, exposure_window, rate

def test_independent_replica_delivery_times():
    e = Transition("e1", 10, "PERMISSION_REVOKE", "d1", {"principal_id":"alice"})
    replicas = {
        "index": ContentIndexReplica("index"),
        "acl": ACLMetadataReplica("acl"),
        "cache": CacheReplica("cache"),
    }
    f = FaultSchedule()
    f.schedule_delay("index","e1",1)
    f.schedule_delay("acl","e1",3)
    f.schedule_delay("cache","e1",5)
    sim = ReplicaSimulator(replicas, f)
    sim.submit(e)

    sim.run_until(11)
    assert replicas["index"].version() == 1
    assert replicas["acl"].version() == 0
    assert replicas["cache"].version() == 0

    sim.run_until(13)
    assert replicas["acl"].version() == 1
    assert replicas["cache"].version() == 0

    sim.run_until(15)
    assert replicas["cache"].version() == 1

def test_out_of_order_delivery_can_be_constructed():
    e1 = Transition("old", 10, "DOCUMENT_UPDATE", "d1", {"version":1})
    e2 = Transition("new", 11, "DOCUMENT_UPDATE", "d1", {"version":2})
    r = ContentIndexReplica("index")
    f = FaultSchedule()
    f.schedule_delay("index","old",5)   # delivered at 15
    f.schedule_delay("index","new",1)   # delivered at 12
    sim = ReplicaSimulator({"index": r}, f)
    sim.submit(e1)
    sim.submit(e2)
    sim.run_until(12)
    assert r.state["d1"]["payload"]["version"] == 2
    sim.run_until(15)
    assert r.state["d1"]["payload"]["version"] == 1

def test_payload_override_models_partial_or_corrupt_propagation():
    e = Transition("e1", 5, "DOCUMENT_UPDATE", "d1", {"version":2, "acl_version":9})
    r = ContentIndexReplica("index")
    f = FaultSchedule()
    f.schedule_payload_override("index","e1", {"acl_version":8})
    sim = ReplicaSimulator({"index": r}, f)
    sim.submit(e)
    sim.run_until(5)
    assert r.state["d1"]["payload"]["version"] == 2
    assert r.state["d1"]["payload"]["acl_version"] == 8

def test_epoch_cache_invalidates_on_permission_change():
    c = DependencyAwareCache("cache")
    c.put(CacheEntry("q1","answer","alice",{"d1":1},0,0))
    assert c.reusable("q1","alice")
    c.bump_permission_epoch("alice")
    assert not c.reusable("q1","alice")

def test_epoch_cache_invalidates_on_index_change():
    c = DependencyAwareCache("cache")
    c.put(CacheEntry("q1","answer","alice",{"d1":1},0,0))
    c.bump_index_epoch()
    assert not c.reusable("q1","alice")

def test_convergence_time_requires_stable_correctness():
    obs = [(10, False),(11, True),(12, False),(13, True),(14,True)]
    assert convergence_time(10, obs) == 3

def test_convergence_time_censors_never_stable():
    obs = [(10, False),(11,True),(12,False)]
    assert convergence_time(10, obs) is None

def test_exposure_window_and_rate():
    assert exposure_window(10,[10,11,14]) == 4
    assert exposure_window(10,[]) == 0
    assert rate([True,False,True,True]) == 0.75
