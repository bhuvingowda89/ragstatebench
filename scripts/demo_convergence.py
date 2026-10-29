from ragstate.transitions.events import Transition
from ragstate.replicas.index import ContentIndexReplica, ACLMetadataReplica, CacheReplica
from ragstate.replicas.simulator import ReplicaSimulator
from ragstate.faults.scheduler import FaultSchedule

event = Transition(
    "revoke-1", 10, "PERMISSION_REVOKE", "POLICY-1",
    {"principal_id":"alice", "logical_document_id":"POLICY-1"}
)

replicas = {
    "index": ContentIndexReplica("index"),
    "acl": ACLMetadataReplica("acl"),
    "cache": CacheReplica("cache"),
}

faults = FaultSchedule()
faults.schedule_delay("index", "revoke-1", 2)
faults.schedule_delay("acl", "revoke-1", 4)
faults.schedule_delay("cache", "revoke-1", 7)

sim = ReplicaSimulator(replicas, faults)
sim.submit(event)

for t in [10, 12, 14, 17]:
    delivered = sim.run_until(t)
    print({
        "t": t,
        "delivered": [(d.replica_name, d.event.event_id) for d in delivered],
        "versions": {name: r.version() for name, r in replicas.items()},
    })
