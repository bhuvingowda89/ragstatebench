from ragstate.world.ledger import WorldLedger
from ragstate.world.models import DocumentVersion, Principal, PermissionRule

def base_ledger():
    w = WorldLedger()
    w.add_principal(Principal("alice", "t1", frozenset({"eng"})))
    w.add_document(DocumentVersion(
        document_id="d1v1", logical_document_id="d1", tenant_id="t1",
        version=1, valid_from=0, content="v1"
    ))
    w.set_query_gold("q1", {"d1"})
    return w

def test_document_visible_when_current_and_allowed():
    w = base_ledger()
    assert w.eligible_evidence("q1", "alice", 1) == {"d1v1"}

def test_cross_tenant_denied():
    w = base_ledger()
    w.add_principal(Principal("bob", "t2", frozenset()))
    assert not w.allowed("bob", "d1", 1)

def test_direct_revoke_denies():
    w = base_ledger()
    w.add_permission(PermissionRule("alice", "d1", False, 5))
    assert w.allowed("alice", "d1", 4)
    assert not w.allowed("alice", "d1", 5)
