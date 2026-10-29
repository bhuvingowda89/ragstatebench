from ragstate.world.ledger import WorldLedger
from ragstate.world.models import DocumentVersion, Principal, PermissionRule
from ragstate.transitions.engine import TransitionEngine
from ragstate.transitions.events import Transition
from ragstate.retrieval.indexing import build_chunks_from_ledger
from ragstate.retrieval.bm25 import BM25Retriever
from ragstate.retrieval.filtering import IndexedAuthorizer, LiveAuthorizer
from ragstate.retrieval.pipeline import run_retrieval

w = WorldLedger()
w.add_principal(Principal("alice","tenant_a"))
w.add_document(DocumentVersion(
    "helios-v1","helios","tenant_a",1,0,
    content="Project Helios compensation adjustment is 12 percent."
))
w.add_permission(PermissionRule("alice","helios",True,0))
chunks = build_chunks_from_ledger(w,0)
retriever = BM25Retriever(chunks)

print("Before revocation:")
print([h.chunk.document_id for h in run_retrieval(
    retriever,"Helios compensation adjustment","alice",
    LiveAuthorizer(w,9),5).context_hits])

TransitionEngine(w).apply(Transition(
    "revoke-helios",10,"PERMISSION_REVOKE","helios",
    {"principal_id":"alice","logical_document_id":"helios"}
))

stale = IndexedAuthorizer({"alice":{"helios"}})
live = LiveAuthorizer(w,10)

print("After revocation, stale indexed ACL:")
print([h.chunk.document_id for h in run_retrieval(
    retriever,"Helios compensation adjustment","alice",stale,5).context_hits])

print("After revocation, live authorization:")
print([h.chunk.document_id for h in run_retrieval(
    retriever,"Helios compensation adjustment","alice",live,5).context_hits])
