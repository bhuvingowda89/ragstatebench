from __future__ import annotations
from dataclasses import replace
from ragstate.world.ledger import WorldLedger
from ragstate.world.models import DocumentVersion, PermissionRule, Principal
from .events import Transition

class TransitionEngine:
    def __init__(self, ledger: WorldLedger):
        self.ledger = ledger

    def apply(self, event: Transition) -> None:
        et = event.event_type
        p = event.payload
        if et in {"DOCUMENT_CREATE", "DOCUMENT_UPDATE", "DOCUMENT_SUPERSEDE"}:
            self.ledger.add_document(DocumentVersion(**p))
            if et == "DOCUMENT_SUPERSEDE" and p.get("supersedes"):
                old = self.ledger.documents[p["supersedes"]]
                self.ledger.documents[old.document_id] = replace(old, valid_until=event.logical_time)
        elif et == "DOCUMENT_DELETE":
            old = self.ledger.documents[event.resource_id]
            self.ledger.documents[event.resource_id] = replace(old, deleted_at=event.logical_time)
        elif et in {"PERMISSION_GRANT", "PERMISSION_REVOKE"}:
            self.ledger.add_permission(PermissionRule(
                principal_id=p["principal_id"],
                logical_document_id=p["logical_document_id"],
                allow=(et == "PERMISSION_GRANT"),
                valid_from=event.logical_time,
                valid_until=None,
            ))
        elif et in {"GROUP_ADD", "GROUP_REMOVE"}:
            self.ledger.group_events.append((event.logical_time, p["principal_id"],
                                             p["group_id"], et == "GROUP_ADD"))
        else:
            raise ValueError(f"Unsupported event type: {et}")
