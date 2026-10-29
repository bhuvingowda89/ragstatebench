from __future__ import annotations
from collections import defaultdict
from dataclasses import replace
from typing import Iterable
from .models import DocumentVersion, Principal, PermissionRule

class WorldLedger:
    def __init__(self):
        self.documents: dict[str, DocumentVersion] = {}
        self.principals: dict[str, Principal] = {}
        self.permissions: list[PermissionRule] = []
        self.query_gold: dict[str, set[str]] = {}
        self.group_events: list[tuple[int, str, str, bool]] = []

    def add_document(self, doc: DocumentVersion) -> None:
        self.documents[doc.document_id] = doc

    def add_principal(self, principal: Principal) -> None:
        self.principals[principal.principal_id] = principal

    def add_permission(self, rule: PermissionRule) -> None:
        self.permissions.append(rule)

    def set_query_gold(self, query_id: str, logical_document_ids: Iterable[str]) -> None:
        self.query_gold[query_id] = set(logical_document_ids)

    def document_at(self, logical_document_id: str, t: int) -> DocumentVersion | None:
        candidates = [
            d for d in self.documents.values()
            if d.logical_document_id == logical_document_id and d.is_temporally_valid(t)
        ]
        if not candidates:
            return None
        # highest current version, then authority
        return sorted(candidates, key=lambda d: (d.version, d.authority_level), reverse=True)[0]

    def allowed(self, principal_id: str, logical_document_id: str, t: int) -> bool:
        p = self.principals[principal_id]
        doc = self.document_at(logical_document_id, t)
        if doc is None or doc.tenant_id != p.tenant_id:
            return False
        # direct rule wins if active
        direct = [
            r for r in self.permissions
            if r.principal_id == principal_id
            and r.logical_document_id == logical_document_id
            and r.active(t)
        ]
        if direct:
            return direct[-1].allow
        # fallback group requirement on doc
        if not doc.required_groups:
            return True
        groups = set(p.groups)
        for at, who, group, add in self.group_events:
            if who == principal_id and at <= t:
                if add:
                    groups.add(group)
                else:
                    groups.discard(group)
        return bool(groups.intersection(doc.required_groups))

    def eligible_evidence(self, query_id: str, principal_id: str, t: int) -> set[str]:
        out = set()
        for logical_id in self.query_gold.get(query_id, set()):
            doc = self.document_at(logical_id, t)
            if doc and self.allowed(principal_id, logical_id, t):
                out.add(doc.document_id)
        return out
