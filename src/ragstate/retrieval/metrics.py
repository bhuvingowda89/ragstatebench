from __future__ import annotations

def current_evidence_recall(hits, gold_document_ids: set[str]) -> float | None:
    if not gold_document_ids:
        return None
    got = {h.chunk.document_id for h in hits}
    return len(got & gold_document_ids) / len(gold_document_ids)

def superseded_retrieval_rate(hits, current_doc_ids: set[str],
                             ledger=None, logical_time: int | None = None) -> float:
    if not hits:
        return 0.0
    if ledger is None:
        bad = sum(h.chunk.document_id not in current_doc_ids for h in hits)
    else:
        if logical_time is None:
            raise ValueError("logical_time is required with ledger")
        bad = sum((current := ledger.document_at(h.chunk.logical_document_id, logical_time))
                  is not None and current.document_id != h.chunk.document_id for h in hits)
    return bad / len(hits)


def deleted_evidence_persistence(hits, ledger, logical_time: int) -> bool:
    return any(ledger.document_at(h.chunk.logical_document_id, logical_time) is None
               for h in hits)


def cross_tenant_exposure(hits, principal_id: str, ledger) -> bool:
    tenant = ledger.principals[principal_id].tenant_id
    return any(h.chunk.tenant_id != tenant for h in hits)

def authorization_violation(chunk, principal_id: str, ledger, logical_time: int) -> bool:
    """Policy denial independent of temporal deletion of the indexed version.

    Unknown resources are outside the authorization evaluation universe. A deleted
    version remains in that universe if its canonical record exists: direct ACL,
    tenant, and group policy can then be evaluated without reviving the document.
    """
    recorded = ledger.documents.get(chunk.document_id)
    if recorded is None or recorded.logical_document_id != chunk.logical_document_id:
        return False
    current = ledger.document_at(chunk.logical_document_id, logical_time)
    if current is not None:
        return not ledger.allowed(principal_id, chunk.logical_document_id, logical_time)
    principal = ledger.principals[principal_id]
    if recorded.tenant_id != principal.tenant_id:
        return True
    direct = [rule for rule in ledger.permissions
              if rule.principal_id == principal_id
              and rule.logical_document_id == chunk.logical_document_id
              and rule.active(logical_time)]
    if direct:
        return not direct[-1].allow
    groups = set(principal.groups)
    for at, who, group, add in ledger.group_events:
        if who == principal_id and at <= logical_time:
            if add:
                groups.add(group)
            else:
                groups.discard(group)
    return bool(recorded.required_groups) and not bool(groups.intersection(recorded.required_groups))


def unauthorized_context_rate(context_hits, principal_id: str, ledger, logical_time: int) -> float:
    if not context_hits:
        return 0.0
    bad = 0
    for h in context_hits:
        if authorization_violation(h.chunk, principal_id, ledger, logical_time):
            bad += 1
    return bad / len(context_hits)


def invalid_evidence_absence_success(hits, required_doc_ids: set[str],
                                     relevant_logical_ids: set[str]) -> int | None:
    """For empty gold only: 1 when no query-target evidence reaches context."""
    if required_doc_ids:
        return None
    return int(not any(h.chunk.logical_document_id in relevant_logical_ids for h in hits))

def authorized_retrieval_denial(required_doc_ids: set[str], context_hits) -> bool:
    got = {h.chunk.document_id for h in context_hits}
    return bool(required_doc_ids - got)

def mixed_version_context_rate(hits) -> float:
    versions = {}
    for h in hits:
        versions.setdefault(h.chunk.logical_document_id, set()).add(h.chunk.version)
    return 1.0 if any(len(v) > 1 for v in versions.values()) else 0.0

def state_consistent_retrieval_success(hits, required_doc_ids: set[str], principal_id: str, ledger, logical_time: int) -> bool:
    got = {h.chunk.document_id for h in hits}
    if not required_doc_ids.issubset(got):
        return False
    for h in hits:
        if not ledger.allowed(principal_id, h.chunk.logical_document_id, logical_time):
            return False
        current = ledger.document_at(h.chunk.logical_document_id, logical_time)
        if current is None or current.document_id != h.chunk.document_id:
            return False
        # Chunk construction joins words across paragraph breaks. Compare the
        # same word sequence while preserving every non-whitespace character.
        if h.chunk.text not in " ".join(current.content.split()):
            return False
    return True
