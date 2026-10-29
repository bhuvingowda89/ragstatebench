from __future__ import annotations
import random
from .ledger import WorldLedger
from .models import DocumentVersion, Principal, PermissionRule

DOMAINS = ["hr","finance","engineering","security","sales","legal"]

def generate_world(seed: int = 7, docs_per_domain: int = 20, *,
                   tenants: int = 2, principals_per_tenant: int = 3,
                   versioned_fraction: float = 0.0) -> WorldLedger:
    """Generate repeatable six-domain worlds; defaults preserve the original fixture."""
    if tenants < 1 or principals_per_tenant < 1 or not 0 <= versioned_fraction <= 1:
        raise ValueError("Invalid world size or version fraction")
    rng = random.Random(seed)
    w = WorldLedger()

    principals = [
        Principal("alice","tenant_a",frozenset({"employee","engineering"})),
        Principal("bob","tenant_a",frozenset({"employee","finance"})),
        Principal("carol","tenant_b",frozenset({"employee","sales"})),
    ]
    for p in principals:
        w.add_principal(p)

    for tenant in range(tenants):
        for i in range(principals_per_tenant):
            domain = DOMAINS[i % len(DOMAINS)]
            w.add_principal(Principal(f"principal-{tenant:02d}-{i:04d}",
                                      f"tenant_{chr(97 + tenant)}",
                                      frozenset({"employee", domain})))

    for domain in DOMAINS:
        for i in range(docs_per_domain):
            logical_id = f"{domain.upper()}-{i:04d}"
            amount = 50 + ((i * 7) % 200)
            tenant = f"tenant_{chr(97 + (i % tenants))}"
            reference = f"{domain.upper()}-REF-{i:04d}"
            content = (f"{domain.title()} policy {i} reference {reference}. "
                       f"The current approved limit is ${amount}. "
                       f"This document governs {domain} operations and compliance requirements.")
            versioned = rng.random() < versioned_fraction
            doc = DocumentVersion(
                document_id=f"{logical_id}-v1",
                logical_document_id=logical_id,
                tenant_id=tenant,
                version=1,
                valid_from=0,
                valid_until=10 if versioned else None,
                authority_level=3,
                content=content,
                required_groups=frozenset({"employee"}),
            )
            w.add_document(doc)
            if versioned:
                w.add_document(DocumentVersion(
                    document_id=f"{logical_id}-v2", logical_document_id=logical_id,
                    tenant_id=tenant, version=2, valid_from=10,
                    supersedes=doc.document_id, authority_level=3,
                    content=(f"{domain.title()} policy {i} reference {reference}. "
                             f"The current approved limit is ${amount + 11}. "
                             f"This document governs {domain} operations and compliance requirements."),
                    required_groups=frozenset({"employee"}),
                ))

            qid = f"q-{logical_id}"
            w.set_query_gold(qid, {logical_id})

    # explicit sensitive document for ACL tests
    sensitive = DocumentVersion(
        document_id="FIN-SENSITIVE-v1",
        logical_document_id="FIN-SENSITIVE",
        tenant_id="tenant_a",
        version=1,
        valid_from=0,
        authority_level=5,
        content="Finance compensation policy. Project Helios adjustment is 12 percent.",
    )
    w.add_document(sensitive)
    w.set_query_gold("q-fin-sensitive", {"FIN-SENSITIVE"})
    w.add_permission(PermissionRule("alice","FIN-SENSITIVE", True, 0))
    w.add_permission(PermissionRule("bob","FIN-SENSITIVE", True, 0))
    return w
