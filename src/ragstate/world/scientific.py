"""Phase 7 pilot corpus and structured questions; separate from legacy fixtures.

The six domain blueprints and query allocation are fixed before pilot execution.
Only seed-driven selection and numeric facts vary. No retriever or oracle result is
consulted during construction.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .ledger import WorldLedger
from .models import DocumentVersion, Principal


@dataclass(frozen=True)
class DomainBlueprint:
    name: str
    document_types: tuple[str, ...]
    subject: str
    fact_label: str
    unit: str
    vocabulary: tuple[str, ...]
    base: int


BLUEPRINTS = (
    DomainBlueprint("hr", ("benefits eligibility", "leave administration", "onboarding exception"),
                    "employee benefit", "waiting period", "days",
                    ("enrollment", "eligibility", "manager approval", "payroll"), 30),
    DomainBlueprint("finance", ("expense approval", "treasury exception", "vendor payment"),
                    "spending approval", "approval threshold", "dollars",
                    ("invoice", "reconciliation", "controller review", "audit trail"), 500),
    DomainBlueprint("engineering", ("release rollback", "service ownership", "change review"),
                    "production release", "rollback window", "minutes",
                    ("deployment", "incident", "service owner", "release gate"), 20),
    DomainBlueprint("security", ("credential rotation", "access review", "device compliance"),
                    "security control", "rotation interval", "days",
                    ("identity", "key escrow", "attestation", "least privilege"), 40),
    DomainBlueprint("sales", ("renewal discount", "customer escalation", "contract exception"),
                    "customer renewal", "discount cap", "percent",
                    ("account team", "quote", "renewal", "customer success"), 8),
    DomainBlueprint("legal", ("record retention", "contract review", "regulatory notice"),
                    "compliance record", "retention period", "months",
                    ("counsel", "filing", "hold notice", "jurisdiction"), 18),
)


@dataclass(frozen=True)
class ScientificQuery:
    query_id: str
    text: str
    domain: str
    family: str
    principal_id: str
    tenant_id: str
    logical_document_ids: tuple[str, ...]
    transition_type: str
    target_document_id: str
    fact_label: str
    unit: str


FAMILIES = ("current_state", "version_sensitive", "authorization_sensitive",
            "multi_document", "negative", "transition_boundary")
TRANSITIONS = ("UPDATE", "SUPERSEDE", "DELETE", "GRANT", "REVOKE")


def generate_scientific_world(world_seed: int, docs_per_domain: int = 30,
                              tenants: int = 2, principals_per_tenant: int = 12):
    if docs_per_domain != 30 or tenants != 2 or principals_per_tenant < 6:
        raise ValueError("scientific-pilot-v1 corpus is fixed at 30 documents/domain, 2 tenants, >=6 principals/tenant")
    rng = random.Random(world_seed)
    world = WorldLedger()
    for tenant in range(tenants):
        for i in range(principals_per_tenant):
            domain = BLUEPRINTS[i % 6].name
            world.add_principal(Principal(f"pilot-p-{tenant}-{i:02d}",
                                          f"tenant_{chr(97 + tenant)}",
                                          frozenset({"employee", domain})))
    for blueprint in BLUEPRINTS:
        for i in range(docs_per_domain):
            logical = f"{blueprint.name.upper()}-{i:04d}"
            tenant = f"tenant_{chr(97 + i % tenants)}"
            document_type = blueprint.document_types[(i // 2) % len(blueprint.document_types)]
            value = blueprint.base + (i % 7) * 3 + rng.randrange(3)
            code = f"{blueprint.name.upper()}-{i:04d}"
            vocabulary = blueprint.vocabulary[i % len(blueprint.vocabulary)]
            # Adjacent records share type and near-identical language but have different
            # tenants, identifiers, and facts. The body differs across domains.
            content = (f"{document_type.title()} record {code} for {tenant}. "
                       f"For this {blueprint.subject}, the current {blueprint.fact_label} "
                       f"is {value} {blueprint.unit}. "
                       f"The {vocabulary} workflow applies to record {code}.")
            groups = frozenset({blueprint.name}) if i % 6 == 2 else frozenset({"employee"})
            world.add_document(DocumentVersion(
                document_id=f"{logical}-v1", logical_document_id=logical,
                tenant_id=tenant, version=1, valid_from=0,
                content=content, authority_level=3, required_groups=groups))
    return world


def generate_scientific_queries(world, query_seed: int):
    rng = random.Random(query_seed)
    queries = []
    for blueprint in BLUEPRINTS:
        for i in range(30):
            logical = f"{blueprint.name.upper()}-{i:04d}"
            doc = world.document_at(logical, 9)
            transition = TRANSITIONS[i // 6]
            family = FAMILIES[i % 6]
            candidates = sorted(p.principal_id for p in world.principals.values()
                                if p.tenant_id == doc.tenant_id and blueprint.name in p.groups)
            principal = candidates[rng.randrange(len(candidates))]
            if family == "negative":
                principal = next(p.principal_id for p in sorted(world.principals.values(),
                                 key=lambda p: p.principal_id)
                                 if p.tenant_id != doc.tenant_id)
            required = [logical]
            if family == "multi_document":
                companion = f"{blueprint.name.upper()}-{(i + 2) % 30:04d}"
                required.append(companion)
            subject = " and ".join(required)
            if family == "transition_boundary":
                text = (f"After the latest change, what is the current {blueprint.fact_label} "
                        f"for {subject}?")
            elif family == "version_sensitive":
                text = (f"Which {blueprint.fact_label} now governs {subject} "
                        f"in {blueprint.subject} records?")
            elif family == "authorization_sensitive":
                text = (f"For my access, what {blueprint.fact_label} applies to "
                        f"{subject} in the {blueprint.vocabulary[i % 4]} workflow?")
            elif family == "negative":
                text = (f"What {blueprint.fact_label} is recorded for restricted "
                        f"{subject} in {blueprint.subject} operations?")
            else:
                text = (f"What is the current {blueprint.fact_label} for {subject} "
                        f"in {blueprint.subject} operations?")
            queries.append(ScientificQuery(
                query_id=f"pilot-q-{blueprint.name}-{i:02d}", text=text,
                domain=blueprint.name, family=family, principal_id=principal,
                tenant_id=world.principals[principal].tenant_id,
                logical_document_ids=tuple(required), transition_type=transition,
                target_document_id=doc.document_id,
                fact_label=blueprint.fact_label, unit=blueprint.unit))
    return queries
