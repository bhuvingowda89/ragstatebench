"""Structured, state-sensitive questions generated independently of retrieval."""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkQuery:
    query_id: str
    family: str
    text: str
    principal_id: str
    logical_time: int
    required_logical_documents: tuple[str, ...]
    required_document_versions: tuple[str, ...]
    required_atomic_claims: tuple[str, ...]
    forbidden_claims: tuple[str, ...]
    answerable: bool
    domain: str


def generate_queries(world, *, query_seed: int, logical_time: int = 10,
                     count: int = 30) -> list[BenchmarkQuery]:
    if len({p.tenant_id for p in world.principals.values()}) < 2:
        raise ValueError("State-sensitive negative queries require at least two tenants")
    rng = random.Random(query_seed)
    logical_ids = sorted({d.logical_document_id for d in world.documents.values()
                          if d.logical_document_id != "FIN-SENSITIVE"})
    rng.shuffle(logical_ids)
    families = ("snapshot", "authorization", "multi_document",
                "transition_boundary", "negative")
    out = []
    for i in range(count):
        family = families[i % len(families)]
        lid = logical_ids[i % len(logical_ids)]
        doc = world.document_at(lid, logical_time)
        if doc is None:
            continue
        candidates = sorted(p.principal_id for p in world.principals.values()
                            if p.tenant_id == doc.tenant_id)
        principal = candidates[i % len(candidates)]
        required = (lid,)
        if family == "multi_document":
            other = next((x for x in logical_ids if x != lid and
                          world.document_at(x, logical_time) is not None and
                          world.document_at(x, logical_time).tenant_id == doc.tenant_id), None)
            if other:
                required = (lid, other)
        if family == "negative":
            principal = next(p.principal_id for p in sorted(world.principals.values(),
                             key=lambda x: x.principal_id) if p.tenant_id != doc.tenant_id)
        docs = [world.document_at(x, logical_time) for x in required]
        allowed = all(world.allowed(principal, x, logical_time) for x in required)
        tokens = [f"{x.logical_document_id.split('-')[0]} policy {int(x.logical_document_id.split('-')[1])}"
                  for x in docs]
        text = "What is the current approved limit for " + " and ".join(tokens) + "?"
        if family == "transition_boundary":
            text = "After the latest policy transition, " + text[0].lower() + text[1:]
        out.append(BenchmarkQuery(
            query_id=f"query-{query_seed}-{i:05d}", family=family, text=text,
            principal_id=principal, logical_time=logical_time,
            required_logical_documents=required,
            required_document_versions=tuple(x.document_id for x in docs) if allowed else (),
            required_atomic_claims=tuple(x.content.split("The current approved limit is ")[1].split(".")[0]
                                         for x in docs) if allowed else (),
            forbidden_claims=() if allowed else tuple(x.document_id for x in docs),
            answerable=allowed, domain=lid.split("-")[0].lower(),
        ))
    return out
