"""Scenario manifests describe transitions and delivery faults, never outcomes."""
from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass

from ragstate.transitions.events import Transition
from ragstate.world.models import DocumentVersion


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    transition_type: str
    fault_type: str
    severity: int
    logical_document_id: str
    principal_id: str
    query: str
    domain: str
    events: tuple[Transition, ...]
    delays: tuple[tuple[str, str, int], ...] = ()
    drops: tuple[tuple[str, str], ...] = ()
    overrides: tuple[tuple[str, str, dict], ...] = ()
    stale_cache: bool = False
    initial_deny: bool = False
    replica_only_event_ids: tuple[str, ...] = ()
    retain_old_version: bool = False
    cache_owner_id: str | None = None

    def to_dict(self):
        return asdict(self)


TEMPLATES = (
    ("update_delay", "UPDATE", "delay"),
    ("supersede_partial", "SUPERSEDE", "partial"),
    ("delete_drop", "DELETE", "dropped_event"),
    ("delete_cache", "DELETE", "stale_cache_replay"),
    ("grant_delay", "GRANT", "delay"),
    ("revoke_delay", "REVOKE", "stale_acl_metadata"),
    ("supersede_cache", "SUPERSEDE", "stale_cache_replay"),
    ("revoke_cache", "REVOKE", "stale_cache_replay"),
    ("group_partial", "GROUP_REMOVE", "partial"),
    ("delete_replay", "DELETE", "out_of_order"),
    ("update_revoke", "UPDATE", "compound"),
    ("cross_tenant", "UPDATE", "stale_acl_metadata"),
    ("mixed_versions", "SUPERSEDE", "mixed_document_versions"),
    ("cross_principal", "UPDATE", "cross_principal_cache_reuse"),
    ("unrelated_update", "UPDATE", "none"),
    ("unrelated_acl", "REVOKE", "none"),
    ("unrelated_group", "GROUP_REMOVE", "none"),
    ("irrelevant_delete", "DELETE", "none"),
    ("harmless_refresh", "UPDATE", "none"),
    ("unrelated_invalidation", "UPDATE", "none"),
)


def generate_scenarios(world, *, transition_seed: int, fault_seed: int,
                       query_seed: int = 0,
                       limit: int | None = None) -> list[Scenario]:
    trng, frng, qrng = (random.Random(transition_seed), random.Random(fault_seed),
                       random.Random(query_seed))
    docs = [d for d in world.documents.values() if d.version == 1 and
            d.logical_document_id != "FIN-SENSITIVE" and d.valid_until is None]
    docs.sort(key=lambda x: x.document_id)
    trng.shuffle(docs)
    selected = docs[:len(TEMPLATES)]
    if len(selected) < len(TEMPLATES):
        raise ValueError(f"World needs at least {len(TEMPLATES)} unversioned documents")
    scenarios = []
    for i, ((name, kind, fault), doc) in enumerate(zip(TEMPLATES, selected)):
        principal = next(p for p in sorted(world.principals.values(), key=lambda x: x.principal_id)
                         if p.tenant_id == doc.tenant_id)
        target = doc
        # Negative controls alter another resource while querying a stable target.
        if name.startswith("unrelated") or name in {"irrelevant_delete", "harmless_refresh"}:
            target = next(d for d in docs if d.document_id != doc.document_id and
                          d.tenant_id == doc.tenant_id and
                          (name != "unrelated_invalidation" or
                           d.logical_document_id.split("-")[0] !=
                           doc.logical_document_id.split("-")[0]))
        reference = target.logical_document_id
        subject = f"{reference.split('-')[0]} policy {int(reference.split('-')[1])}"
        query = (f"What is the current approved limit for {subject}?" if qrng.randrange(2)
                 else f"For {subject}, state the current approved limit.")
        eid = f"{name}-event"
        payload = {}
        event_type = {"UPDATE": "DOCUMENT_UPDATE", "SUPERSEDE": "DOCUMENT_SUPERSEDE",
                      "DELETE": "DOCUMENT_DELETE", "GRANT": "PERMISSION_GRANT",
                      "REVOKE": "PERMISSION_REVOKE", "GROUP_REMOVE": "GROUP_REMOVE"}[kind]
        if kind in {"UPDATE", "SUPERSEDE"}:
            tenant = doc.tenant_id
            if name == "cross_tenant":
                tenant = "tenant_b" if tenant == "tenant_a" else "tenant_a"
            new = DocumentVersion(f"{doc.logical_document_id}-v2", doc.logical_document_id,
                                  tenant, 2, 10, supersedes=doc.document_id,
                                  authority_level=doc.authority_level,
                                  content=re.sub(r"\$(\d+)",
                                      lambda m: f"${int(m.group(1)) + 11}", doc.content, count=1))
            payload = asdict(new)
            if name == "harmless_refresh":
                event_type = "DOCUMENT_CREATE"
                payload = asdict(doc)
        elif kind in {"GRANT", "REVOKE"}:
            affected = principal.principal_id
            if name == "unrelated_acl":
                affected = next(p.principal_id for p in sorted(world.principals.values(),
                                key=lambda x: x.principal_id)
                                if p.tenant_id == doc.tenant_id and p.principal_id != affected)
            payload = {"principal_id": affected,
                       "logical_document_id": doc.logical_document_id}
        elif kind == "GROUP_REMOVE":
            affected = principal.principal_id
            if name == "unrelated_group":
                affected = next(p.principal_id for p in sorted(world.principals.values(),
                                key=lambda x: x.principal_id)
                                if p.tenant_id == doc.tenant_id and p.principal_id != affected)
            payload = {"principal_id": affected, "group_id": "employee"}
        events = [Transition(eid, 10, event_type, doc.document_id, payload)]
        delays = []
        drops = []
        overrides = []
        ticks = frng.randint(2, 4)
        if fault in {"delay", "partial", "compound", "stale_acl_metadata"}:
            plane = "auth" if kind in {"GRANT", "REVOKE", "GROUP_REMOVE"} or name == "cross_tenant" else "index"
            delays.append((plane, eid, ticks))
        if name == "supersede_cache":
            delays.append(("index", eid, ticks))
        if name == "revoke_cache":
            delays.append(("auth", eid, ticks))
        if fault == "dropped_event":
            drops.append(("index", eid))
            drops.append(("auth", eid))
        if name == "supersede_partial":
            # The replica receives an older payload at the scheduled delivery.
            overrides.append(("index", eid, {"content": doc.content}))
        if name == "group_partial":
            overrides.append(("auth", eid, {"group_id": "unrelated_group"}))
        if name == "update_revoke":
            rev = Transition(f"{name}-revoke", 10, "PERMISSION_REVOKE", doc.logical_document_id,
                             {"principal_id": principal.principal_id,
                              "logical_document_id": doc.logical_document_id})
            events.append(rev)
            delays.append(("auth", rev.event_id, ticks))
        if name == "delete_replay":
            replay = Transition(f"{name}-replay", 9, "DOCUMENT_CREATE", doc.document_id,
                                asdict(doc))
            events.append(replay)
            delays.extend((("index", eid, 2), ("auth", eid, 2),
                           ("index", replay.event_id, 5), ("auth", replay.event_id, 5)))
        if name == "cross_tenant":
            # Content changes at t=10; stale authorization still sees tenant A.
            delays = [("auth", eid, ticks)]
        owner = None
        if name == "cross_principal":
            owner = principal.principal_id
            principal = next(p for p in sorted(world.principals.values(),
                             key=lambda x: x.principal_id)
                             if p.tenant_id == doc.tenant_id and p.principal_id != owner)
        scenarios.append(Scenario(name, kind, fault, ticks if fault != "none" else 0,
                                  reference, principal.principal_id, query,
                                  reference.split("-")[0].lower(), tuple(events),
                                  tuple(delays), tuple(drops), tuple(overrides),
                                  name in {"supersede_cache", "revoke_cache", "delete_cache",
                                           "unrelated_invalidation"},
                                  name in {"grant_delay", "cross_principal"},
                                  (f"{name}-replay",) if name == "delete_replay" else (),
                                  name == "mixed_versions", owner))
    return scenarios[:limit] if limit is not None else scenarios
