"""Control gates check observable failure signals without supplying retrieval answers."""
from __future__ import annotations


def validate_controls(rows):
    by_id = {}
    for row in rows:
        if row["logical_time"] >= 10:
            by_id.setdefault(row["scenario_id"], []).append(row)
    positive = {
        "update_delay": lambda rs: any(r["srr"] > 0 and r["scrs"] == 0 for r in rs),
        "supersede_partial": lambda rs: any(r["scrs"] == 0 for r in rs),
        "delete_drop": lambda rs: any(r["deleted_evidence_present"] for r in rs),
        "delete_cache": lambda rs: any(r["deleted_evidence_present"] for r in rs),
        "grant_delay": lambda rs: any(r["ardr"] for r in rs),
        "revoke_delay": lambda rs: any(r["ucer"] > 0 for r in rs),
        "supersede_cache": lambda rs: any(r["logical_time"] >= 13 and r["scrs"] == 0
                                            for r in rs),
        "revoke_cache": lambda rs: any(r["ucer"] > 0 for r in rs),
        "group_partial": lambda rs: any(r["ucer"] > 0 for r in rs),
        "delete_replay": lambda rs: any(r["serr"] for r in rs),
        "update_revoke": lambda rs: any(r["ucer"] > 0 and r["srr"] > 0 for r in rs),
        "cross_tenant": lambda rs: any(r["cross_tenant_exposure"] for r in rs),
        "mixed_versions": lambda rs: any(r["mvcr"] > 0 for r in rs),
        "cross_principal": lambda rs: any(r["ucer"] > 0 for r in rs),
    }
    negative = ("unrelated_update", "unrelated_acl", "unrelated_group",
                "irrelevant_delete", "harmless_refresh", "unrelated_invalidation")
    checks = {name: bool(by_id.get(name)) and predicate(by_id[name])
              for name, predicate in positive.items()}
    checks.update({name: bool(by_id.get(name)) and all(
        r["scrs"] == 1 and r["ucer"] == 0 and r["ardr"] == 0 and
        r["deleted_evidence_present"] == 0 and r["cross_tenant_exposure"] == 0
        for r in by_id[name]) for name in negative})
    return {"pass": all(checks.values()), "checks": checks}
