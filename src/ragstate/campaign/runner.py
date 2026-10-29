"""Isolated transition campaigns with canonical and replica state kept separate."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import re
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from ragstate.faults.scheduler import FaultSchedule
from ragstate.oracle.metrics import convergence_time, exposure_window
from ragstate.replicas.simulator import ReplicaSimulator
from ragstate.retrieval.bm25 import BM25Retriever
from ragstate.retrieval.chunking import chunk_document
from ragstate.retrieval.dense import DenseRetriever, SentenceTransformerEncoder
from ragstate.retrieval.hybrid import HybridRetriever
from ragstate.retrieval.indexing import build_chunks_from_ledger
from ragstate.retrieval.metrics import (authorized_retrieval_denial,
    cross_tenant_exposure, current_evidence_recall, deleted_evidence_persistence,
    invalid_evidence_absence_success, mixed_version_context_rate,
    state_consistent_retrieval_success, superseded_retrieval_rate,
    unauthorized_context_rate)
from ragstate.transitions.engine import TransitionEngine
from ragstate.world.generator import generate_world
from ragstate.world.models import PermissionRule
from ragstate.world.queries import generate_queries
from .scenarios import generate_scenarios

BENCHMARK_VERSION = "0.1.1"
CSV_FIELDS = ("campaign_id", "campaign_version", "scenario_id", "query_id", "principal_id",
              "domain", "logical_time", "transition_type", "fault_type", "severity",
              "architecture", "retriever", "top_k", "scrs", "cer_at_k", "srr", "ucer",
              "candidate_cer_at_k", "post_filter_cer_at_k", "context_cer_at_k",
              "candidate_ucer", "post_filter_ucer", "context_ucer",
              "invalid_evidence_absence_success",
              "ardr", "mvcr", "deleted_evidence_present", "cross_tenant_exposure",
              "candidate_deleted_evidence_present", "post_filter_deleted_evidence_present",
              "convergence_tick", "rew", "t_rsc", "serr", "latency_ms", "world_seed",
              "query_seed", "transition_seed", "fault_seed", "canonical_state_hash",
              "index_state_hash", "auth_state_hash", "cache_state_hash")


def stable_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=_json_default).encode()).hexdigest()


def _json_default(value):
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def ledger_snapshot(ledger, tick: int) -> dict:
    logical = sorted({d.logical_document_id for d in ledger.documents.values()})
    return {"documents": {x: asdict(d) if (d := ledger.document_at(x, tick)) else None
                          for x in logical},
            "principals": {x: asdict(p) for x, p in sorted(ledger.principals.items())},
            "permissions": [asdict(p) for p in ledger.permissions if p.active(tick)],
            "group_events": [x for x in ledger.group_events if x[0] <= tick]}


class LedgerReplica:
    def __init__(self, ledger, plane: str, retained_logical_ids=()):
        self.ledger, self.plane = ledger, plane
        self.retained_logical_ids = frozenset(retained_logical_ids)
        self.applied = []

    def apply(self, event):
        content = event.event_type.startswith("DOCUMENT_")
        if self.plane == "auth" or content:
            TransitionEngine(self.ledger).apply(event)
            self.applied.append(event.event_id)

    def chunks_at(self, tick):
        chunks = build_chunks_from_ledger(self.ledger, tick)
        for logical_id in sorted(self.retained_logical_ids):
            for doc in sorted(self.ledger.documents.values(), key=lambda d: d.document_id):
                if (doc.logical_document_id == logical_id and doc.version == 1 and
                        tick >= 10 and self.ledger.document_at(logical_id, tick) != doc):
                    chunks.extend(chunk_document(doc))
        return chunks

    def state_hash(self, tick):
        state = ledger_snapshot(self.ledger, tick)
        if self.plane == "index":
            state["indexed_chunks"] = [(c.chunk_id, c.text) for c in self.chunks_at(tick)]
        return stable_hash(state)


def _hits(hits):
    return [{"chunk": asdict(h.chunk), "score": h.score,
             "components": h.components} for h in hits]


def _retriever(chunks, config, encoder=None):
    method = config["retriever"]
    sparse = BM25Retriever(chunks)
    if method == "bm25":
        return sparse
    dense = DenseRetriever(chunks, model_name=config.get("embedding_model",
        "sentence-transformers/all-MiniLM-L6-v2"),
        model_version=config.get("embedding_model_version", "unspecified"),
        cache_dir=config.get("embedding_cache_dir", ".cache/ragstate"),
        encoder=encoder)
    if method == "dense":
        return dense
    if method == "hybrid":
        return HybridRetriever(sparse, dense)
    raise ValueError(f"Unknown retriever {method}")


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _make_schedule(scenario):
    schedule = FaultSchedule()
    for plane, event_id, ticks in scenario.delays:
        schedule.schedule_delay(plane, event_id, ticks)
    for plane, event_id in scenario.drops:
        schedule.schedule_drop(plane, event_id)
    for plane, event_id, payload in scenario.overrides:
        schedule.schedule_payload_override(plane, event_id, payload)
    return schedule


def evaluate_scenario(base_world, scenario, config, provenance, encoder=None):
    canonical = copy.deepcopy(base_world)
    index = LedgerReplica(copy.deepcopy(base_world), "index",
                          [scenario.logical_document_id] if scenario.retain_old_version else ())
    auth = LedgerReplica(copy.deepcopy(base_world), "auth")
    if scenario.initial_deny:
        for ledger in (canonical, index.ledger, auth.ledger):
            ledger.add_permission(PermissionRule(scenario.principal_id,
                                                 scenario.logical_document_id, False, 0))
    schedule = _make_schedule(scenario)
    simulator = ReplicaSimulator({"index": index, "auth": auth}, schedule)
    for event in scenario.events:
        if event.event_id not in scenario.replica_only_event_ids:
            TransitionEngine(canonical).apply(event)
        simulator.submit(event)
    ticks = config.get("observation_ticks", [9, 10, 11, 12, 13, 14, 15])
    top_k = config.get("top_k", 10)
    architecture = config["architecture"]
    rows = []
    cached_context = None
    cached_claims = None
    cache_epoch = 0
    retriever_cache = {}
    resurrected = False
    latest_source_time = max(e.logical_time for e in scenario.events
                             if e.event_id not in scenario.replica_only_event_ids)
    for tick in ticks:
        started = time.perf_counter()
        delivered = simulator.run_until(tick)
        if any(d.replica_name == "index" and
               d.event.event_type in {"DOCUMENT_CREATE", "DOCUMENT_UPDATE", "DOCUMENT_SUPERSEDE"} and
               d.event.logical_time < latest_source_time for d in delivered):
            if (canonical.document_at(scenario.logical_document_id, tick) is None and
                    index.ledger.document_at(scenario.logical_document_id, tick) is not None):
                resurrected = True
        if architecture == "A1" and tick >= 10 and tick % config.get("sync_interval", 3) == 0:
            index.ledger = copy.deepcopy(canonical)
            auth.ledger = copy.deepcopy(canonical)
            cache_epoch += 1
            cached_context = None
            cached_claims = None
        relevant = any(d.event.resource_id in {scenario.logical_document_id,
            next((x.document_id for x in base_world.documents.values()
                  if x.logical_document_id == scenario.logical_document_id), "")}
            or d.event.payload.get("principal_id") == scenario.principal_id for d in delivered)
        if delivered and (architecture in {"A2", "A5"} or
                          (architecture == "A3" and relevant)):
            cache_epoch += 1
            cached_context = None
            cached_claims = None
        chunks = index.chunks_at(tick)
        chunk_key = stable_hash([(c.chunk_id, c.text) for c in chunks])
        if chunk_key not in retriever_cache:
            retriever_cache[chunk_key] = _retriever(chunks, config, encoder)
        raw = retriever_cache[chunk_key].retrieve(scenario.query, top_k)
        if architecture in {"A4", "A5"}:
            allowed = lambda h: canonical.allowed(scenario.principal_id,
                                                   h.chunk.logical_document_id, tick)
        else:
            allowed = lambda h: auth.ledger.allowed(scenario.principal_id,
                                                     h.chunk.logical_document_id, tick)
        filtered = [h for h in raw if allowed(h)]
        if architecture == "A5":
            filtered = [h for h in filtered if (current := canonical.document_at(
                h.chunk.logical_document_id, tick)) is not None and
                current.document_id == h.chunk.document_id and h.chunk.text in current.content]
        context = filtered
        if scenario.cache_owner_id and tick == ticks[0]:
            cached_context = [h for h in raw if auth.ledger.allowed(
                scenario.cache_owner_id, h.chunk.logical_document_id, tick)]
            cached_claims = sorted({claim for h in cached_context
                                    for claim in re.findall(r"\$\d+", h.chunk.text)})
        elif scenario.cache_owner_id and architecture in {"A0", "A1"}:
            context = cached_context or filtered
        elif scenario.stale_cache and architecture in {"A0", "A1"}:
            if cached_context is None:
                cached_context = filtered
                cached_claims = sorted({claim for h in cached_context
                                        for claim in re.findall(r"\$\d+", h.chunk.text)})
            else:
                context = cached_context
        elif scenario.stale_cache and cached_context is None:
            cached_context = filtered
            cached_claims = sorted({claim for h in cached_context
                                    for claim in re.findall(r"\$\d+", h.chunk.text)})
        doc = canonical.document_at(scenario.logical_document_id, tick)
        required = {doc.document_id} if doc and canonical.allowed(
            scenario.principal_id, scenario.logical_document_id, tick) else set()
        current = {d.document_id for lid in {x.logical_document_id for x in canonical.documents.values()}
                   if (d := canonical.document_at(lid, tick))}
        deleted = deleted_evidence_persistence(context, canonical, tick)
        candidate_deleted = deleted_evidence_persistence(raw, canonical, tick)
        filtered_deleted = deleted_evidence_persistence(filtered, canonical, tick)
        cross_tenant = cross_tenant_exposure(context, scenario.principal_id, canonical)
        candidate_unauthorized = unauthorized_context_rate(raw, scenario.principal_id, canonical, tick)
        filtered_unauthorized = unauthorized_context_rate(filtered, scenario.principal_id, canonical, tick)
        unauthorized = unauthorized_context_rate(context, scenario.principal_id, canonical, tick)
        absence = invalid_evidence_absence_success(
            context, required, {scenario.logical_document_id})
        # SCRS requires current, authorized context as well as required evidence.
        scrs = state_consistent_retrieval_success(context, required,
                                                   scenario.principal_id, canonical, tick)
        row = {**provenance, "scenario_id": scenario.scenario_id,
               "query_id": scenario.scenario_id + "-query", "query": scenario.query,
               "principal_id": scenario.principal_id, "domain": scenario.domain,
               "logical_time": tick, "transition_type": scenario.transition_type,
               "fault_type": scenario.fault_type, "severity": scenario.severity,
               "architecture": architecture, "retriever": config["retriever"],
               "top_k": top_k, "scrs": int(scrs),
               "cer_at_k": current_evidence_recall(context, required),
               "srr": superseded_retrieval_rate(raw, current, canonical, tick),
               "ucer": unauthorized,
               "ardr": int(authorized_retrieval_denial(required, context)),
               "mvcr": mixed_version_context_rate(context),
               "deleted_evidence_present": int(deleted),
               "candidate_deleted_evidence_present": int(candidate_deleted),
               "post_filter_deleted_evidence_present": int(filtered_deleted),
               "cross_tenant_exposure": int(cross_tenant),
               "serr": int(resurrected),
               "candidate_ucer": candidate_unauthorized,
               "post_filter_ucer": filtered_unauthorized,
               "context_ucer": unauthorized,
               "invalid_evidence_absence_success": absence,
               "candidate_cer_at_k": current_evidence_recall(raw, required),
               "post_filter_cer_at_k": current_evidence_recall(filtered, required),
               "context_cer_at_k": current_evidence_recall(context, required),
               "latency_ms": round((time.perf_counter() - started) * 1000, 3),
               "canonical_state_hash": stable_hash(ledger_snapshot(canonical, tick)),
               "index_state_hash": index.state_hash(tick),
               "auth_state_hash": auth.state_hash(tick),
               "cache_state_hash": stable_hash({"epoch": cache_epoch,
                    "query": scenario.query,
                    "owner": scenario.cache_owner_id or scenario.principal_id,
                    "chunks": [h.chunk.chunk_id for h in cached_context or []],
                    "claims": cached_claims or []}),
               "cache_entry": None if cached_context is None else {
                   "kind": "derived_claims", "principal_id": scenario.cache_owner_id or scenario.principal_id,
                   "claims": cached_claims or [],
                   "source_chunk_ids": [h.chunk.chunk_id for h in cached_context]},
               "replica_state_hashes": {"index": index.state_hash(tick),
                                          "auth": auth.state_hash(tick)},
               "transition": [asdict(e) for e in scenario.events],
               "fault_schedule": {"delays": scenario.delays, "drops": scenario.drops,
                                  "overrides": scenario.overrides},
               "retrieved_chunks": _hits(raw), "filtered_chunks": _hits(filtered),
               "context_chunks": _hits(context), "gold_evidence": sorted(required),
               "gold": {"required_logical_documents": [scenario.logical_document_id],
                        "required_document_versions": sorted(required),
                        "required_atomic_claims": re.findall(r"\$\d+", doc.content)
                            if required and doc else [],
                        "forbidden_claims": [] if required else
                            sorted({d.document_id for d in canonical.documents.values()
                                    if d.logical_document_id == scenario.logical_document_id}),
                        "principal_id": scenario.principal_id,
                        "logical_time": tick,
                        "answerable": bool(required)},
               "metrics": {"scrs": int(scrs), "cer_at_k": current_evidence_recall(context, required),
                           "srr": superseded_retrieval_rate(raw, current, canonical, tick),
                           "ucer": unauthorized,
                           "candidate_ucer": candidate_unauthorized,
                           "post_filter_ucer": filtered_unauthorized,
                           "context_ucer": unauthorized,
                           "invalid_evidence_absence_success": absence}}
        rows.append(row)
    convergence = convergence_time(10, [(r["logical_time"], bool(r["scrs"])) for r in rows])
    exposures = [r["logical_time"] for r in rows if r["ucer"] > 0]
    for row in rows:
        row["convergence_tick"] = None if convergence is None else 10 + convergence
        row["t_rsc"] = convergence
        row["rew"] = exposure_window(10, exposures) if scenario.transition_type == "REVOKE" else 0
    return rows


def aggregate(rows):
    post = [r for r in rows if r["logical_time"] >= 10]
    grouped = {}
    for r in post:
        key = (r["architecture"], r["retriever"], r["fault_type"])
        grouped.setdefault(key, []).append(r)
    fields = ("scrs", "cer_at_k", "srr", "ucer", "ardr", "mvcr",
              "deleted_evidence_present", "cross_tenant_exposure", "serr",
              "invalid_evidence_absence_success")
    def mean_non_null(group, field):
        values = [row[field] for row in group if row.get(field) is not None]
        return sum(values) / len(values) if values else None
    return [{"architecture": k[0], "retriever": k[1], "fault_type": k[2],
             "n": len(group),
             **{f"mean_{field}": mean_non_null(group, field) for field in fields}}
            for k, group in sorted(grouped.items())]


def _write_outputs(out: Path, rows, status):
    out.mkdir(parents=True, exist_ok=True)
    for name, content in (("results.jsonl", "".join(json.dumps(r, sort_keys=True,
                         default=_json_default) + "\n" for r in rows)),
                          ("status.json", json.dumps(status, indent=2, sort_keys=True))):
        tmp = out / (name + ".tmp")
        tmp.write_text(content)
        tmp.replace(out / name)
    with (out / "summary.csv.tmp").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({k: r.get(k) for k in CSV_FIELDS} for r in rows)
    (out / "summary.csv.tmp").replace(out / "summary.csv")
    with (out / "aggregate.csv.tmp").open("w", newline="") as f:
        summary = aggregate(rows)
        if summary:
            writer = csv.DictWriter(f, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    (out / "aggregate.csv.tmp").replace(out / "aggregate.csv")


def run_campaign(config: dict, output: str | Path):
    if config.get("architecture") not in {"A0", "A1", "A2", "A3", "A4", "A5"}:
        raise ValueError("architecture must be A0 through A5")
    if config.get("retriever") not in {"bm25", "dense", "hybrid"}:
        raise ValueError("retriever must be bm25, dense, or hybrid")
    ticks = config.get("observation_ticks", [9, 10, 11, 12, 13, 14, 15])
    if sorted(set(ticks)) != ticks or not ticks or ticks[0] >= 10 or ticks[-1] < 10:
        raise ValueError("observation_ticks must be increasing and straddle tick 10")
    if config.get("top_k", 10) < 1 or config.get("sync_interval", 3) < 1:
        raise ValueError("top_k and sync_interval must be positive")
    out = Path(output)
    world = generate_world(config["world_seed"], config.get("docs_per_domain", 20),
                           tenants=config.get("tenants", 2),
                           principals_per_tenant=config.get("principals_per_tenant", 3))
    scenarios = generate_scenarios(world, transition_seed=config["transition_seed"],
                                   fault_seed=config["fault_seed"],
                                   query_seed=config["query_seed"],
                                   limit=config.get("scenario_limit"))
    manifest = {"benchmark_version": BENCHMARK_VERSION, "config": config,
                "scenarios": [s.to_dict() for s in scenarios]}
    manifest_hash = stable_hash(manifest)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and stable_hash(json.loads(manifest_path.read_text())) != manifest_hash:
        raise ValueError("Output directory contains a different campaign manifest")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=_json_default))
    rows = ([json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
            if (out / "results.jsonl").exists() else [])
    status = json.loads((out / "status.json").read_text()) if (out / "status.json").exists() else {}
    provenance = {"benchmark_version": BENCHMARK_VERSION,
                  "campaign_version": config.get("campaign_version", "pilot-1"),
                  "campaign_id": config.get("campaign_id", "pilot"),
                  "git_commit": _git_commit(),
                  "world_seed": config["world_seed"], "query_seed": config["query_seed"],
                  "transition_seed": config["transition_seed"], "fault_seed": config["fault_seed"],
                  "embedding_model": config.get("embedding_model") if config["retriever"] != "bm25" else None,
                  "embedding_model_version": config.get("embedding_model_version")
                    if config["retriever"] != "bm25" else None}
    encoder = None
    if config["retriever"] in {"dense", "hybrid"} and any(
        status.get(s.scenario_id, {}).get("state") != "complete" for s in scenarios):
        revision = config.get("embedding_model_version")
        if not revision:
            raise ValueError("Dense campaigns require embedding_model_version revision")
        encoder = SentenceTransformerEncoder(config.get("embedding_model",
            "sentence-transformers/all-MiniLM-L6-v2"), revision)
    for scenario in scenarios:
        if status.get(scenario.scenario_id, {}).get("state") == "complete":
            continue
        try:
            new_rows = evaluate_scenario(world, scenario, config, provenance, encoder)
            rows = [r for r in rows if r["scenario_id"] != scenario.scenario_id] + new_rows
            status[scenario.scenario_id] = {"state": "complete", "rows": len(new_rows)}
        except Exception as exc:
            status[scenario.scenario_id] = {"state": "failed", "reason": str(exc)}
            _write_outputs(out, rows, status)
            raise
        _write_outputs(out, rows, status)
    return rows


def validate_clean(config):
    world = generate_world(config["world_seed"], config.get("docs_per_domain", 20),
                           tenants=config.get("tenants", 2),
                           principals_per_tenant=config.get("principals_per_tenant", 3),
                           versioned_fraction=config.get("versioned_fraction", 0.25))
    tick = config.get("clean_tick", 10)
    queries = generate_queries(world, query_seed=config["query_seed"], logical_time=tick,
                               count=config.get("clean_query_count", 30))
    retriever = _retriever(build_chunks_from_ledger(world, tick), config)
    scores = []
    for q in queries:
        if not q.answerable:
            continue
        hits = [h for h in retriever.retrieve(q.text, 10)
                if world.allowed(q.principal_id, h.chunk.logical_document_id, tick)]
        scores.append(current_evidence_recall(hits, set(q.required_document_versions)))
    mean = sum(scores) / len(scores) if scores else 0.0
    return {"cer_at_10": mean, "answerable_queries": len(scores),
            "pass": bool(scores) and mean >= 0.90}
