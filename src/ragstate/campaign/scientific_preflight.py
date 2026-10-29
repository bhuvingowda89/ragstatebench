"""Frozen clean integrity check before the Phase 7 state-transition campaign."""
from __future__ import annotations

import csv
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path

from ragstate.retrieval.bm25 import BM25Retriever
from ragstate.retrieval.dense import DenseRetriever, SentenceTransformerEncoder
from ragstate.retrieval.hybrid import HybridRetriever
from ragstate.retrieval.indexing import build_chunks_from_ledger
from ragstate.retrieval.metrics import (current_evidence_recall,
    invalid_evidence_absence_success, state_consistent_retrieval_success)
from ragstate.world.scientific import generate_scientific_queries, generate_scientific_world
from .runner import BENCHMARK_VERSION, _git_commit, ledger_snapshot, stable_hash


def _p95(values):
    values = sorted(values)
    return values[math.ceil(0.95 * len(values)) - 1]


def run_preflight(config, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Preflight output already exists; previous artifacts are immutable")
    if config["top_k"] != 10 or config["docs_per_domain"] != 30:
        raise ValueError("Frozen scientific preflight requires top_k=10 and 30 documents/domain")
    snapshot = Path(config["embedding_snapshot"])
    if not snapshot.is_dir() or snapshot.name != config["embedding_revision"]:
        raise ValueError("Pinned local MiniLM snapshot is unavailable")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    import sentence_transformers
    torch.manual_seed(0)
    torch.set_num_threads(1)
    world = generate_scientific_world(config["world_seed"], config["docs_per_domain"],
                                      config["tenants"], config["principals_per_tenant"])
    queries = generate_scientific_queries(world, config["query_seed"])
    chunks = build_chunks_from_ledger(world, 9)
    model = SentenceTransformerEncoder(str(snapshot), config["embedding_revision"])
    sparse = BM25Retriever(chunks)
    dense = DenseRetriever(chunks, model_name=config["embedding_model"],
                           model_version=config["embedding_revision"], encoder=model)
    retrievers = {"bm25": sparse, "dense": dense, "hybrid": HybridRetriever(sparse, dense)}
    rows = []
    for q in queries:
        required = {world.document_at(lid, 9).document_id for lid in q.logical_document_ids
                    if world.allowed(q.principal_id, lid, 9)}
        for name, retriever in retrievers.items():
            started = time.perf_counter()
            candidate = retriever.retrieve(q.text, config["top_k"])
            latency = (time.perf_counter() - started) * 1000
            context = [hit for hit in candidate if world.allowed(
                q.principal_id, hit.chunk.logical_document_id, 9)]
            rows.append({"query_id": q.query_id, "domain": q.domain, "family": q.family,
                         "principal_id": q.principal_id, "tenant_id": q.tenant_id,
                         "retriever": name, "logical_time": 9,
                         "answerable": int(bool(required)),
                         "required_document_ids": json.dumps(sorted(required)),
                         "cer_at_10": current_evidence_recall(context, required),
                         "scrs": int(state_consistent_retrieval_success(
                             context, required, q.principal_id, world, 9)),
                         "invalid_evidence_absence_success": invalid_evidence_absence_success(
                             context, required, set(q.logical_document_ids)),
                         "latency_ms": round(latency, 3)})
    by_group = {}
    for row in rows:
        if row["answerable"]:
            by_group.setdefault((row["retriever"], row["domain"]), []).append(row)
    clean = []
    for (method, domain), group in sorted(by_group.items()):
        latency = [row["latency_ms"] for row in group]
        clean.append({"retriever": method, "domain": domain,
                      "answerable_queries": len(group),
                      "cer_at_10": sum(row["cer_at_10"] for row in group) / len(group),
                      "scrs": sum(row["scrs"] for row in group) / len(group),
                      "median_query_ms": statistics.median(latency),
                      "p95_query_ms": _p95(latency)})
    overall = []
    for method in retrievers:
        group = [row for row in rows if row["retriever"] == method and row["answerable"]]
        overall.append({"retriever": method, "answerable_queries": len(group),
                        "cer_at_10": sum(row["cer_at_10"] for row in group) / len(group),
                        "scrs": sum(row["scrs"] for row in group) / len(group)})
    config_hash = stable_hash(config)
    summary = {"campaign_id": "scientific-pilot-v1-preflight",
               "benchmark_version": BENCHMARK_VERSION,
               "campaign_version": config["campaign_version"],
               "config_hash": config_hash, "git_commit": _git_commit(),
               "python": sys.version.split()[0], "platform": platform.platform(),
               "sentence_transformers": sentence_transformers.__version__,
               "torch": torch.__version__, "device": "cpu",
               "embedding_model": config["embedding_model"],
               "embedding_revision": config["embedding_revision"],
               "seeds": {key: config[key] for key in ("world_seed", "query_seed",
                                                     "transition_seed", "fault_seed")},
               "canonical_state_hash": stable_hash(ledger_snapshot(world, 9)),
               "base_questions": len(queries), "indexed_chunks": len(chunks),
               "overall": overall, "by_domain": clean,
               "stop_condition": "Clean retrieval integrity is poor for dense and uneven for hybrid; no fault campaign executed."}
    output.mkdir(parents=True)
    (output / "manifest.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    with (output / "per_query.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = run_preflight(json.loads(Path(args.config).read_text()), args.output)
    print(json.dumps({"base_questions": summary["base_questions"],
                      "indexed_chunks": summary["indexed_chunks"],
                      "overall": summary["overall"],
                      "stop_condition": summary["stop_condition"]}, indent=2))


if __name__ == "__main__":
    main()
