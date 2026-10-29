"""Fixed clean-query CPU smoke using the production dense and hybrid adapters."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

from ragstate.retrieval.bm25 import BM25Retriever
from ragstate.retrieval.dense import DenseRetriever, SentenceTransformerEncoder
from ragstate.retrieval.hybrid import HybridRetriever
from ragstate.retrieval.indexing import build_chunks_from_ledger
from ragstate.retrieval.metrics import current_evidence_recall
from ragstate.world.generator import generate_world
from ragstate.world.queries import generate_queries

COMPARISON_FIELDS = ("query_id", "gold_document_id", "bm25_rank", "dense_rank",
                     "hybrid_rank", "bm25_hit_at_10", "dense_hit_at_10",
                     "hybrid_hit_at_10")
PER_QUERY_FIELDS = COMPARISON_FIELDS + ("principal_id", "dense_score", "hybrid_score",
                                    "dense_query_ms", "hybrid_query_ms")


def _p95(values):
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def _rank(hits, document_id):
    return next(((i, hit.score) for i, hit in enumerate(hits, 1)
                 if hit.chunk.document_id == document_id), (None, None))


def run_smoke(config, *, model_revision: str, encoder=None, model_path: str | None = None):
    """One cold index build and warm-query pass on the unchanged BM25 clean set."""
    model_name = config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    if model_path is not None:
        if not Path(model_path).is_dir():
            raise ValueError(f"Local model snapshot not found: {model_path}")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    world = generate_world(config["world_seed"], config.get("docs_per_domain", 20),
                           tenants=config.get("tenants", 2),
                           principals_per_tenant=config.get("principals_per_tenant", 3),
                           versioned_fraction=config.get("versioned_fraction", 0.25))
    tick = config.get("clean_tick", 10)
    chunks = build_chunks_from_ledger(world, tick)
    queries = [q for q in generate_queries(world, query_seed=config["query_seed"],
               logical_time=tick, count=config.get("clean_query_count", 30)) if q.answerable]
    if not queries:
        raise ValueError("No answerable clean-gate queries")
    model_load_ms = 0.0
    package_versions = {"python": sys.version.split()[0],
                        "sentence_transformers": None, "torch": None}
    if encoder is None:
        try:
            import torch
            import sentence_transformers
        except ImportError as exc:
            raise RuntimeError("Install ragstatebench[dense] before real CPU smoke validation") from exc
        torch.manual_seed(0)
        torch.set_num_threads(1)
        package_versions["torch"] = torch.__version__
        package_versions["sentence_transformers"] = sentence_transformers.__version__
        started = time.perf_counter()
        encoder = SentenceTransformerEncoder(model_path or model_name, model_revision)
        model_load_ms = (time.perf_counter() - started) * 1000
    sparse = BM25Retriever(chunks)
    with tempfile.TemporaryDirectory(prefix="ragstate-smoke-") as cache:
        started = time.perf_counter()
        dense = DenseRetriever(chunks, model_name=model_name,
                               model_version=model_revision, encoder=encoder,
                               cache_dir=cache)
        encoding_ms = (time.perf_counter() - started) * 1000
        hybrid = HybridRetriever(sparse, dense)
        # Warm both query paths; model load and first-use setup are not query latency.
        dense.retrieve(queries[0].text, 10)
        hybrid.retrieve(queries[0].text, 10)
        scores = {"bm25": [], "dense": [], "hybrid": []}
        latencies = {"dense": [], "hybrid": []}
        overlap = {"both": 0, "bm25_only": 0, "dense_only": 0, "neither": 0,
                   "hybrid_rescues_bm25": 0, "hybrid_rescues_dense": 0,
                   "hybrid_regressions": 0}
        origins = {key: [] for key in ("both", "bm25_only", "dense_only", "neither")}
        per_query = []
        stable_ties = True
        for q in queries:
            started = time.perf_counter()
            dense_hits = dense.retrieve(q.text, 10)
            dense_query_ms = (time.perf_counter() - started) * 1000
            latencies["dense"].append(dense_query_ms)
            started = time.perf_counter()
            hybrid_hits = hybrid.retrieve(q.text, 10)
            hybrid_query_ms = (time.perf_counter() - started) * 1000
            latencies["hybrid"].append(hybrid_query_ms)
            sparse_hits = sparse.retrieve(q.text, 10)
            stable_ties &= ([h.chunk.chunk_id for h in hybrid_hits] ==
                            [h.chunk.chunk_id for h in hybrid.retrieve(q.text, 10)])
            gold = set(q.required_document_versions)
            hits_by_method = {"bm25": sparse_hits, "dense": dense_hits, "hybrid": hybrid_hits}
            success = {}
            for method, hits in hits_by_method.items():
                authorized = [h for h in hits if world.allowed(
                    q.principal_id, h.chunk.logical_document_id, tick)]
                scores[method].append(current_evidence_recall(authorized, gold))
                success[method] = gold <= {h.chunk.document_id for h in authorized}
            origin = ("both" if success["bm25"] and success["dense"] else
                      "bm25_only" if success["bm25"] else
                      "dense_only" if success["dense"] else "neither")
            overlap[origin] += 1
            origins[origin].append(q.query_id)
            overlap["hybrid_rescues_bm25"] += int(not success["bm25"] and success["hybrid"])
            overlap["hybrid_rescues_dense"] += int(not success["dense"] and success["hybrid"])
            overlap["hybrid_regressions"] += int(
                (success["bm25"] or success["dense"]) and not success["hybrid"])
            for document_id in sorted(gold):
                bm25_rank, _ = _rank(sparse_hits, document_id)
                dense_rank, dense_score = _rank(dense_hits, document_id)
                hybrid_rank, hybrid_score = _rank(hybrid_hits, document_id)
                per_query.append({"query_id": q.query_id, "gold_document_id": document_id,
                                  "principal_id": q.principal_id,
                                  "bm25_rank": bm25_rank, "dense_rank": dense_rank,
                                  "hybrid_rank": hybrid_rank,
                                  "bm25_hit_at_10": int(bm25_rank is not None),
                                  "dense_hit_at_10": int(dense_rank is not None),
                                  "hybrid_hit_at_10": int(hybrid_rank is not None),
                                  "dense_score": dense_score, "hybrid_score": hybrid_score,
                                  "dense_query_ms": round(dense_query_ms, 3),
                                  "hybrid_query_ms": round(hybrid_query_ms, 3)})
    return {"model_name": model_name, "model_revision": model_revision,
            "model_path": model_path,
            "device": "cpu", **package_versions,
            "model_load_ms": round(model_load_ms, 3),
            "document_encoding_ms": round(encoding_ms, 3),
            "logical_documents": len({c.logical_document_id for c in chunks}),
            "indexed_chunks": len(chunks), "answerable_queries": len(queries),
            "bm25": {"cer_at_10": sum(scores["bm25"]) / len(queries)},
            "dense": {"cer_at_10": sum(scores["dense"]) / len(queries),
                      "median_query_ms": round(statistics.median(latencies["dense"]), 3),
                      "p95_query_ms": round(_p95(latencies["dense"]), 3)},
            "hybrid": {"cer_at_10": sum(scores["hybrid"]) / len(queries),
                       "median_query_ms": round(statistics.median(latencies["hybrid"]), 3),
                       "p95_query_ms": round(_p95(latencies["hybrid"]), 3),
                       "stable_repeated_ranking": bool(stable_ties),
                       "gold_origin_counts": {k: overlap[k] for k in origins},
                       "representative_query_ids": {k: v[:3] for k, v in origins.items()}},
            "overlap": overlap, "per_query": per_query}


def _write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_artifacts(first, second, output_root):
    """Write independent dense/hybrid artifacts and an all-query comparison."""
    output_root = Path(output_root)
    first_rows, second_rows = first["per_query"], second["per_query"]
    fields = ("query_id", "gold_document_id", "bm25_rank", "dense_rank", "hybrid_rank")
    rankings_identical = ([tuple(row[key] for key in fields) for row in first_rows] ==
                          [tuple(row[key] for key in fields) for row in second_rows])
    cer_identical = all(first[method]["cer_at_10"] == second[method]["cer_at_10"]
                        for method in ("bm25", "dense", "hybrid"))
    score_differences = [abs(a[key] - b[key]) for a, b in zip(first_rows, second_rows)
                         for key in ("dense_score", "hybrid_score")
                         if a[key] is not None and b[key] is not None]
    determinism = {"rankings_identical": rankings_identical,
                   "cer_identical": cer_identical,
                   "stable_hybrid_ties": (first["hybrid"]["stable_repeated_ranking"] and
                                          second["hybrid"]["stable_repeated_ranking"]),
                   "max_gold_score_abs_difference": max(score_differences, default=0.0)}
    for method in ("dense", "hybrid"):
        destination = output_root / f"{method}-smoke"
        destination.mkdir(parents=True, exist_ok=True)
        summary = {"model_name": first["model_name"],
                   "model_revision": first["model_revision"],
                   "model_path": first["model_path"],
                   "device": first["device"], "python": first["python"],
                   "sentence_transformers": first["sentence_transformers"],
                   "torch": first["torch"], "indexed_chunks": first["indexed_chunks"],
                   "logical_documents": first["logical_documents"],
                   "answerable_queries": first["answerable_queries"],
                   "model_load_ms": first["model_load_ms"],
                   "document_encoding_ms": first["document_encoding_ms"],
                   "second_model_load_ms": second["model_load_ms"],
                   "second_document_encoding_ms": second["document_encoding_ms"],
                   "first_run": first[method], "second_run": second[method],
                   "determinism": determinism}
        if method == "hybrid":
            summary["overlap"] = first["overlap"]
        (destination / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
        _write_csv(destination / "per_query.csv", first_rows, PER_QUERY_FIELDS)
        _write_csv(destination / "per_query_run2.csv", second_rows, PER_QUERY_FIELDS)
    _write_csv(output_root / "retrieval-smoke-comparison.csv", first_rows, COMPARISON_FIELDS)
    return determinism


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output-root", default="results")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text())
    try:
        first = run_smoke(config, model_revision=args.model_revision,
                          model_path=args.model_path)
        second = run_smoke(config, model_revision=args.model_revision,
                           model_path=args.model_path)
    except RuntimeError as exc:
        parser.exit(2, f"{exc}\n")
    determinism = write_artifacts(first, second, args.output_root)
    print(json.dumps({"dense": first["dense"], "hybrid": first["hybrid"],
                      "overlap": first["overlap"], "determinism": determinism},
                     indent=2, sort_keys=True))
    return 0 if determinism["rankings_identical"] and determinism["cer_identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
