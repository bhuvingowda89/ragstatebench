import json
from dataclasses import replace

import pytest

from ragstate.campaign.runner import evaluate_scenario, run_campaign, stable_hash, validate_clean
from ragstate.campaign.scenarios import generate_scenarios
from ragstate.campaign.validation import validate_controls
from ragstate.retrieval.bm25 import BM25Retriever
from ragstate.retrieval.dense import DenseRetriever
from ragstate.retrieval.hybrid import HybridRetriever
from ragstate.retrieval.metrics import state_consistent_retrieval_success, unauthorized_context_rate
from ragstate.retrieval.models import Chunk, RetrievalHit
from ragstate.transitions.engine import TransitionEngine
from ragstate.transitions.events import Transition
from ragstate.world.generator import generate_world
from ragstate.world.queries import generate_queries


class MockEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts, batch_size):
        self.calls.append(tuple(texts))
        return [[float("alpha" in x), float("beta" in x)] for x in texts]


def chunk(cid, text):
    return Chunk(cid, cid.split(":")[0], cid.split(":")[0], 1, "tenant_a", 0, text)


def test_dense_batched_cache_and_tie_breaking(tmp_path):
    chunks = [chunk("z:c0", "alpha"), chunk("a:c0", "alpha"),
              chunk("b:c0", "beta")]
    encoder = MockEncoder()
    retriever = DenseRetriever(chunks, "mock", model_version="1", encoder=encoder,
                               cache_dir=tmp_path, batch_size=2)
    assert encoder.calls[:2] == [("alpha", "alpha"), ("beta",)]
    assert [h.chunk.chunk_id for h in retriever.retrieve("alpha", 2)] == ["a:c0", "z:c0"]
    second = MockEncoder()
    DenseRetriever(chunks, "mock", model_version="1", encoder=second,
                   cache_dir=tmp_path, batch_size=2)
    assert second.calls == []
    DenseRetriever(chunks, "mock", model_version="2", encoder=second,
                   cache_dir=tmp_path, batch_size=2)
    assert second.calls


def test_hybrid_rrf_provenance_and_weights():
    chunks = [chunk("a:c0", "alpha"), chunk("b:c0", "beta")]
    sparse = BM25Retriever(chunks)
    dense = DenseRetriever(chunks, "mock", encoder=MockEncoder())
    fusion = HybridRetriever(sparse, dense, rrf_k=0, sparse_weight=2, dense_weight=1)
    hits = fusion.retrieve("alpha", 2)
    assert hits[0].chunk.chunk_id == "a:c0"
    assert hits[0].components == (("bm25", 1, hits[0].components[0][2]),
                                   ("dense", 1, 1.0))
    assert fusion.retrieve("alpha", 0) == []
    with pytest.raises(ValueError):
        HybridRetriever(sparse, dense, sparse_weight=0, dense_weight=0)


def test_large_world_and_queries_use_independent_seeds():
    world = generate_world(7, 170, tenants=3, principals_per_tenant=70,
                           versioned_fraction=0.3)
    assert len(world.principals) >= 210
    assert len(world.documents) > 1000
    queries = generate_queries(world, query_seed=42, count=25)
    assert {q.family for q in queries} == {"snapshot", "authorization",
        "multi_document", "transition_boundary", "negative"}
    assert queries == generate_queries(world, query_seed=42, count=25)
    assert queries != generate_queries(world, query_seed=43, count=25)
    assert all(q.required_atomic_claims for q in queries if q.answerable)
    assert all(not q.answerable for q in queries if q.family == "negative")


def test_group_transition_obeys_logical_time():
    world = generate_world(1, 3)
    logical = "HR-0001"
    principal = "carol"
    TransitionEngine(world).apply(Transition("g", 10, "GROUP_REMOVE", logical,
                                            {"principal_id": principal,
                                             "group_id": "employee"}))
    assert world.allowed(principal, logical, 9)
    assert not world.allowed(principal, logical, 10)


def test_scenarios_reproduce_and_cover_controls():
    world = generate_world(7, 20)
    one = generate_scenarios(world, transition_seed=8, fault_seed=9)
    two = generate_scenarios(world, transition_seed=8, fault_seed=9)
    assert one == two
    assert len(one) == 20
    varied = generate_scenarios(world, transition_seed=8, fault_seed=9, query_seed=11)
    assert [s.events for s in one] == [s.events for s in varied]
    assert [s.query for s in one] != [s.query for s in varied]
    assert {s.fault_type for s in one} >= {"delay", "partial", "out_of_order",
                                               "stale_cache_replay", "dropped_event"}
    assert stable_hash([s.to_dict() for s in one]) == stable_hash([s.to_dict() for s in two])


def config():
    return {"campaign_id": "pilot", "campaign_version": "pilot-test",
            "world_seed": 101, "query_seed": 202, "transition_seed": 303,
            "fault_seed": 404, "docs_per_domain": 20, "retriever": "bm25",
            "architecture": "A0", "top_k": 10,
            "observation_ticks": [9, 10, 11, 12, 13, 14, 15]}


def test_campaign_outputs_controls_and_resume(tmp_path):
    cfg = config()
    rows = run_campaign(cfg, tmp_path)
    assert len(rows) == 140
    assert validate_controls(rows)["pass"]
    assert (tmp_path / "summary.csv").read_text().count("\n") == 141
    assert len((tmp_path / "results.jsonl").read_text().splitlines()) == 140
    assert len(json.loads((tmp_path / "manifest.json").read_text())["scenarios"]) == 20
    before = (tmp_path / "results.jsonl").read_bytes()
    assert len(run_campaign(cfg, tmp_path)) == len(rows)
    assert (tmp_path / "results.jsonl").read_bytes() == before
    with pytest.raises(ValueError, match="different campaign"):
        run_campaign({**cfg, "fault_seed": 405}, tmp_path)


def test_clean_gate_and_forced_forbidden_evidence():
    cfg = config()
    assert validate_clean(cfg)["pass"]
    world = generate_world(7, 3)
    forbidden = world.documents["FIN-SENSITIVE-v1"]
    hit = RetrievalHit(Chunk("FIN-SENSITIVE-v1:c0", "FIN-SENSITIVE-v1",
                             "FIN-SENSITIVE", 1, "tenant_a", 0, forbidden.content), 999.0)
    assert not state_consistent_retrieval_success([hit], set(), "carol", world, 0)
    assert unauthorized_context_rate([hit], "carol", world, 0) == 1.0


def test_architecture_mitigations_and_only_stale_version():
    cfg = config()
    world = generate_world(cfg["world_seed"], cfg["docs_per_domain"])
    scenarios = {s.scenario_id: s for s in generate_scenarios(
        world, transition_seed=cfg["transition_seed"], fault_seed=cfg["fault_seed"])}
    def evaluate(name, architecture, **extras):
        return evaluate_scenario(world, scenarios[name],
                                 {**cfg, "architecture": architecture, **extras}, {})
    stale = evaluate("update_delay", "A0")
    assert stale[1]["srr"] > 0 and stale[1]["ardr"] == 1
    assert evaluate("update_delay", "A1", sync_interval=1)[1]["scrs"] == 1
    assert evaluate("revoke_delay", "A4")[1]["ucer"] == 0
    assert evaluate("revoke_delay", "A5")[1]["ucer"] == 0
    assert evaluate("supersede_cache", "A0")[-1]["scrs"] == 0
    assert evaluate("supersede_cache", "A3")[-1]["scrs"] == 1
    assert evaluate("mixed_versions", "A0")[1]["mvcr"] == 1.0


def test_campaign_rejects_invalid_configuration(tmp_path):
    with pytest.raises(ValueError, match="architecture"):
        run_campaign({**config(), "architecture": "A6"}, tmp_path)
    with pytest.raises(ValueError, match="observation_ticks"):
        run_campaign({**config(), "observation_ticks": [10, 9]}, tmp_path)
