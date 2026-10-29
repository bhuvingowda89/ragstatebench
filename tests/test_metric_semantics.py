import csv
import json
from dataclasses import replace

from ragstate.campaign.runner import aggregate, evaluate_scenario, run_campaign
from ragstate.campaign.scenarios import generate_scenarios
from ragstate.retrieval.chunking import chunk_document
from ragstate.retrieval.metrics import (
    authorized_retrieval_denial, cross_tenant_exposure, current_evidence_recall,
    deleted_evidence_persistence, invalid_evidence_absence_success,
    mixed_version_context_rate, state_consistent_retrieval_success,
    superseded_retrieval_rate, unauthorized_context_rate,
)
from ragstate.retrieval.models import RetrievalHit
from ragstate.world.generator import generate_world
from ragstate.world.ledger import WorldLedger
from ragstate.world.models import DocumentVersion, PermissionRule, Principal
from ragstate.campaign.smoke import run_smoke, write_artifacts


def _fixture(deleted=False, revoked=False, tenant="t1"):
    world = WorldLedger()
    world.add_principal(Principal("p", "t1", frozenset({"employee"})))
    doc = DocumentVersion("d-v1", "d", tenant, 1, 0, deleted_at=10 if deleted else None,
                          content="Current limit is $85.",
                          required_groups=frozenset({"employee"}))
    world.add_document(doc)
    if revoked:
        world.add_permission(PermissionRule("p", "d", False, 10))
    return world, RetrievalHit(chunk_document(doc)[0], 1.0)


def test_deleted_previously_authorized_is_dep_without_ucer():
    world, hit = _fixture(deleted=True)
    assert deleted_evidence_persistence([hit], world, 10)
    assert unauthorized_context_rate([hit], "p", world, 10) == 0.0
    assert current_evidence_recall([hit], set()) is None
    assert invalid_evidence_absence_success([hit], set(), {"d"}) == 0
    assert superseded_retrieval_rate([hit], set(), world, 10) == 0.0


def test_current_revoked_is_ucer_without_dep():
    world, hit = _fixture(revoked=True)
    assert not deleted_evidence_persistence([hit], world, 10)
    assert unauthorized_context_rate([hit], "p", world, 10) == 1.0
    assert invalid_evidence_absence_success([hit], set(), {"d"}) == 0


def test_deleted_and_independently_revoked_can_raise_both():
    world, hit = _fixture(deleted=True, revoked=True)
    assert deleted_evidence_persistence([hit], world, 10)
    assert unauthorized_context_rate([hit], "p", world, 10) == 1.0


def test_absence_and_ardr_have_distinct_domains():
    world, hit = _fixture(deleted=True)
    assert invalid_evidence_absence_success([], set(), {"d"}) == 1
    assert invalid_evidence_absence_success([hit], set(), {"unrelated"}) == 1
    assert invalid_evidence_absence_success([], {"d-v1"}, {"d"}) is None
    assert not authorized_retrieval_denial(set(), [])
    assert authorized_retrieval_denial({"d-v1"}, [])


def test_metric_dimensions_are_separable():
    world, current = _fixture()
    old = replace(current.chunk, chunk_id="d-v0:c0", document_id="d-v0", version=0)
    old_hit = RetrievalHit(old, 1.0)
    assert superseded_retrieval_rate([old_hit], {"d-v1"}, world, 10) == 1.0
    assert not deleted_evidence_persistence([old_hit], world, 10)
    assert unauthorized_context_rate([old_hit], "p", world, 10) == 0.0
    assert mixed_version_context_rate([current, old_hit]) == 1.0
    assert mixed_version_context_rate([current]) == 0.0
    assert not cross_tenant_exposure([current], "p", world)
    other_tenant = RetrievalHit(replace(current.chunk, tenant_id="t2"), 1.0)
    assert cross_tenant_exposure([other_tenant], "p", world)
    assert unauthorized_context_rate([other_tenant], "p", world, 10) == 0.0
    corrupted = RetrievalHit(replace(current.chunk, text="Current limit is $999."), 1.0)
    assert current_evidence_recall([corrupted], {"d-v1"}) == 1.0
    assert superseded_retrieval_rate([corrupted], {"d-v1"}, world, 10) == 0.0
    assert not state_consistent_retrieval_success([corrupted], {"d-v1"}, "p", world, 10)


def _scenario_rows(name, architecture="A0"):
    world = generate_world(101, 20)
    scenarios = {s.scenario_id: s for s in generate_scenarios(
        world, transition_seed=303, fault_seed=404, query_seed=202)}
    config = {"retriever": "bm25", "architecture": architecture, "top_k": 10,
              "observation_ticks": [9, 10, 11, 12, 13, 14, 15]}
    return evaluate_scenario(world, scenarios[name], config, {})


def test_scenario_regressions_and_boundary_localization():
    deleted = _scenario_rows("delete_drop")[1]
    assert deleted["deleted_evidence_present"] == 1
    assert deleted["ucer"] == deleted["context_ucer"] == 0
    assert deleted["cer_at_k"] is None
    assert deleted["invalid_evidence_absence_success"] == 0
    revoked = _scenario_rows("revoke_delay")[1]
    assert revoked["ucer"] > 0 and revoked["deleted_evidence_present"] == 0
    assert revoked["candidate_ucer"] > 0
    assert revoked["post_filter_ucer"] > 0
    assert _scenario_rows("revoke_delay", "A4")[1]["post_filter_ucer"] == 0
    replayed = _scenario_rows("revoke_cache")[-1]
    assert replayed["candidate_ucer"] > 0
    assert replayed["post_filter_ucer"] == 0
    assert replayed["context_ucer"] > 0
    partial = _scenario_rows("supersede_partial")[-1]
    assert partial["cer_at_k"] == 1.0 and partial["srr"] == 0.0
    assert partial["scrs"] == 0
    assert _scenario_rows("mixed_versions")[1]["mvcr"] == 1.0
    assert _scenario_rows("cross_tenant")[1]["cross_tenant_exposure"] == 1
    assert _scenario_rows("grant_delay")[1]["ardr"] == 1


def test_null_cer_serializes_and_aggregate_excludes_it(tmp_path):
    config = {"campaign_id": "pilot", "campaign_version": "pilot-test",
              "world_seed": 101, "query_seed": 202, "transition_seed": 303,
              "fault_seed": 404, "docs_per_domain": 20, "retriever": "bm25",
              "architecture": "A0", "top_k": 10,
              "observation_ticks": [9, 10, 11, 12, 13, 14, 15]}
    rows = run_campaign(config, tmp_path)
    blank = next(r for r in rows if r["scenario_id"] == "delete_drop" and r["logical_time"] == 10)
    assert blank["cer_at_k"] is None
    with (tmp_path / "summary.csv").open(newline="") as f:
        csv_rows = list(csv.DictReader(f))
    csv_blank = next(r for r in csv_rows if r["scenario_id"] == "delete_drop" and r["logical_time"] == "10")
    assert csv_blank["cer_at_k"] == ""
    json_blank = next(json.loads(line) for line in (tmp_path / "results.jsonl").read_text().splitlines()
                      if json.loads(line)["scenario_id"] == "delete_drop"
                      and json.loads(line)["logical_time"] == 10)
    assert json_blank["cer_at_k"] is None
    group = [dict(blank, fault_type="synthetic", cer_at_k=None),
             dict(blank, fault_type="synthetic", cer_at_k=0.5)]
    assert aggregate(group)[0]["mean_cer_at_k"] == 0.5
    assert aggregate([group[0]])[0]["mean_cer_at_k"] is None


def test_smoke_harness_reports_fusion_and_latency_with_injected_encoder(tmp_path):
    class Encoder:
        def encode(self, texts, batch_size):
            return [[float("policy" in text.lower()), float("finance" in text.lower())]
                    for text in texts]
    report = run_smoke({"world_seed": 101, "query_seed": 202,
                        "docs_per_domain": 20, "clean_query_count": 10},
                       model_revision="mock-revision", encoder=Encoder())
    assert report["indexed_chunks"] >= 120
    assert report["answerable_queries"] > 0
    assert report["document_encoding_ms"] >= 0
    assert report["hybrid"]["stable_repeated_ranking"]
    assert sum(report["hybrid"]["gold_origin_counts"].values()) == report["answerable_queries"]
    repeated = run_smoke({"world_seed": 101, "query_seed": 202,
                          "docs_per_domain": 20, "clean_query_count": 10},
                         model_revision="mock-revision", encoder=Encoder())
    deterministic = write_artifacts(report, repeated, tmp_path)
    assert deterministic["rankings_identical"] and deterministic["cer_identical"]
    assert (tmp_path / "dense-smoke" / "summary.json").exists()
    assert (tmp_path / "hybrid-smoke" / "per_query.csv").exists()
    with (tmp_path / "retrieval-smoke-comparison.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows and {"query_id", "gold_document_id", "bm25_rank", "dense_rank",
                     "hybrid_rank"} <= rows[0].keys()
