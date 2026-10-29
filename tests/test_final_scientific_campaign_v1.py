"""Final campaign provenance, nonduplication, and scientific hard gates."""
import csv
import hashlib
import json
from pathlib import Path

from ragstate.campaign.final_scientific_campaign_v1 import _completed, verify_freeze


ROOT=Path('results/final-scientific-campaign-v1')


def test_final_policy_and_manifest_frozen_before_observations():
    config,manifest,provenance=verify_freeze()
    assert hashlib.sha256((ROOT/'analysis_policy.json').read_bytes()).hexdigest()==\
        provenance['policy_sha256']
    assert manifest['scenario_count']==6678
    assert sum(manifest['block_counts'].values())==6678
    assert config['campaign_version']=='final-scientific-campaign-v1'
    assert not any(s['final_block']=='FA2' for s in manifest['scenarios'])


def test_new_expansion_has_no_semantically_identical_deterministic_scenarios():
    _,manifest,_=verify_freeze()
    def key(s):
        return (s['query_id'],s['transition'],s['severity'],s['architecture'],
                s['fault'],s.get('compound'),s.get('component'),
                tuple(s.get('component_tuple') or ()),s.get('partial_topology'))
    inherited={key(s) for s in manifest['scenarios']
               if s.get('source_applicability_scenario_sha256')}
    added=[s for s in manifest['scenarios']
           if not s.get('source_applicability_scenario_sha256')]
    added_keys=[key(s) for s in added]
    assert len(added)==2436
    assert len(added_keys)==len(set(added_keys))
    assert not inherited.intersection(added_keys)


def test_final_checkpoints_and_counts_reconcile():
    _,manifest,_=verify_freeze()
    execution=json.loads((ROOT/'execution_summary.json').read_text())
    status=list(csv.DictReader((ROOT/'scenario_status.csv').open()))
    assert execution['scenario_count']==len(status)==6678
    assert execution['observation_count']==15*6678
    assert execution['converged']+execution['censored']==6678
    assert _completed(manifest['scenarios'][0])['observation_count']==15
    assert _completed(manifest['scenarios'][-1])['observation_count']==15


def test_final_gates_preserve_h4_t5_and_separate_tracks():
    gate=json.loads((ROOT/'gate_summary.json').read_text())
    assert all(gate['gates'][f'FC{i}']=='PASS' for i in range(1,11))
    effects=list(csv.DictReader((ROOT/'trackB_transfer_effects.csv').open()))
    assert all(r['classification']=='not estimable' for r in effects
               if r['effect_class']=='T5')
    assert gate['D11']['agreement_numerator']==gate['D11']['agreement_denominator']==16
    reproduction=json.loads((ROOT/'reproducibility.json').read_text())
    assert reproduction['identical'] and reproduction['track_b_full_repeat_identical']


def test_final_figures_and_tables_trace_to_analysis_artifacts():
    trace=json.loads((ROOT/'traceability.json').read_text())
    assert len(trace['figures'])>=13 and len(trace['tables'])==12
    assert all(item['match'] for item in trace['h1_raw_recomputation'])
    assert all(item['source_sha256']==item['copy_sha256'] for item in trace['tables'])
    assert all(item['svg_sha256'] and item['png_sha256'] for item in trace['figures'])
