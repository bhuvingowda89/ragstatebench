"""Topology-aware PARTIAL matrix compilation before scientific execution.

The v1 compiler and invalid campaigns remain immutable. This module replaces
only TA2's universal percentage PARTIAL candidates; every other frozen
candidate is carried forward and recompiled with v1's correctness rules.
No scientific observation rows or metrics are loaded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from ragstate.replicas.scientific import ReplicaAuthorizationPlane, ReplicaContentPlane
from .scenario_applicability import (
    AUTH, CONTENT, CONFIG, SOURCE_MANIFEST, TICK, ApplicabilityResult,
    _candidate_row, _csv, _matrix, _operator_names, _transition_check,
    _transition_facts, compile_candidate, dry_run,
)
from .state_pilot_manifest import _load_frozen, digest


POLICY_VERSION = 'scenario-applicability-v2'
OUTPUT = Path('results/scenario-applicability-v2')
PARTIAL_TRANSITIONS = CONTENT + ('REVOKE',)
PARTIAL_PLAN_VERSION = 'topology_units_seeded_v1'
PRIMARY_CONTENT_TRANSITIONS = CONTENT


def midpoint_updated(n: int) -> int:
    """Round N/2 half up for positive integral N."""
    if n < 1:
        raise ValueError('N must be positive')
    return (n + 1) // 2


def topology_states(n: int) -> tuple[tuple[str, int], ...]:
    """Return distinct mixed states; midpoint is retained only if distinct."""
    if n < 2:
        return ()
    low, high, mid = 1, n - 1, midpoint_updated(n)
    states = [('P_LOW', low)]
    if mid not in (low, high):
        states.append(('P_MID', mid))
    if high != low:
        states.append(('P_HIGH', high))
    return tuple(states)


def selected_unit_ids(unit_ids, updated_count: int, *, seed: int,
                      query_id: str, transition: str, severity: str,
                      target_plane: str) -> tuple[str, ...]:
    """Stable relevance-blind subset, nested across topology states."""
    ids = tuple(unit_ids)
    if not 1 <= updated_count < len(ids) or len(set(ids)) != len(ids):
        raise ValueError('Selected-unit request is not a strict unique subset')
    base = f'{seed}|{query_id}|{transition}|{severity}|{target_plane}'
    rank = sorted(ids, key=lambda identity: (
        hashlib.sha256(f'{base}|{identity}'.encode()).hexdigest(), identity))
    chosen = set(rank[:updated_count])
    return tuple(identity for identity in ids if identity in chosen)


def _partial_units(facts, transition):
    lid = facts['lid']
    old = tuple(facts['chunk'].chunks.get(lid, ()))
    if transition in ('UPDATE', 'SUPERSEDE'):
        fresh = ReplicaContentPlane(facts['post'], TICK, chunked=True)
        new = tuple(fresh.chunks.get(lid, ()))
        return 'content_chunk', old, new, tuple(chunk.chunk_id for chunk in new)
    if transition == 'DELETE':
        return 'content_chunk', old, (), tuple(chunk.chunk_id for chunk in old)
    return 'acl_entry', old, (), tuple(chunk.chunk_id for chunk in old)


def _uniform_control_id(scenario):
    return '__'.join(('TA1', scenario['query_id'], scenario['transition'],
                     scenario['severity'], scenario['architecture'], 'DELAY'))


def _make_partial_candidates(source, world, metadata, query_by_id, config):
    controls = {s['scenario_id']: s for s in source if s['block'] == 'TA1'}
    base = sorted((s for s in source if s['block'] == 'TA1'
                   and s['severity'] == 'S2' and s['architecture'] == 'A0'
                   and s['transition'] in PARTIAL_TRANSITIONS),
                  key=lambda s: s['scenario_id'])
    candidates = []
    facts_cache = {}
    for control in base:
        query = query_by_id[control['query_id']]
        key = (query.query_id, control['transition'])
        if key not in facts_cache:
            facts_cache[key] = _transition_facts(world, metadata, query,
                                                  control['transition'])
        facts = facts_cache[key]
        plane, old, new, identities = _partial_units(facts, control['transition'])
        n = len(identities)
        states = topology_states(n) or (('P_LOW', 1),)  # retain unsplittable candidates as exclusions
        for label, updated in states:
            scenario = dict(control)
            scenario.update(block='TA2', fault='PARTIAL',
                            partial_topology=label, partial_plan_version=PARTIAL_PLAN_VERSION,
                            total_unit_count=n, updated_unit_count=updated,
                            realized_fraction=updated / n if n else None,
                            partial_fraction=updated / n if n else None,
                            replica_unit_type=plane,
                            selected_unit_ids=(selected_unit_ids(identities, updated,
                                seed=config['selection_seed'], query_id=query.query_id,
                                transition=control['transition'], severity='S2',
                                target_plane=plane) if 1 <= updated < n else ()),
                            matched_uniform_scenario_id=control['scenario_id'],
                            partial_selection_seed=config['selection_seed'])
            scenario['scenario_id'] = '__'.join(('TA2', query.query_id,
                control['transition'], 'S2', 'A0', 'PARTIAL', label,
                f'{updated}of{n}'))
            if _uniform_control_id(scenario) not in controls:
                raise ValueError('Partial candidate lacks matched uniform control')
            candidates.append(scenario)
    return candidates


def compile_topology_partial(scenario, query, facts, config):
    transition = _transition_check(scenario, query, facts)
    plane, old, new, unit_ids = _partial_units(facts, scenario['transition'])
    n, updated = scenario['total_unit_count'], scenario['updated_unit_count']
    reasons = []
    if not transition['effective']:
        reasons.append('transition_precondition_failed')
    if n != len(unit_ids):
        reasons.append('replica_unit_count_mismatch')
    if n < 2 or not 1 <= updated < n:
        reasons.append('partial_split_not_representable')
    if scenario['transition'] in ('UPDATE', 'SUPERSEDE') and len(old) != len(new):
        reasons.append('old_new_chunk_alignment_not_representable')
    if scenario['transition'] not in PARTIAL_TRANSITIONS:
        reasons.append('partial_transition_not_supported')
    if scenario['matched_uniform_scenario_id'] != _uniform_control_id(scenario):
        reasons.append('matched_uniform_control_mismatch')
    if scenario['realized_fraction'] != (updated / n if n else None):
        reasons.append('realized_fraction_mismatch')
    if not config['architecture_components'].get(scenario['architecture']):
        reasons.append('unknown_architecture')
    if 1 <= updated < n:
        expected_selection = selected_unit_ids(unit_ids, updated,
            seed=config['selection_seed'], query_id=query.query_id,
            transition=scenario['transition'], severity=scenario['severity'],
            target_plane=plane)
        if tuple(scenario['selected_unit_ids']) != expected_selection:
            reasons.append('deterministic_unit_selection_mismatch')
    component = {'applicable': not reasons, 'reasons': reasons,
                 'unit_count': n, 'updated_unit_count': updated,
                 'stale_unit_count': n - updated,
                 'target_plane': 'replica_chunk_units' if plane == 'content_chunk'
                                 else 'replica_acl_units',
                 'unit_type': plane,
                 'effectiveness_predicate': 'strict_updated_stale_split'}
    return ApplicabilityResult(scenario['scenario_id'], not reasons, tuple(reasons),
        {'old_chunk_units': len(old), 'new_chunk_units': len(new),
         'selected_unit_ids': list(scenario['selected_unit_ids']),
         'matched_uniform_scenario_id': scenario['matched_uniform_scenario_id']},
        {'PARTIAL': n}, transition, {'PARTIAL': component},
        {'runnable': scenario['architecture'] in config['architecture_components'],
         'informative': {'partial': True}}, (component['target_plane'],),
        ('strict_updated_stale_split',), {'PARTIAL': component}, updated)


def apply_topology_partial(scenario, query, facts):
    """State-only operator shared by v2 compiler and future runtime adapter.

    It updates the declared subset of real replica units and preserves the
    complementary stale subset. The canonical ledger and evaluator are absent.
    """
    transition, event, lid = scenario['transition'], facts['event'], facts['lid']
    chunk = ReplicaContentPlane(facts['pre'], TICK - 1, chunked=True)
    parent = ReplicaContentPlane(facts['pre'], TICK - 1, chunked=False)
    acl = ReplicaAuthorizationPlane(facts['pre'])
    plane, old, new, identities = _partial_units(facts, transition)
    chosen = set(scenario['selected_unit_ids'])
    if len(chosen) != scenario['updated_unit_count'] or not chosen <= set(identities):
        raise ValueError('Partial plan selected units do not match replica topology')
    if plane == 'content_chunk':
        if transition == 'DELETE':
            chunk.chunks[lid] = [part for part in old if part.chunk_id not in chosen]
        else:
            chunk.chunks[lid] = [new[index] if new[index].chunk_id in chosen else old[index]
                                 for index in range(len(old))]
            parent.apply(event)
        chunk.epoch += 1
        chunk._state_hash = None
        stale = sum(part.document_id == old[index].document_id
                    for index, part in enumerate(chunk.chunks[lid]))
        actual_updated = scenario['updated_unit_count'] if transition == 'DELETE' else \
            len(chunk.chunks[lid]) - stale
    else:
        units = sorted(chunk.chunks[lid], key=lambda part: part.chunk_id)
        old_values = {(query.principal_id, part.chunk_id):
                      acl.allowed(query.principal_id, part, TICK - 1) for part in units}
        acl.policy.apply_authorization_event(event)
        for part in units:
            key = (query.principal_id, part.chunk_id)
            if part.chunk_id in chosen:
                acl.chunk_overrides.pop(key, None)
            else:
                acl.chunk_overrides[key] = old_values[key]
        acl.epoch += 1
        acl._state_hash = None
        stale = sum((query.principal_id, part.chunk_id) in acl.chunk_overrides
                    for part in units)
        actual_updated = len(units) - stale
    return {'updated_units': actual_updated, 'stale_units': stale,
            'unit_count': len(identities), 'realized_fraction': actual_updated / len(identities),
            'selected_unit_ids': tuple(scenario['selected_unit_ids']),
            'chunk_hash': chunk.state_hash(), 'parent_hash': parent.state_hash(),
            'acl_hash': acl.state_hash(TICK)}


def dry_run_topology(scenario, query, facts, result):
    if not result.applicable:
        return {'scenario_id': scenario['scenario_id'], 'realizable': False,
                'reason': 'excluded_by_applicability', 'dry_run_performed': False,
                'checks': []}
    actual = apply_topology_partial(scenario, query, facts)
    checks = [actual['updated_units'] == scenario['updated_unit_count'],
              actual['stale_units'] == scenario['total_unit_count'] - scenario['updated_unit_count'],
              0 < actual['updated_units'] < actual['unit_count'],
              actual['realized_fraction'] == scenario['realized_fraction']]
    return {'scenario_id': scenario['scenario_id'], 'realizable': all(checks),
            'reason': None if all(checks) else 'topology_state_not_realizable',
            'dry_run_performed': True, 'checks': checks,
            'actual': actual}


def _prepare_candidates(config, world, metadata, query_by_id):
    source = json.loads(SOURCE_MANIFEST.read_text())
    scenarios = source['scenarios']
    if digest(scenarios) != source['sha256']:
        raise ValueError('Frozen source candidate hash mismatch')
    unchanged = [s for s in scenarios if not
                 (s['block'] == 'TA2' and s['fault'] == 'PARTIAL')]
    partial = _make_partial_candidates(scenarios, world, metadata,
                                        query_by_id, config)
    candidates = unchanged + partial
    if len({s['scenario_id'] for s in candidates}) != len(candidates):
        raise ValueError('Duplicate scenario ID in topology-aware matrix')
    return candidates, source['sha256'], len(scenarios) - len(unchanged)


def _v2_row(scenario, query, result):
    row = _candidate_row(scenario, query, result)
    row.update(partial_topology=scenario.get('partial_topology'),
               realized_fraction=scenario.get('realized_fraction'),
               selected_unit_ids=list(scenario.get('selected_unit_ids', ())),
               matched_uniform_scenario_id=scenario.get('matched_uniform_scenario_id'),
               partial_plan_version=scenario.get('partial_plan_version'),
               total_unit_count=scenario.get('total_unit_count'))
    if scenario.get('partial_topology'):
        row['requested_partial_fraction'] = None
    return row


def compile_all(output=OUTPUT, *, write=True, perform_dry_run=True):
    """Recompile all frozen nonpartial candidates plus topology-aware PARTIAL."""
    config = json.loads(CONFIG.read_text())
    world, metadata, queries, _ = _load_frozen(config)
    query_by_id = {query.query_id: query for query in queries}
    scenarios, source_hash, retired_partial_count = _prepare_candidates(
        config, world, metadata, query_by_id)
    facts_cache = {}
    rows, audits, emitted = [], [], []
    for scenario in scenarios:
        query = query_by_id[scenario['query_id']]
        key = (query.query_id, scenario['transition'])
        if key not in facts_cache:
            try:
                facts_cache[key] = _transition_facts(world, metadata, query,
                                                      scenario['transition'])
            except Exception as exc:
                facts_cache[key] = exc
        facts = facts_cache[key]
        if isinstance(facts, Exception):
            result = ApplicabilityResult(scenario['scenario_id'], False,
                ('transition_construction_failed:' + type(facts).__name__,),
                {}, {}, {'effective': False}, {}, {'runnable': False}, (), (), {}, None)
            audit = {'scenario_id': scenario['scenario_id'], 'realizable': False,
                     'reason': 'excluded_by_applicability', 'dry_run_performed': False,
                     'checks': []}
        else:
            if scenario.get('partial_plan_version') == PARTIAL_PLAN_VERSION:
                result = compile_topology_partial(scenario, query, facts, config)
                audit = dry_run_topology(scenario, query, facts, result) if perform_dry_run else {
                    'scenario_id': scenario['scenario_id'], 'realizable': result.applicable,
                    'reason': 'determinism_pass_without_dry_run',
                    'dry_run_performed': False, 'checks': []}
            else:
                result = compile_candidate(scenario, query, facts, config)
                audit = dry_run(scenario, query, facts, result) if perform_dry_run else {
                    'scenario_id': scenario['scenario_id'], 'realizable': result.applicable,
                    'reason': 'determinism_pass_without_dry_run',
                    'dry_run_performed': False, 'checks': []}
        row = _v2_row(scenario, query, result)
        rows.append(row)
        audits.append(audit)
        if result.applicable and audit['realizable']:
            emitted.append(scenario)
    row_by_id = {row['scenario_id']: row for row in rows}
    emitted_ids = {scenario['scenario_id'] for scenario in emitted}
    missing_controls = [scenario['scenario_id'] for scenario in emitted
                        if scenario.get('partial_topology') and
                        scenario['matched_uniform_scenario_id'] not in emitted_ids]
    if missing_controls:
        raise ValueError('Topology-aware PARTIAL lacks executable matched control: '
                         + ','.join(missing_controls[:3]))
    failures = [audit for audit in audits if row_by_id[audit['scenario_id']]['applicable']
                and not audit['realizable']]
    excluded = [row for row in rows if not row['applicable']]
    output = Path(output)
    compiled = {'config': config, 'output': output, 'rows': rows,
                'audits': audits, 'emitted': emitted, 'excluded': excluded,
                'failed_realizability': failures, 'source_hash': source_hash,
                'candidate_hash': digest(rows), 'retired_partial_count': retired_partial_count}
    if write:
        if output.exists() and any(output.iterdir()):
            raise ValueError('Versioned output already exists; refusing to overwrite')
        output.mkdir(parents=True, exist_ok=True)
        fields = list(rows[0])
        executable_rows = [row_by_id[scenario['scenario_id']] for scenario in emitted]
        _csv(output / 'candidate_scenarios.csv', rows, fields)
        _csv(output / 'executable_scenarios.csv', executable_rows, fields)
        _csv(output / 'excluded_scenarios.csv', excluded, fields)
        normalized_audits = [{**audit, 'actual': audit.get('actual')}
                             for audit in audits]
        _csv(output / 'realizability_audit.csv', normalized_audits,
             ['scenario_id', 'realizable', 'reason', 'dry_run_performed', 'checks', 'actual'])
        matrix_fields = ['transition', 'fault', 'severity', 'domain', 'architecture']
        _csv(output / 'transition_fault_applicability.csv',
             _matrix(rows, matrix_fields), matrix_fields + ['considered', 'applicable', 'excluded'])
        compounds = [row for row in rows if row['compound']]
        compound_fields = ['compound', 'component', 'domain', 'architecture']
        _csv(output / 'compound_applicability.csv',
             _matrix(compounds, compound_fields),
             compound_fields + ['considered', 'applicable', 'excluded'])
        partial = [row for row in rows if row['partial_topology']]
        topology_fields = ['transition', 'domain', 'partial_topology', 'total_unit_count']
        _csv(output / 'partial_topology_matrix.csv',
             _matrix(partial, topology_fields),
             topology_fields + ['considered', 'applicable', 'excluded'])
        fraction_fields = ['scenario_id', 'query_id', 'domain', 'transition',
                           'partial_topology', 'replica_unit_type', 'total_unit_count',
                           'updated_unit_count', 'realized_fraction', 'applicable']
        _csv(output / 'partial_realized_fraction.csv',
             [{key: row[key] for key in fraction_fields} for row in partial],
             fraction_fields)
        pairs = [{'partial_scenario_id': scenario['scenario_id'],
                  'uniform_scenario_id': scenario['matched_uniform_scenario_id'],
                  'query_id': scenario['query_id'], 'domain': row_by_id[scenario['scenario_id']]['domain'],
                  'transition': scenario['transition'], 'architecture': scenario['architecture'],
                  'severity': scenario['severity'], 'partial_topology': scenario['partial_topology'],
                  'realized_fraction': scenario['realized_fraction']}
                 for scenario in emitted if scenario.get('partial_topology')]
        _csv(output / 'matched_partial_uniform_pairs.csv', pairs, list(pairs[0]))
    return compiled


def coverage(compiled):
    emitted = compiled['emitted']
    rows = {row['scenario_id']: row for row in compiled['rows']}
    partial = [s for s in emitted if s.get('partial_topology')]
    pairs = [(s['transition'], rows[s['scenario_id']]['domain'], s['query_id']) for s in partial]
    unique_queries = defaultdict(set)
    for transition, domain, qid in pairs:
        unique_queries[transition, domain].add(qid)
    content_cell_counts = [len(unique_queries[transition, domain])
                           for transition in PRIMARY_CONTENT_TRANSITIONS
                           for domain in sorted({row['domain'] for row in rows.values()})]
    by_transition = {transition: len(set(qid for t, _, qid in pairs if t == transition))
                     for transition in PARTIAL_TRANSITIONS}
    by_domain_transition = [{
        'transition': transition, 'domain': domain,
        'matched_distinct_queries': len(unique_queries[transition, domain]),
        'partial_scenarios': sum(t == transition and d == domain for t, d, _ in pairs)}
        for transition in PARTIAL_TRANSITIONS
        for domain in sorted({row['domain'] for row in rows.values()})]
    fractions = sorted(s['realized_fraction'] for s in partial)
    unit_counts = Counter(s['total_unit_count'] for s in partial)
    compound = [s for s in emitted if s.get('compound')]
    return {'considered': len(compiled['rows']), 'applicable': len(emitted),
            'excluded': len(compiled['excluded']),
            'retired_universal_percentage_candidates': compiled['retired_partial_count'],
            'realizability_failures': len(compiled['failed_realizability']),
            'by_block': dict(Counter(s['block'] for s in emitted)),
            'by_transition': dict(Counter(s['transition'] for s in emitted)),
            'by_domain': dict(Counter(rows[s['scenario_id']]['domain'] for s in emitted)),
            'partial': {'scenario_count': len(partial),
                        'matched_pair_count': len(partial),
                        'unique_query_counts_by_transition': by_transition,
                        'transition_domain': by_domain_transition,
                        'topology_counts': dict(Counter(s['partial_topology'] for s in partial)),
                        'realized_fraction': {'min': fractions[0] if fractions else None,
                                              'median': fractions[len(fractions)//2] if fractions else None,
                                              'max': fractions[-1] if fractions else None,
                                              'distribution': dict(Counter(str(v) for v in fractions))},
                        'replica_unit_counts': dict(sorted(unit_counts.items())),
                        'content_cell_min': min(content_cell_counts),
                        'content_cell_median': sorted(content_cell_counts)[len(content_cell_counts)//2],
                        'content_cell_max': max(content_cell_counts),
                        'content_domain_coverage': {transition: sum(
                            len(unique_queries[transition, domain]) > 0
                            for domain in sorted({row['domain'] for row in rows.values()}))
                            for transition in PRIMARY_CONTENT_TRANSITIONS}},
            'compounds': {'count': len(compound),
                          'by_compound': dict(Counter(s['compound'] for s in compound)),
                          'cf3_component_counts': dict(Counter(s['component'] for s in compound
                                                               if s['compound'] == 'CF3'))},
            'exclusion_reasons': dict(Counter(reason for row in compiled['excluded']
                for reason in row['exclusion_reason'].split(';') if reason))}


def topology_smoke(compiled):
    """Exercise actual replica planes at N=2, N=3, and N>=4 in >=2 domains."""
    candidates = [scenario for scenario in compiled['emitted']
                  if scenario.get('partial_topology')]
    rows = {row['scenario_id']: row for row in compiled['rows']}
    strata = ((2, lambda n: n == 2), (3, lambda n: n == 3),
              (4, lambda n: n >= 4))
    chosen = []
    domains = set()
    for label, predicate in strata:
        matches = sorted((scenario for scenario in candidates
                          if predicate(scenario['total_unit_count'])),
                         key=lambda scenario: (rows[scenario['scenario_id']]['domain'] in domains,
                                               scenario['scenario_id']))
        if matches:
            chosen.append((label, matches[0]))
            domains.add(rows[matches[0]['scenario_id']]['domain'])
    config = compiled['config']
    world, metadata, queries, _ = _load_frozen(config)
    by_id = {query.query_id: query for query in queries}
    results = []
    for label, scenario in chosen:
        query = by_id[scenario['query_id']]
        facts = _transition_facts(world, metadata, query, scenario['transition'])
        plan = compile_topology_partial(scenario, query, facts, config)
        actual = dry_run_topology(scenario, query, facts, plan)
        results.append({'unit_stratum': label, 'domain': rows[scenario['scenario_id']]['domain'],
                        'scenario_id': scenario['scenario_id'],
                        'predicted_updated': scenario['updated_unit_count'],
                        'actual_updated': actual.get('actual', {}).get('updated_units'),
                        'predicted_stale': scenario['total_unit_count'] - scenario['updated_unit_count'],
                        'actual_stale': actual.get('actual', {}).get('stale_units'),
                        'plan_applicable': plan.applicable,
                        'replica_transform_realizable': actual['realizable']})
    return {'agreement': len(results) == 3 and len({r['domain'] for r in results}) >= 2
            and all(r['plan_applicable'] and r['replica_transform_realizable']
                    and r['predicted_updated'] == r['actual_updated']
                    and r['predicted_stale'] == r['actual_stale'] for r in results),
            'records': results,
            'scope': 'state-only replica operator; no oracle metric or campaign trajectory'}


def fingerprint(compiled):
    return {'source_hash': compiled['source_hash'],
            'candidate_table_sha256': compiled['candidate_hash'],
            'exclusions_sha256': digest(compiled['excluded']),
            'executable_manifest_sha256': digest(compiled['emitted'])}


def h4_policy():
    return {
        'version': 'H4-topology-aware-partial-v1',
        'frozen_before_transition_outcomes': True,
        'primary_population': 'matched applicable Track-A partial and uniform-delay scenarios',
        'primary_exposure': {'uniform_stale': 0, 'mixed_partial': 1},
        'matching_keys': ['query_id', 'transition', 'architecture', 'severity'],
        'primary_model': 'failure ~ partial_state + transition + domain + (1|query_id)',
        'primary_interpretation': 'mixed replica state versus uniformly stale replica at the same nominal delay',
        'secondary_explanatory_variables': ['realized_fraction', 'partial_topology',
                                             'updated_unit_count', 'total_unit_count'],
        'secondary_exploratory_model': 'failure ~ realized_fraction + transition + domain + (1|query_id)',
        'no_forced_monotonicity': True,
        'no_universal_percentage_categories': True,
        'unit_selection': 'SHA-256 rank by seed, query, transition, severity, plane, and unit identity; no claim/gold input',
        'deduplication': 'P_LOW and P_HIGH retained when distinct; P_MID only when distinct from both',
        'midpoint_rule': 'round_half_up(N / 2) = ceil(N / 2)',
    }


def coverage_gate(summary):
    """Frozen pilot coverage rule for the primary content PARTIAL comparison."""
    counts = summary['partial']['transition_domain']
    content_cells = [cell for cell in counts
                     if cell['transition'] in PRIMARY_CONTENT_TRANSITIONS]
    return (all(summary['partial']['content_domain_coverage'][transition] >= 4
                for transition in PRIMARY_CONTENT_TRANSITIONS)
            and all(cell['matched_distinct_queries'] >= 5 for cell in content_cells
                    if cell['matched_distinct_queries'] > 0)
            and all(summary['partial']['unique_query_counts_by_transition'][transition] >= 30
                    for transition in PRIMARY_CONTENT_TRANSITIONS))


def finalize(compiled, smoke, repeat):
    output = compiled['output']
    summary = coverage(compiled)
    emitted_ids = {scenario['scenario_id'] for scenario in compiled['emitted']}
    ap1 = all(row['applicable'] for row in compiled['rows']
              if row['scenario_id'] in emitted_ids)
    ap2 = not compiled['failed_realizability'] and all(audit['realizable']
        for audit in compiled['audits'] if audit['scenario_id'] in emitted_ids)
    ap4 = coverage_gate(summary)
    gates = {'AP1': 'PASS' if ap1 else 'FAIL',
             'AP2': 'PASS' if ap2 else 'FAIL',
             'AP3': 'PASS' if smoke['agreement'] else 'FAIL',
             'AP4': 'PASS' if ap4 else 'FAIL',
             'AP5': 'PASS' if repeat['identical'] else 'FAIL',
             'coverage_rule': {'content_transition_domains_min': 4,
                               'matched_distinct_queries_per_represented_transition_domain_min': 5,
                               'matched_distinct_queries_per_content_transition_min': 30}}
    manifest = {
        'version': POLICY_VERSION, 'status': 'APPLICABILITY_VALIDATION_ONLY',
        'source_candidate_manifest_sha256': compiled['source_hash'],
        'candidate_table_sha256': compiled['candidate_hash'],
        'executable_manifest_sha256': digest(compiled['emitted']),
        'executable_scenarios': compiled['emitted'],
        'partial_plan_version': PARTIAL_PLAN_VERSION,
        'partial_topology_states': {'P_LOW': 'one unit updated',
                                    'P_MID': 'round_half_up(N/2) units updated, only if distinct',
                                    'P_HIGH': 'N-1 units updated'},
        'unit_selection_seed': compiled['config']['selection_seed'],
        'corpus_sha256': compiled['config']['corpus_sha256'],
        'scientific_outcomes_consulted': False,
        'scientific_campaign_frozen': False,
        'unchanged_fault_scenarios_recompiled': True,
    }
    for name, value in (('applicability_manifest.json', manifest),
                        ('coverage_summary.json', summary),
                        ('h4_analysis_policy.json', h4_policy()),
                        ('gate_summary.json', gates)):
        (output / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    (output / 'summary.md').write_text(
        '# Scenario applicability v2\n\n'
        f"Considered {summary['considered']} candidates; {summary['applicable']} are "
        f"applicable and realizable, {summary['excluded']} are excluded.\n\n"
        'Primary PARTIAL uses mixed topology states rather than universal percentages. '
        'Every emitted PARTIAL case has an executable, matched uniform-delay control. '
        'Unit selection is seeded, stable, and relevance-blind.\n\n'
        + '\n'.join(f'- {key}: {gates[key]}' for key in ('AP1', 'AP2', 'AP3', 'AP4', 'AP5'))
        + '\n\nNo scientific campaign was run. CF3 remains the frozen partial-ACL compound '
          'and was recompiled without changing its compound semantics.\n')
    return gates, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(OUTPUT))
    parser.add_argument('--fingerprint-only', action='store_true')
    args = parser.parse_args()
    if args.fingerprint_only:
        print(json.dumps(fingerprint(compile_all(write=False, perform_dry_run=False)),
                         sort_keys=True))
        return
    compiled = compile_all(args.output)
    smoke = topology_smoke(compiled)
    (compiled['output'] / 'runtime_agreement_smoke.json').write_text(
        json.dumps(smoke, indent=2, sort_keys=True) + '\n')
    env = dict(os.environ)
    env['PYTHONPATH'] = str(Path('src').resolve()) + os.pathsep + env.get('PYTHONPATH', '')
    fresh = subprocess.run([sys.executable, '-m',
        'ragstate.campaign.scenario_applicability_v2', '--fingerprint-only'],
        capture_output=True, text=True, env=env)
    fresh_hash = json.loads(fresh.stdout) if fresh.returncode == 0 else None
    repeat = {'identical': fresh_hash == fingerprint(compiled),
              'fresh_process_exit': fresh.returncode,
              'local': fingerprint(compiled), 'fresh': fresh_hash,
              'stderr': fresh.stderr}
    (compiled['output'] / 'determinism_audit.json').write_text(
        json.dumps(repeat, indent=2, sort_keys=True) + '\n')
    gates, summary = finalize(compiled, smoke, repeat)
    print(json.dumps({'gates': gates,
                      'counts': {key: summary[key] for key in ('considered', 'applicable', 'excluded')},
                      'partial': {key: summary['partial'][key] for key in
                                  ('scenario_count', 'matched_pair_count', 'topology_counts',
                                   'content_cell_min', 'content_cell_median', 'content_cell_max')}},
                     indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
