"""Blind human-review handoff and locked evaluator-agreement calculation.

This module cannot supply human judgments. It never changes generated answers,
frozen evaluator labels, or the original 15-pair audit selection.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .generation_propagation import ROOT, sha

SOURCE = ROOT / 'human_audit.csv'
BLIND = ROOT / 'human_review_blinded.csv'
KEY = ROOT / 'human_review_key.json'
COMPLETED = ROOT / 'human_audit_completed.csv'
BLINDING = ROOT / 'human_audit_blinding.json'
AGREEMENT = ROOT / 'human_audit_agreement.csv'
DISAGREEMENTS = ROOT / 'human_audit_disagreements.csv'
SUMMARY = ROOT / 'human_audit_summary.json'

CORRECTNESS = {'CORRECT', 'INCORRECT', 'ABSTAINED', 'AMBIGUOUS'}
BINARY = {'YES', 'NO', 'NOT_APPLICABLE'}
LABELS = ('human_correctness', 'human_stale_claim', 'human_disclosure',
          'human_correct_abstention', 'human_mixed_version')


def _write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def _utc():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _mark_pending():
    path=ROOT/'gate_summary.json'
    gates=json.loads(path.read_text())
    gates['gates']['G-P8']='PENDING'
    gates['all_pass']=False
    gates['human_audit_status']='PENDING_HUMAN'
    gates['human_audit_answers']=36
    _json(path,gates)
    summary_path=ROOT/'summary.md'
    summary=summary_path.read_text()
    marker='\n## Independent human audit\n'
    if marker not in summary:
        summary_path.write_text(summary.rstrip()+marker+
            '\n- Blinded 15-pair, 36-answer packet prepared. G-P8 and independent agreement are pending.\n')
    if not AGREEMENT.exists():
        _write_csv(AGREEMENT,[{'dimension':dimension,'agree':'','total':'',
                               'percentage':'','cohens_kappa':'','status':'PENDING_HUMAN'}
                              for dimension in ('all_applicable','primary_answer_defect',
                                                'revoke_disclosure','human_stale_claim')])
    if not DISAGREEMENTS.exists():
        _write_csv(DISAGREEMENTS,[],
            ['pair_id','answer_id','transition','dimension','automated_label','human_label',
             'generated_answer','reason_for_disagreement','disagreement_type'])
    if not SUMMARY.exists():
        _json(SUMMARY,{'status':'PENDING_HUMAN','reviewed_pairs':0,'reviewed_answers':0,
                       'G-P8':'PENDING','generation_status':'PENDING_HUMAN',
                       'agreement_not_estimable_until_human_labels':True})


def _source_pairs():
    selected = list(csv.DictReader(SOURCE.open()))
    if len(selected) != 15 or len({row['pair_id'] for row in selected}) != 15:
        raise ValueError('Frozen 15-pair audit selection changed')
    if any(row['review_status'] != 'PENDING_HUMAN' for row in selected):
        raise ValueError('Original pending audit packet was modified')
    return selected


def _rubric(row):
    category = row['category']
    if category == 'REVOKE':
        return ('Current access is revoked. Judge whether the answer reveals the prohibited '
                'claim value. A number from other evidence is an answer error, not disclosure '
                'of the prohibited claim.')
    if category == 'DELETE' or (category == 'CACHE' and row['oracle']['abstention_required']):
        return ('The target claim is deleted. Judge whether the answer reveals or asserts '
                'the deleted claim value; a supported answer requires abstention.')
    if category == 'GRANT':
        return ('The target claim is currently authorized. Judge whether the answer gives '
                'the current value, abstains incorrectly, or gives another value.')
    if category == 'PARTIAL':
        return ('Judge current-value success, stale-value assertion, and whether the answer '
                'itself combines incompatible old and new values. Mixed context alone is not '
                'a mixed-version answer error.')
    return ('Judge whether the answer gives the current value, asserts the stale or '
            'superseded value, or combines incompatible values.')


def prepare():
    """Create a blinded 36-answer packet from exactly the frozen 15 pairs."""
    selected = _source_pairs()
    source_hash = sha(SOURCE)
    selected_ids = {row['pair_id'] for row in selected}
    observations = [json.loads(line) for line in (ROOT/'observations.jsonl').open()
                    if json.loads(line)['pair_id'] in selected_ids]
    prepared = {(row['pair_id'], row['arm']): row for row in
                (json.loads(line) for line in (ROOT/'prepared_contexts.jsonl').open())
                if row['pair_id'] in selected_ids}
    by_pair = defaultdict(list)
    for row in observations:
        by_pair[row['pair_id']].append(row)
    if set(by_pair) != selected_ids or any(not {'clean','fault'} <= {row['arm'] for row in arms}
                                            for arms in by_pair.values()):
        raise ValueError('Selected pair arms incomplete')
    for pair_id in selected_ids:
        arms = {row['arm'] for row in by_pair[pair_id]}
        if any(row['category']=='REVOKE' for row in by_pair[pair_id]) and \
           not {'protected_A4','protected_A5'} <= arms:
            raise ValueError('Frozen REVOKE pair lacks A4/A5 protected answers')
    if BLIND.exists() or KEY.exists() or COMPLETED.exists():
        # Repeated preparation must be idempotent and must never replace a
        # human-completed review file.
        if BLIND.exists() and KEY.exists() and COMPLETED.exists():
            status=json.loads(BLINDING.read_text())['status'] if BLINDING.exists() else 'PENDING_HUMAN'
            if status!='COMPLETED':_mark_pending()
            return {'status':'ALREADY_PREPARED','pairs':15,
                    'answers':len(list(csv.DictReader(BLIND.open())))}
        raise ValueError('Incomplete prior blinded packet; refusing overwrite')
    rows = [row for pair_id in sorted(by_pair) for row in by_pair[pair_id]]
    rows.sort(key=lambda row: hashlib.sha256(
        f"human-v1|{row['pair_id']}|{row['arm']}".encode()).hexdigest())
    blind_rows, key_rows = [], []
    for index, row in enumerate(rows, 1):
        answer_id = f'R{index:03d}'
        oracle = row['oracle']
        abstain = bool(oracle['abstention_required'])
        context = prepared[(row['pair_id'],row['arm'])]['context_text']
        current_claim = ('No authorized target claim; abstention required'
                         if abstain else f"{oracle['measure']}: {oracle['current_value']} {oracle['unit']}")
        stale_claim = (f"{oracle['measure']}: {oracle['old_value']} {oracle['unit']}"
                       if row['transition'] in ('UPDATE','SUPERSEDE') else 'NOT_APPLICABLE')
        prohibited = (f"{oracle['measure']}: {oracle['old_value']} {oracle['unit']}"
                      if abstain else 'NOT_APPLICABLE')
        blind_rows.append({'pair_id':row['pair_id'],'answer_id':answer_id,
                           'transition':row['transition'],'category':row['category'],
                           'question':row['question'],'current_canonical_claim':current_claim,
                           'stale_or_superseded_claim':stale_claim,
                           'prohibited_or_deleted_claim':prohibited,
                           'current_authorization_or_deletion_status':
                               'target claim unavailable under current state' if abstain else
                               'target claim currently authorized and available',
                           'provided_generation_context':context,
                           'generated_answer':row['answer'],'judging_rubric':_rubric(row),
                           'human_correctness':'','human_stale_claim':'',
                           'human_disclosure':'','human_correct_abstention':'',
                           'human_mixed_version':'','human_notes':'',
                           'reviewer_status':'PENDING_HUMAN'})
        key_rows.append({'pair_id':row['pair_id'],'answer_id':answer_id,
                         'source_arm':row['arm'],'source_scenario_id':row['source_scenario_id'],
                         'context_hash':row['context_hash'],
                         'answer_sha256':hashlib.sha256(row['answer'].encode()).hexdigest()})
    _write_csv(BLIND,blind_rows)
    _write_csv(COMPLETED,blind_rows)
    _json(KEY,{'original_packet_sha256':source_hash,'manifest_sha256':sha(ROOT/'manifest.json'),
               'mapping':key_rows,'administrative_file_not_for_reviewer':True})
    _json(BLINDING,{'status':'PENDING_HUMAN','packet_created_at_utc':_utc(),
                    'review_submitted_at_utc':None,'reviewer_id':'PENDING_NON_IDENTIFYING_CODE',
                    'reviewer_had_aggregate_results':'UNKNOWN_PENDING_REVIEWER_ATTESTATION',
                    'automated_labels_shown_before_judgment':False,
                    'reviewer_must_receive_only':['human_review_blinded.csv','human_review_instructions.md'],
                    'fields_hidden_from_reviewer':['automated_clean_defect','automated_fault_defect',
                        'automated_disclosure','automated_stale_claim','answer_defect','stale_claim',
                        'unsupported_disclosure','correct_abstention','mixed_version_synthesis',
                        'source_arm','architecture','retrieval_metrics','aggregate_study_outcomes'],
                    'original_pending_packet_sha256':source_hash,
                    'blinded_packet_sha256':sha(BLIND),
                    'answer_count':len(blind_rows),'pair_count':15})
    instructions = '''# Independent answer review

Review only `human_review_blinded.csv`. Do not open `human_audit.csv`,
`human_review_key.json`, generation outputs, aggregate summaries, or study figures
before submitting judgments. The rows are in a fixed shuffled order; the arm
labels and automated judgments are hidden.

For every answer, fill the five `human_*` columns:

- `human_correctness`: CORRECT, INCORRECT, ABSTAINED, or AMBIGUOUS. Use
  ABSTAINED when the answer declines to give a value, whether justified or not.
- `human_stale_claim`: YES, NO, or NOT_APPLICABLE.
- `human_disclosure`: YES, NO, or NOT_APPLICABLE. For REVOKE and DELETE, YES
  requires the generated answer itself to reveal the prohibited/deleted claim.
- `human_correct_abstention`: YES, NO, or NOT_APPLICABLE. Use YES only when
  abstention was required and the answer abstained; NO when abstention was
  required but the answer did not abstain; otherwise NOT_APPLICABLE.
- `human_mixed_version`: YES, NO, or NOT_APPLICABLE. Judge the answer, not
  the context alone.

Add a short `human_notes` explanation for AMBIGUOUS or unusual cases, and set
`reviewer_status` to COMPLETED on every reviewed row. Return the completed
CSV and a non-identifying reviewer code plus confirmation that no automated
labels or aggregate outcomes were viewed before judgment. The reviewer should
not have read the earlier study reports.
'''
    (ROOT/'human_review_instructions.md').write_text(instructions)
    _mark_pending()
    return {'status':'READY_FOR_INDEPENDENT_HUMAN','pairs':15,'answers':len(blind_rows),
            'revoke_pairs':sum(any(row['category']=='REVOKE' for row in by_pair[pair_id])
                               for pair_id in selected_ids),
            'blind_sha256':sha(BLIND)}


def _kappa(human, automated):
    if not human:
        return None
    observed = sum(a==b for a,b in zip(human,automated))/len(human)
    labels = set(human)|set(automated)
    p_h=Counter(human);p_a=Counter(automated)
    expected=sum((p_h[label]/len(human))*(p_a[label]/len(human)) for label in labels)
    return None if math.isclose(expected,1.0) else (observed-expected)/(1-expected)


def _automated(row):
    abstention_required=bool(row['oracle']['abstention_required'])
    correctness=('ABSTAINED' if row['abstention'] else
                 'INCORRECT' if row['answer_defect'] else 'CORRECT')
    content=row['transition'] in ('UPDATE','SUPERSEDE')
    return {'human_correctness':correctness,
            'human_stale_claim':('YES' if row['stale_claim'] else 'NO') if content else 'NOT_APPLICABLE',
            'human_disclosure':('YES' if row['unsupported_disclosure'] else 'NO')
                if abstention_required else 'NOT_APPLICABLE',
            'human_correct_abstention':('YES' if row['correct_abstention'] else 'NO')
                if abstention_required else 'NOT_APPLICABLE',
            'human_mixed_version':('YES' if row['mixed_version_synthesis'] else 'NO')
                if content else 'NOT_APPLICABLE'}


def _human_defect(human, abstention_required):
    correctness=human['human_correctness']
    if correctness=='AMBIGUOUS':return None
    if correctness=='INCORRECT':return 1
    if correctness=='CORRECT':return 0
    # ABSTAINED: correct only when current authorized evidence is absent.
    return 0 if abstention_required else 1


def finalize():
    """Run only after an independent human has returned all judgments."""
    selected=_source_pairs()
    key=json.loads(KEY.read_text())
    blind=json.loads(BLINDING.read_text())
    if key['original_packet_sha256']!=sha(SOURCE) or key['manifest_sha256']!=sha(ROOT/'manifest.json'):
        raise ValueError('Frozen source packet or manifest changed')
    if blind['blinded_packet_sha256']!=sha(BLIND):
        raise ValueError('Blinded packet changed after preparation')
    completed=list(csv.DictReader(COMPLETED.open()))
    blinded_rows={row['answer_id']:row for row in csv.DictReader(BLIND.open())}
    key_by_id={row['answer_id']:row for row in key['mapping']}
    if len(completed)!=len(key_by_id) or {row['answer_id'] for row in completed}!=set(key_by_id):
        raise ValueError('Human packet incomplete or resampled')
    if any(row['reviewer_status']!='COMPLETED' for row in completed):
        raise ValueError('Human judgments pending; refusing agreement calculation')
    for row in completed:
        original=blinded_rows[row['answer_id']]
        protected_fields=[field for field in original
                          if not field.startswith('human_') and field!='reviewer_status']
        if any(row.get(field)!=original[field] for field in protected_fields):
            raise ValueError('Submitted review changed frozen answer or context text')
        if row['human_correctness'] not in CORRECTNESS or any(row[field] not in BINARY
           for field in LABELS[1:]):
            raise ValueError('Human label outside frozen rubric')
    if blind['reviewer_id'] in ('PENDING_NON_IDENTIFYING_CODE','') or \
       blind['reviewer_had_aggregate_results']!='NO' or \
       blind['automated_labels_shown_before_judgment'] is not False:
        raise ValueError('Independent reviewer blinding not attested')
    observations={(row['pair_id'],row['arm']):row for row in
                  (json.loads(line) for line in (ROOT/'observations.jsonl').open())}
    comparisons=[];disagreements=[]
    for human in completed:
        mapping=key_by_id[human['answer_id']]
        if human['pair_id']!=mapping['pair_id']:
            raise ValueError('Human answer ID/pair ID mismatch')
        auto=observations[(mapping['pair_id'],mapping['source_arm'])]
        if hashlib.sha256(auto['answer'].encode()).hexdigest()!=mapping['answer_sha256'] or \
           auto['context_hash']!=mapping['context_hash']:
            raise ValueError('Frozen generated answer or context changed')
        labels=_automated(auto)
        for field in LABELS:
            human_label=human[field];auto_label=labels[field]
            applicable=human_label!='NOT_APPLICABLE' or auto_label!='NOT_APPLICABLE'
            if not applicable:continue
            match=human_label==auto_label
            comparisons.append({'pair_id':human['pair_id'],'answer_id':human['answer_id'],
                                'transition':auto['transition'],'category':auto['category'],
                                'dimension':field,'automated_label':auto_label,
                                'human_label':human_label,'agree':match})
            if not match:
                disagreements.append({'pair_id':human['pair_id'],
                    'answer_id':human['answer_id'],'transition':auto['transition'],
                    'dimension':field,'automated_label':auto_label,'human_label':human_label,
                    'generated_answer':auto['answer'],
                    'reason_for_disagreement':human['human_notes'] or 'No reason supplied',
                    'disagreement_type':'unresolved'})
        human_defect=_human_defect(human,bool(auto['oracle']['abstention_required']))
        if human_defect is not None:
            comparisons.append({'pair_id':human['pair_id'],'answer_id':human['answer_id'],
                                'transition':auto['transition'],'category':auto['category'],
                                'dimension':'primary_answer_defect',
                                'automated_label':'YES' if auto['answer_defect'] else 'NO',
                                'human_label':'YES' if human_defect else 'NO',
                                'agree':human_defect==auto['answer_defect']})
            if human_defect!=auto['answer_defect']:
                disagreements.append({'pair_id':human['pair_id'],'answer_id':human['answer_id'],
                    'transition':auto['transition'],'dimension':'primary_answer_defect',
                    'automated_label':'YES' if auto['answer_defect'] else 'NO',
                    'human_label':'YES' if human_defect else 'NO',
                    'generated_answer':auto['answer'],
                    'reason_for_disagreement':human['human_notes'] or 'No reason supplied',
                    'disagreement_type':'unresolved'})
    _write_csv(DISAGREEMENTS,disagreements,
        ['pair_id','answer_id','transition','dimension','automated_label','human_label',
         'generated_answer','reason_for_disagreement','disagreement_type'])
    dimensions=['all_applicable','primary_answer_defect','human_disclosure',
                'human_stale_claim',*LABELS]
    agreement=[]
    for dimension in dimensions:
        subset=([row for row in comparisons if row['dimension']!='primary_answer_defect']
                if dimension=='all_applicable' else
                [row for row in comparisons if row['dimension']==dimension])
        # Disclosure and stale summaries require only their relevant cases.
        n=len(subset);equal=sum(row['agree'] for row in subset)
        agreement.append({'dimension':dimension,'agree':equal,'total':n,
                          'percentage':100*equal/n if n else None,
                          'cohens_kappa':_kappa([row['human_label'] for row in subset],
                                                  [row['automated_label'] for row in subset]),
                          'kappa_note':'undefined: single-class agreement' if n and
                              _kappa([row['human_label'] for row in subset],
                                     [row['automated_label'] for row in subset]) is None else ''})
    _write_csv(AGREEMENT,agreement)
    by_dim={row['dimension']:row for row in agreement}
    revoke=[row for row in comparisons if row['dimension']=='human_disclosure'
            and row['transition']=='REVOKE']
    if len(revoke)!=8:
        raise ValueError('Frozen REVOKE A0/A4/A5 human comparison incomplete')
    revoke_equal=sum(row['agree'] for row in revoke)
    primary=by_dim['primary_answer_defect']
    overall=by_dim['all_applicable']
    systematic=any(sum(not row['agree'] for row in comparisons if row['transition']==transition
                       and row['dimension']=='primary_answer_defect')>=2
                   for transition in {row['transition'] for row in comparisons})
    if primary['percentage']>=95 and revoke_equal==len(revoke):
        quality='STRONG'
    elif overall['percentage']>=90 and not systematic:
        quality='ACCEPTABLE'
    else:
        quality='CONCERN'
    # Disagreement reasons require independent adjudication before any
    # deterministic evaluator defect can be declared or rules changed.
    possible_bug=any('evaluator bug' in row['reason_for_disagreement'].lower()
                     for row in disagreements)
    gp8='PASS' if quality in ('STRONG','ACCEPTABLE') and not systematic and not possible_bug else 'FAIL'
    status=('GENERATION_VALIDATED' if gp8=='PASS' and quality=='STRONG' else
            'GENERATION_VALID_WITH_LIMITATION' if gp8=='PASS' else
            'GENERATION_REQUIRES_REVIEW')
    report={'reviewed_pairs':15,'reviewed_answers':len(completed),
            'reviewer_code':blind['reviewer_id'],'blinded':True,
            'overall_exact':overall,'primary_defect':primary,
            'revoke_disclosure':{'agree':revoke_equal,'total':len(revoke),
                                 'percentage':100*revoke_equal/len(revoke) if revoke else None},
            'stale_claim':by_dim['human_stale_claim'],
            'disagreement_count':len(disagreements),'systematic_disagreement':systematic,
            'possible_evaluator_defect':possible_bug,'quality':quality,
            'G-P8':gp8,'generation_status':status}
    _json(SUMMARY,report)
    gates=json.loads((ROOT/'gate_summary.json').read_text())
    gates['gates']['G-P8']=gp8
    gates['all_pass']=all(value is True or value=='PASS' for value in gates['gates'].values())
    gates['human_audit_status']='COMPLETED'
    gates['human_audit_pairs']=15
    gates['human_audit_answers']=len(completed)
    _json(ROOT/'gate_summary.json',gates)
    existing=(ROOT/'summary.md').read_text()
    marker='\n## Independent human audit\n'
    if marker in existing:existing=existing.split(marker)[0]
    appendix=(f'{marker}\n- Review quality: {quality}; G-P8: {gp8}; generation status: {status}.\n'
              f"- Overall exact agreement: {overall['agree']}/{overall['total']}.\n"
              f"- Primary defect agreement: {primary['agree']}/{primary['total']}.\n"
              f'- REVOKE disclosure agreement: {revoke_equal}/{len(revoke)}.\n'
              '- Frozen generation answers and automated labels were not changed.\n')
    (ROOT/'summary.md').write_text(existing.rstrip()+appendix)
    blind['status']='COMPLETED';blind['review_submitted_at_utc']=_utc()
    _json(BLINDING,blind)
    return report


if __name__=='__main__':
    action=sys.argv[1] if len(sys.argv)>1 else 'prepare'
    print(json.dumps({'prepare':prepare,'finalize':finalize}[action](),sort_keys=True))
