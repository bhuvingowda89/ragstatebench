"""Independent v3 enterprise corpus with heterogeneous sources and frozen gold."""
from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass

from .ledger import WorldLedger
from .models import DocumentVersion, Principal
from .scientific_v2 import DOMAINS, FAMILY_COUNTS, TRANSITIONS, LOGICAL_TIME, V2Query, _question

CORPUS_VERSION = 'v3'
ARCHETYPES = (
    'policy_memo', 'wiki_page', 'engineering_runbook', 'customer_support',
    'legal_notice', 'internal_email', 'incident_ticket', 'change_note',
)
DOMAIN_ARCHETYPES = {
    'hr': ('policy_memo','wiki_page','internal_email','incident_ticket'),
    'finance': ('policy_memo','wiki_page','incident_ticket','internal_email','change_note'),
    'engineering': ('engineering_runbook','wiki_page','incident_ticket','change_note'),
    'security': ('policy_memo','engineering_runbook','incident_ticket','wiki_page'),
    'sales': ('customer_support','wiki_page','internal_email','change_note'),
    'legal': ('legal_notice','policy_memo','internal_email','change_note'),
}
SOURCE_AUTHORITY = {
    'policy_memo':4,'wiki_page':2,'engineering_runbook':3,'customer_support':3,
    'legal_notice':4,'internal_email':2,'incident_ticket':2,'change_note':3,
}
VERSION_CHANGES = ('threshold_change','scope_expansion','exception_removal',
                   'effective_date_change','escalation_path_change','renamed_entity',
                   'procedural_step_change','deleted_clause')
RELATED_ROLE = {
    'hr':'approved exception', 'finance':'contractual amendment',
    'engineering':'environment override', 'security':'escalation matrix override',
    'sales':'customer-tier addendum', 'legal':'jurisdiction addendum',
}


@dataclass(frozen=True)
class V3Query(V2Query):
    evidence_relation: str


def logical_id(domain, topic_index:int, variant:int, kind:str)->str:
    return f'V3-{domain.name.upper()}-{topic_index:02d}-{variant//3}-{variant%3}-{kind}'


def tenant_for(world_seed:int,domain_index:int,topic_index:int,variant:int)->str:
    # Region and audience both vary within each tenant; neither axis determines tenant.
    bit=(world_seed + domain_index + topic_index + variant//3 + variant%3)%2
    return ('tenant_a','tenant_b')[bit]


def _header(archetype,domain,topic,axis1,axis2,kind,version):
    label=f'{topic.subject.title()} / {axis1.title()} {axis2.title()}'
    if archetype=='policy_memo':return f'POLICY MEMO — {label}\nOwner: {domain.name.title()} governance. Approved cycle {2024+version}. Scope review follows.'
    if archetype=='wiki_page':return f'{label}\nKnowledge base / maintained by {domain.name.title()} operations. Last reviewed in cycle {2024+version}.'
    if archetype=='engineering_runbook':return f'RUNBOOK: {label}\nService owner: {domain.name.title()} response. Revision {version}; approved for active operations.'
    if archetype=='customer_support':return f'Account support brief — {label}\nCustomer-facing operations / approved guidance, revision {version}.'
    if archetype=='legal_notice':return f'COMPLIANCE NOTICE: {label}\nCounsel signoff recorded for cycle {2024+version}; applicability reviewed.'
    if archetype=='internal_email':return f'From: {domain.name.title()} operations\nSubject: approved guidance on {label}\nSent in review cycle {2024+version}.'
    if archetype=='incident_ticket':return f'Case record / {label}\nStatus: approved resolution. Handling team: {domain.name.title()}. Revision {version}.'
    return f'Change note {version}: {label}\nApproval recorded by {domain.name.title()} owner; active release guidance.'


def _notes(domain,topic,axis1,axis2,kind,archetype,version_change):
    scope=f'{axis1} {axis2}'
    context=topic.context
    subject=topic.subject
    type_note={
        'policy_memo':'The governance owner records the interpretation in the policy register before teams adopt the rule.',
        'wiki_page':'This page summarizes the active workflow and links to the underlying approved source for audit.',
        'engineering_runbook':'Operators confirm preconditions in the service dashboard and record the action in the change log.',
        'customer_support':'Account teams use the approved schedule when explaining the decision to the customer contact.',
        'legal_notice':'Counsel checks applicability before treating the requirement as binding for this jurisdiction.',
        'internal_email':'The sender requests that recipients keep the approval thread with the operational record.',
        'incident_ticket':'The case owner attaches evidence of review before closing the incident or ticket.',
        'change_note':'Release owners document compatibility and migration implications before adopting this revision.',
    }[archetype]
    notes=[
        f'Applicability is limited to {scope}; a neighboring scope can have a different approved value. Teams must verify the requester and location before acting on {subject}.',
        f'The {context} record is the normal place to attach supporting evidence. Its owner checks that the relevant approval chain was followed before the case is finalized.',
        f'When a question is escalated, the local operations lead compares this guidance with the currently approved register. Informal summaries do not replace that check.',
        f'{type_note} The record of that review should name the responsible team and the date of the decision.',
        f'A nearby {axis1} case may look similar but use a different audience rule. Reviewers should confirm whether the request belongs to {axis2} before applying any number.',
        f'The approved schedule is reviewed when responsibilities or operating conditions change. Until a replacement is published, teams continue to use the effective record.',
        f'If supporting documentation is incomplete, the owner pauses the request and seeks clarification. This avoids applying a limit to the wrong {context} case.',
        f'For audit purposes, the decision trail should show who checked scope, who approved the action, and what evidence was available at the time.',
        f'This material also appears in related guidance for neighboring groups. Those references are useful context but do not override the current {scope} record.',
        f'The team should explain the applicable condition in plain language to the requester. It should preserve the approval reference alongside the outcome.',
        f'Operational examples may involve both a standard path and an exception path. The documented exception must be approved separately before it changes the standard path.',
        f'During routine review, the owner compares recent cases for {subject} with the approved wording and corrects any outdated local summaries.',
        f'Questions about ownership go to the {domain.name.title()} lead. The lead can request specialist review when the facts do not fit the ordinary pattern.',
        f'Recorded dates establish when this guidance was in force. A later draft is not effective merely because its timestamp is newer.',
        f'The {context} workflow should avoid copying a decision from another tenant. Each organization maintains its own approval and evidence history.',
        f'When operations hand off a case, the next team checks the source record rather than relying solely on a brief chat or an earlier ticket.',
        f'{type_note} Any exception remains bounded by the scope and approval recorded with the case.',
        f'Historical notes explain why the current rule changed, but they are not themselves the operative requirement for a new request.',
    ]
    if version_change=='scope_expansion':notes[0]+= ' The current approval covers the full listed scope.'
    if version_change=='exception_removal':notes[10]=f'An earlier broad exception is retired for {scope}; only a separately approved exception remains applicable.'
    if version_change=='escalation_path_change':notes[12]=f'Escalation now goes to the {domain.name.title()} owner before any operational override is applied.'
    if version_change=='procedural_step_change':notes[3]+= ' The owner review step is mandatory.'
    if version_change=='deleted_clause':notes[14]=f'Only the active {context} condition is retained; a withdrawn supplementary clause no longer applies.'
    return notes


def _format_document(archetype, header, paragraphs):
    """Render eight distinct source layouts; the fact keeps its chosen position."""
    if archetype=='internal_email':
        body=['Hello team,', *paragraphs,
              'Please retain the approval record and send corrections to the owner. Regards, Operations.']
        return header+'\n\n'+'\n\n'.join(body)
    if archetype=='incident_ticket':
        sections=['Summary','Investigation','Timeline','Resolution','Follow-up']
        lines=[header]
        for i,paragraph in enumerate(paragraphs):
            heading=sections[min(len(sections)-1,i*len(sections)//len(paragraphs))]
            lines.append(f'{heading} entry {i+1}: {paragraph}')
        return '\n\n'.join(lines)
    if archetype=='change_note':
        sections=['Change summary','Compatibility','Deprecation','Migration guidance']
        lines=[header]
        for i,paragraph in enumerate(paragraphs):
            heading=sections[min(len(sections)-1,i*len(sections)//len(paragraphs))]
            lines.append(f'- {heading}: {paragraph}')
        return '\n'.join(lines)
    if archetype=='engineering_runbook':
        sections=['Preconditions','Steps','Rollback','Escalation']
        lines=[header]
        for i,paragraph in enumerate(paragraphs):
            heading=sections[min(len(sections)-1,i*len(sections)//len(paragraphs))]
            lines.append(f'{heading} / step {i+1}\n{paragraph}')
        return '\n\n'.join(lines)
    if archetype=='customer_support':
        sections=['Customer tier','SLA','Exceptions','Escalation policy']
        lines=[header]
        for i,paragraph in enumerate(paragraphs):
            heading=sections[min(len(sections)-1,i*len(sections)//len(paragraphs))]
            lines.append(f'{heading}: {paragraph}')
        return '\n\n'.join(lines)
    if archetype=='legal_notice':
        sections=['Applicability','Requirement','Jurisdiction','Effective date','Amendment history']
        lines=[header]
        for i,paragraph in enumerate(paragraphs):
            heading=sections[min(len(sections)-1,i*len(sections)//len(paragraphs))]
            lines.append(f'Clause {i+1} — {heading}. {paragraph}')
        return '\n\n'.join(lines)
    if archetype=='wiki_page':
        sections=['Overview','Procedure','Notes','Related systems']
        lines=[header]
        for i,paragraph in enumerate(paragraphs):
            heading=sections[min(len(sections)-1,i*len(sections)//len(paragraphs))]
            lines.append(f'## {heading}\n{paragraph}')
        return '\n\n'.join(lines)
    sections=['Purpose','Scope','Rule','Exceptions','Effective date']
    lines=[header]
    for i,paragraph in enumerate(paragraphs):
        heading=sections[min(len(sections)-1,i*len(sections)//len(paragraphs))]
        lines.append(f'{heading}\n{paragraph}')
    return '\n\n'.join(lines)


def _content(domain,topic,axis1,axis2,kind,value,version,archetype,bucket,position,version_change,seed):
    rng=random.Random(seed)
    target={'short':85,'medium':230,'long':430}[bucket]+rng.randrange(-18,19)
    subject=topic.subject if version_change!='renamed_entity' or version>1 else f'legacy {topic.subject}'
    label=RELATED_ROLE[domain.name] if kind=='EXC' else 'standard rule'
    core=(f'For {axis1} {axis2}, the {label} governing {subject} sets the '
          f'{topic.measure} at {value} {topic.unit}. '
          f'This figure applies to {topic.context} after the responsible owner confirms the scope.')
    if kind=='EXC':
        core+=f' This {label} modifies the default only after documented approval.'
    else:
        core+=' A separate approved exception may change the result for a qualifying case.'
    scope=f'{axis1} {axis2}'
    change_lines={
        'scope_expansion':('The operative scope is a reviewed pilot subset of '+scope+'.',
                           'The operative scope now includes all '+scope+' cases.'),
        'exception_removal':('A broad exception may be applied without individual approval.',
                             'The earlier broad exception is retired; individual approval is required.'),
        'effective_date_change':('The rule takes effect after the next review cycle.',
                                  'The rule takes effect in the current review cycle.'),
        'escalation_path_change':('Escalation goes first to a local line manager.',
                                  'Escalation goes first to the '+domain.name.title()+' owner.'),
        'procedural_step_change':('One owner check completes the procedure.',
                                  'A second recorded owner signoff completes the procedure.'),
        'deleted_clause':('A supplementary clause permits bypassing ordinary review.',
                           'The former bypass clause is removed from the active procedure.'),
    }
    if version_change in change_lines:
        core+=' '+change_lines[version_change][0 if version==1 else 1]
    header=_header(archetype,domain,topic,axis1,axis2,kind,version)
    notes=_notes(domain,topic,axis1,axis2,kind,archetype,version_change)
    order=list(range(len(notes)));rng.shuffle(order)
    filler=[];count=len((header+' '+core).split())
    for i in order:
        if count>=target:break
        filler.append(notes[i]);count+=len(notes[i].split())
    if position=='beginning':paragraphs=[core]+filler
    elif position=='middle':
        cut=len(filler)//2;paragraphs=filler[:cut]+[core]+filler[cut:]
    else:paragraphs=filler+[core]
    return _format_document(archetype,header,paragraphs)


def generate_v3_world(world_seed:int):
    rng=random.Random(world_seed)
    world=WorldLedger();metadata={}
    for tenant in ('tenant_a','tenant_b'):
        for domain in DOMAINS:
            world.add_principal(Principal(f'v3-{tenant}-{domain.name}-specialist',tenant,
                                          frozenset({'employee',domain.name})))
            world.add_principal(Principal(f'v3-{tenant}-{domain.name}-generalist',tenant,
                                          frozenset({'employee'})))
    for di,domain in enumerate(DOMAINS):
        options=DOMAIN_ARCHETYPES[domain.name]
        for ti,topic in enumerate(domain.topics):
            for variant in range(6):
                axis1=domain.axis_one[variant//3];axis2=domain.axis_two[variant%3]
                tenant=tenant_for(world_seed,di,ti,variant)
                for ki,kind in enumerate(('BASE','EXC')):
                    logical=logical_id(domain,ti,variant,kind)
                    archetype=options[(ti+variant+ki)%len(options)]
                    bucket=('short','medium','medium','long','short','medium','long','medium','short','medium')[(di*3+ti*6+variant+ki)%10]
                    position=('beginning','middle','end')[(di+ti+variant+ki)%3]
                    authority=SOURCE_AUTHORITY[archetype]
                    versioned=kind=='BASE' and variant%3==0
                    change=VERSION_CHANGES[(di*5+ti*2+variant)%len(VERSION_CHANGES)] if versioned else None
                    value=topic.base+(variant//3)*3+(variant%3)*2+ti+(4 if kind=='EXC' else 0)+rng.randrange(2)
                    groups=frozenset({domain.name}) if kind=='EXC' else frozenset({'employee'})
                    if versioned:
                        previous_value=value-3 if change=='threshold_change' else value
                        old=_content(domain,topic,axis1,axis2,kind,previous_value,1,archetype,
                                     bucket,position,change,rng.randrange(2**32))
                        if change=='exception_removal':old+='\n\nPreviously, a broad exception was available without separate approval.'
                        if change=='deleted_clause':old+='\n\nFormer supplementary clause: the owner may bypass the ordinary review.'
                        if change=='escalation_path_change':old=old.replace('Escalation now goes to','Escalation previously went to')
                        world.add_document(DocumentVersion(f'{logical}-v1',logical,tenant,1,0,valid_until=5,
                            authority_level=authority,content=old,required_groups=groups))
                    version=2 if versioned else 1
                    content=_content(domain,topic,axis1,axis2,kind,value,version,archetype,
                                     bucket,position,change,rng.randrange(2**32))
                    world.add_document(DocumentVersion(f'{logical}-v{version}',logical,tenant,version,
                        5 if versioned else 0,supersedes=f'{logical}-v1' if versioned else None,
                        authority_level=authority,content=content,required_groups=groups))
                    metadata[logical]={'logical_document_id':logical,'domain':domain.name,'topic_index':ti,
                        'topic':topic.subject,'variant':variant,'axis_one':axis1,'axis_two':axis2,
                        'tenant_id':tenant,'kind':kind,'archetype':archetype,'length_bucket':bucket,
                        'evidence_position':position,'version_sensitive':versioned,
                        'version_change_type':change,'authority_level':authority,
                        'related_logical_document_id':logical_id(domain,ti,variant,'EXC' if kind=='BASE' else 'BASE')}
            # One lower-authority, same-topic operational summary or draft per topic.
            logical=f'V3-{domain.name.upper()}-{ti:02d}-AUX'
            aux_kind='draft' if ti%2==0 else 'faq'
            archetype='wiki_page' if aux_kind=='faq' else ('internal_email' if domain.name in ('hr','legal','sales') else 'change_note')
            topic_axis1=domain.axis_one[ti%2];topic_axis2=domain.axis_two[ti%3]
            content=_content(domain,topic,topic_axis1,topic_axis2,'BASE',topic.base+ti+7,1,
                             archetype,'medium','middle',None,rng.randrange(2**32))
            content=('DRAFT — not approved.\n\n' if aux_kind=='draft' else 'FAQ summary — verify the approved source.\n\n')+content
            world.add_document(DocumentVersion(f'{logical}-v1',logical,'tenant_a' if ti%2==0 else 'tenant_b',
                1,0,authority_level=1,content=content,required_groups=frozenset({'employee'})))
            metadata[logical]={'logical_document_id':logical,'domain':domain.name,'topic_index':ti,
                'topic':topic.subject,'variant':None,'axis_one':topic_axis1,'axis_two':topic_axis2,
                'tenant_id':'tenant_a' if ti%2==0 else 'tenant_b','kind':aux_kind,
                'archetype':archetype,'length_bucket':'medium','evidence_position':'middle',
                'version_sensitive':False,'version_change_type':None,'authority_level':1,
                'related_logical_document_id':None}
    return world,metadata


def generate_v3_queries(world,query_seed:int):
    rng=random.Random(query_seed);queries=[];seen=set()
    for domain in DOMAINS:
        allocation=[family for family,count in FAMILY_COUNTS.items() for _ in range(count)]
        rng.shuffle(allocation)
        for j,family in enumerate(allocation):
            ti,variant=(j//6)%5,j%6
            base=logical_id(domain,ti,variant,'BASE')
            exc=logical_id(domain,ti,variant,'EXC')
            ids=(base,exc) if family=='multi_document' else ((exc,) if family=='negative' else (base,))
            tenant=world.document_at(base,LOGICAL_TIME).tenant_id
            principal_tenant=({'tenant_a':'tenant_b','tenant_b':'tenant_a'}[tenant]
                              if family=='negative' else tenant)
            role='generalist' if family=='natural_descriptive' and j%3==0 else 'specialist'
            principal=f'v3-{principal_tenant}-{domain.name}-{role}'
            text=None;attempts=0
            for attempt in range(4):
                candidate=_question(domain,domain.topics[ti],variant,family,'BASE',j+attempt,base)
                if family=='multi_document':
                    candidate=candidate.replace('documented exception',RELATED_ROLE[domain.name]).replace(
                        'approved exception',RELATED_ROLE[domain.name])
                if candidate not in seen:text=candidate;attempts=attempt;break
            if text is None:raise ValueError('Duplicate query text')
            seen.add(text)
            required=tuple(world.document_at(lid,LOGICAL_TIME).document_id for lid in ids
                           if world.allowed(principal,lid,LOGICAL_TIME))
            claims=[]
            for lid in ids:
                doc=world.document_at(lid,LOGICAL_TIME)
                if not world.allowed(principal,lid,LOGICAL_TIME):continue
                match=re.search(r'\b(\d+) '+re.escape(domain.topics[ti].unit)+r'\b',doc.content)
                if match is None:raise ValueError(f'No atomic value in {doc.document_id}')
                claims.append({'logical_document_id':lid,'document_id':doc.document_id,
                    'claim_type':'approved_exception' if lid.endswith('EXC') else 'default_rule',
                    'measure':domain.topics[ti].measure,'value':int(match.group(1)),
                    'unit':domain.topics[ti].unit,'precedence':2 if lid.endswith('EXC') else 1})
            target=world.document_at(ids[0],LOGICAL_TIME)
            relation=(RELATED_ROLE[domain.name].replace(' ','_') if family=='multi_document' else 'single_source')
            queries.append(V3Query(f'v3-q-{domain.name}-{j:02d}',text,domain.name,family,principal,
                principal_tenant,LOGICAL_TIME,ids,required,tuple(claims),bool(required),
                family=='multi_document',family in ('multi_document','negative') or j%4==1,
                any(world.document_at(lid,LOGICAL_TIME).version>1 for lid in ids),
                TRANSITIONS[j%len(TRANSITIONS)],target.document_id,attempts,None,relation))
    return queries


def corpus_hash(world,metadata,queries):
    payload={'documents':[asdict(world.documents[k]) for k in sorted(world.documents)],
             'principals':[asdict(world.principals[k]) for k in sorted(world.principals)],
             'metadata':[metadata[k] for k in sorted(metadata)],
             'queries':[asdict(q) for q in queries]}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),
        default=lambda x:sorted(x) if isinstance(x,(set,frozenset)) else x).encode()).hexdigest()
