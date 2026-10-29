"""Construction and confounding checks for the frozen v3 corpus."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import replace

from ragstate.retrieval.bm25 import tokenize
from .scientific_v2 import DOMAINS, LOGICAL_TIME, _question
from .scientific_v3 import RELATED_ROLE
from .v2_validation import _longest_phrase

IDENTIFIER=re.compile(r'V3-[A-Z]+-\d{2}-[01]-[012]-(?:BASE|EXC)(?:-v\d+)?')


def leakage_rows(world,metadata,queries):
    current={lid:world.document_at(lid,LOGICAL_TIME) for lid in metadata}
    by_domain={d.name:[doc for lid,doc in current.items() if metadata[lid]['domain']==d.name]
               for d in DOMAINS}
    term_freq={domain:Counter(term for d in docs for term in set(tokenize(d.content)))
               for domain,docs in by_domain.items()}
    rows=[]
    for q in queries:
        terms=set(tokenize(q.text))
        gold=[world.documents[doc_id] for doc_id in q.required_document_ids]
        nongold=[d for d in by_domain[q.domain] if d.document_id not in q.required_document_ids]
        unique=sorted(t for t in terms if term_freq[q.domain][t]==1 and
                      any(t in set(tokenize(d.content)) for d in gold))
        ids=IDENTIFIER.findall(q.text)
        gold_ids={d.logical_document_id for d in gold}|{d.document_id for d in gold}
        def coverage(doc):return len(terms&set(tokenize(doc.content)))/len(terms) if terms else 0.0
        best=max(nongold,key=lambda d:(coverage(d),d.document_id)) if nongold else None
        gold_overlap=sum(coverage(d) for d in gold)/len(gold) if gold else None
        nongold_overlap=coverage(best) if best else None
        title_overlap=any(d.content.split('\n',1)[0].lower() in q.text.lower() for d in gold)
        warning=q.family!='identifier_bearing' and (bool(ids) or bool(unique))
        rows.append({'query_id':q.query_id,'domain':q.domain,'family':q.family,
                     'answerable':int(q.answerable),'explicit_identifier':int(bool(ids)),
                     'gold_identifier_overlap':int(any(x in gold_ids for x in ids)),
                     'unique_gold_tokens':'|'.join(unique),'unique_gold_token_count':len(unique),
                     'unique_gold_token_leak':int(bool(unique)),
                     'exact_title_overlap':int(title_overlap),
                     'longest_shared_phrase_tokens':max((_longest_phrase(q.text,d.content) for d in gold),default=0),
                     'query_gold_overlap':gold_overlap,'query_best_nongold_overlap':nongold_overlap,
                     'gold_lexical_advantage':gold_overlap-nongold_overlap if gold_overlap is not None else None,
                     'best_nongold_document_id':best.document_id if best else None,
                     'construction_warning':int(warning),'regeneration_attempts':q.regeneration_attempts})
    return rows


def repair_lexical_leakage(world,metadata,queries):
    initial=leakage_rows(world,metadata,queries)
    revised=[];seen=set()
    for q,row in zip(queries,initial):
        if not row['construction_warning']:
            revised.append(q);seen.add(q.text);continue
        domain=next(d for d in DOMAINS if d.name==q.domain)
        parts=q.logical_document_ids[0].split('-')
        topic_index=int(parts[2]);variant=int(parts[3])*3+int(parts[4])
        repaired=None
        for attempt in range(1,7):
            text=_question(domain,domain.topics[topic_index],variant,q.family,'BASE',
                           attempt,q.logical_document_ids[0])
            if q.family=='multi_document':
                text=text.replace('documented exception',RELATED_ROLE[q.domain]).replace(
                    'approved exception',RELATED_ROLE[q.domain])
            trial=replace(q,text=text,regeneration_attempts=q.regeneration_attempts+attempt)
            if text not in seen and not leakage_rows(world,metadata,[trial])[0]['construction_warning']:
                repaired=trial;break
        final=repaired or replace(q,construction_warning='unresolved lexical leakage')
        revised.append(final);seen.add(final.text)
    return revised,leakage_rows(world,metadata,revised)


def validate_gold(world,metadata,queries):
    errors=[];seen=set()
    for q in queries:
        if q.text in seen:errors.append(q.query_id+': duplicate text')
        seen.add(q.text)
        if q.principal_id not in world.principals:
            errors.append(q.query_id+': missing principal');continue
        expected=[]
        for lid in q.logical_document_ids:
            doc=world.document_at(lid,q.logical_time)
            if doc is None:errors.append(q.query_id+': nonexistent target '+lid);continue
            if world.allowed(q.principal_id,lid,q.logical_time):expected.append(doc.document_id)
        if tuple(expected)!=q.required_document_ids:
            errors.append(q.query_id+': current authorized gold mismatch')
        if bool(expected)!=q.answerable:errors.append(q.query_id+': answerability mismatch')
        if len(expected)!=len(q.required_atomic_claims):errors.append(q.query_id+': incomplete claims')
        for claim in q.required_atomic_claims:
            doc=world.documents.get(claim['document_id'])
            if doc is None or doc.document_id not in expected or \
                    f"{claim['value']} {claim['unit']}" not in doc.content:
                errors.append(q.query_id+': claim not in current authorized source')
        if q.family=='multi_document':
            kinds={metadata[lid]['kind'] for lid in q.logical_document_ids}
            if len(expected)!=2 or kinds!={'BASE','EXC'} or not q.precedence_matters or \
                    {c['precedence'] for c in q.required_atomic_claims}!={1,2}:
                errors.append(q.query_id+': incomplete independent evidence')
        if q.family=='negative' and expected:errors.append(q.query_id+': answerable negative')
        if q.version_sensitive and not any(world.documents[d].version>1 for d in expected):
            errors.append(q.query_id+': wrong version metadata')
    return errors


def cramers_v(records,axis):
    pairs=[(m['tenant_id'],m[axis]) for m in records if m['kind']=='BASE']
    tenants=sorted({t for t,_ in pairs});scopes=sorted({s for _,s in pairs})
    table=Counter(pairs);n=len(pairs)
    rows=Counter(t for t,_ in pairs);cols=Counter(s for _,s in pairs)
    chi2=sum((table[t,s]-rows[t]*cols[s]/n)**2/(rows[t]*cols[s]/n)
             for t in tenants for s in scopes)
    denominator=n*min(len(tenants)-1,len(scopes)-1)
    return {'axis':axis,'cramers_v':math.sqrt(chi2/denominator) if denominator else 0.0,
            'n':n,'contingency':{t:{s:table[t,s] for s in scopes} for t in tenants}}
