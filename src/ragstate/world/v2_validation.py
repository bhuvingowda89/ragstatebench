"""Construction checks for the v2 corpus, independent of retrieval scores."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import replace

from ragstate.retrieval.bm25 import tokenize
from .scientific_v2 import DOMAINS, LOGICAL_TIME, V2Query, _question

IDENTIFIER = re.compile(r'V2-[A-Z]+-\d{2}-[01]-[012]-(?:BASE|EXC)(?:-v\d+)?')


def _longest_phrase(left: str, right: str) -> int:
    a, b = tokenize(left), tokenize(right)
    previous = [0] * (len(b) + 1)
    longest = 0
    for word in a:
        current = [0] * (len(b) + 1)
        for j, other in enumerate(b, 1):
            if word == other:
                current[j] = previous[j - 1] + 1
                longest = max(longest, current[j])
        previous = current
    return longest


def leakage_rows(world, queries: list[V2Query]) -> list[dict]:
    documents = [d for d in world.documents.values()
                 if world.document_at(d.logical_document_id, LOGICAL_TIME) == d]
    by_domain = {domain.name: [d for d in documents if d.logical_document_id.startswith(
        f'V2-{domain.name.upper()}-')] for domain in DOMAINS}
    rows = []
    for q in queries:
        qterms = set(tokenize(q.text))
        gold = [world.documents[doc_id] for doc_id in q.required_document_ids]
        nongold = [d for d in by_domain[q.domain] if d.document_id not in q.required_document_ids]
        doc_freq = Counter(t for d in by_domain[q.domain] for t in set(tokenize(d.content)))
        unique = sorted(t for t in qterms if doc_freq[t] == 1 and
                        any(t in set(tokenize(d.content)) for d in gold))
        identifiers = IDENTIFIER.findall(q.text)
        gold_logical = {d.logical_document_id for d in gold}
        exact_gold_ids = sorted(x for x in identifiers if any(
            x == lid or x == d.document_id for lid in gold_logical for d in gold))
        def coverage(doc):
            return len(qterms & set(tokenize(doc.content))) / len(qterms) if qterms else 0.0
        gold_overlap = sum(coverage(d) for d in gold) / len(gold) if gold else None
        best_nongold = max(nongold, key=lambda d: (coverage(d), d.document_id)) if nongold else None
        nongold_overlap = coverage(best_nongold) if best_nongold else None
        # Title is the first sentence of the document; exact overlap means the full title appears in the question.
        titles = [d.content.split('.', 1)[0] for d in gold]
        title_overlap = any(title.lower() in q.text.lower() for title in titles)
        phrase = max((_longest_phrase(q.text, d.content) for d in gold), default=0)
        warning = (q.family != 'identifier_bearing' and (bool(identifiers) or bool(unique)))
        rows.append({'query_id': q.query_id, 'domain': q.domain, 'family': q.family,
                     'answerable': int(q.answerable), 'required_count': len(gold),
                     'explicit_identifier': int(bool(identifiers)),
                     'gold_identifier_overlap': int(bool(exact_gold_ids)),
                     'unique_gold_tokens': '|'.join(unique),
                     'unique_gold_token_count': len(unique),
                     'unique_gold_token_leak': int(bool(unique)),
                     'exact_title_overlap': int(title_overlap),
                     'longest_shared_phrase_tokens': phrase,
                     'query_gold_overlap': gold_overlap,
                     'query_best_nongold_overlap': nongold_overlap,
                     'gold_lexical_advantage': gold_overlap - nongold_overlap if gold_overlap is not None and nongold_overlap is not None else None,
                     'best_nongold_document_id': best_nongold.document_id if best_nongold else None,
                     'construction_warning': int(warning),
                     'regeneration_attempts': q.regeneration_attempts})
    return rows


def repair_lexical_leakage(world, queries: list[V2Query]) -> tuple[list[V2Query], list[dict]]:
    """Try fixed alternate templates; retain and report any unresolved violation."""
    original = leakage_rows(world, queries)
    revised = []
    for q, row in zip(queries, original):
        if not row['construction_warning']:
            revised.append(q)
            continue
        domain = next(d for d in DOMAINS if d.name == q.domain)
        parts = q.logical_document_ids[0].split('-')
        ti = int(parts[2]); variant = int(parts[3]) * 3 + int(parts[4])
        found = None
        for attempt in range(1, 6):
            text = _question(domain, domain.topics[ti], variant, q.family, parts[5],
                             attempt, q.logical_document_ids[0])
            candidate = replace(q, text=text, regeneration_attempts=q.regeneration_attempts + attempt)
            check = leakage_rows(world, [candidate])[0]
            if not check['construction_warning'] and text not in {x.text for x in revised}:
                found = candidate
                break
        revised.append(found or replace(q, construction_warning='unresolved lexical leakage'))
    return revised, leakage_rows(world, revised)


def validate_gold(world, queries: list[V2Query]) -> list[str]:
    errors = []
    seen_text = set()
    for q in queries:
        if q.text in seen_text:
            errors.append(f'{q.query_id}: duplicate question text')
        seen_text.add(q.text)
        if q.principal_id not in world.principals:
            errors.append(f'{q.query_id}: missing principal')
            continue
        expected = []
        for lid in q.logical_document_ids:
            current = world.document_at(lid, q.logical_time)
            if current is None:
                errors.append(f'{q.query_id}: missing logical document {lid}')
                continue
            if world.allowed(q.principal_id, lid, q.logical_time):
                expected.append(current.document_id)
        if tuple(expected) != q.required_document_ids:
            errors.append(f'{q.query_id}: incorrect current authorized gold')
        if q.answerable != bool(expected):
            errors.append(f'{q.query_id}: incorrect answerability')
        if q.answerable and len(q.required_atomic_claims) != len(expected):
            errors.append(f'{q.query_id}: missing atomic claims')
        for claim in q.required_atomic_claims:
            doc = world.documents.get(claim['document_id'])
            if doc is None or claim['document_id'] not in expected or \
                    f"{claim['value']} {claim['unit']}" not in doc.content:
                errors.append(f'{q.query_id}: atomic claim does not match current evidence')
        if q.family == 'multi_document':
            if len(expected) != 2 or not q.precedence_matters:
                errors.append(f'{q.query_id}: incomplete multi-document gold')
            elif {c['claim_type'] for c in q.required_atomic_claims} != {'default_rule','approved_exception'}:
                errors.append(f'{q.query_id}: incorrect multi-document roles')
        if q.version_sensitive and not any(world.documents[d].version > 1 for d in expected):
            errors.append(f'{q.query_id}: version-sensitive metadata mismatch')
        if q.family == 'negative' and expected:
            errors.append(f'{q.query_id}: negative is answerable')
    return errors
