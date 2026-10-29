"""Versioned, deterministic Phase 7B enterprise corpus; v1 remains untouched."""
from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass

from .ledger import WorldLedger
from .models import DocumentVersion, Principal


@dataclass(frozen=True)
class Topic:
    subject: str
    measure: str
    unit: str
    base: int
    context: str
    paraphrase: str


@dataclass(frozen=True)
class Domain:
    name: str
    axis_one: tuple[str, str]
    axis_two: tuple[str, str, str]
    topics: tuple[Topic, ...]
    policy_context: str


DOMAINS = (
    Domain('hr', ('domestic', 'international'), ('employees', 'contractors', 'managers'), (
        Topic('travel meals', 'reimbursement cap', 'dollars', 42, 'expense claims', 'per-meal travel expense limit'),
        Topic('family leave', 'notice period', 'days', 18, 'leave requests', 'advance notice for family absence'),
        Topic('professional training', 'annual allowance', 'dollars', 750, 'learning plans', 'yearly training budget'),
        Topic('remote equipment', 'purchase ceiling', 'dollars', 310, 'home office requests', 'home-office spending limit'),
        Topic('health enrollment', 'waiting period', 'days', 24, 'benefit enrollment', 'time before health coverage begins')),
        'People Operations records approved employment conditions.'),
    Domain('finance', ('domestic', 'international'), ('routine', 'strategic', 'regulated'), (
        Topic('purchase orders', 'approval threshold', 'dollars', 1400, 'procurement review', 'spending level requiring approval'),
        Topic('expense claims', 'filing deadline', 'days', 21, 'reimbursement review', 'time allowed to submit expenses'),
        Topic('vendor invoices', 'review period', 'days', 12, 'accounts payable', 'time available for invoice review'),
        Topic('treasury transfers', 'signoff threshold', 'dollars', 4800, 'cash controls', 'transfer amount requiring signoff'),
        Topic('audit evidence', 'retention period', 'months', 30, 'financial records', 'duration for keeping audit material')),
        'Controller approval governs payment and accounting records.'),
    Domain('engineering', ('production', 'staging'), ('core', 'edge', 'internal'), (
        Topic('release rollback', 'decision window', 'minutes', 35, 'deployment recovery', 'time to decide on rollback'),
        Topic('release freeze', 'lead time', 'hours', 16, 'change calendar', 'notice before a release freeze'),
        Topic('incident escalation', 'escalation interval', 'minutes', 20, 'on-call response', 'time before escalation'),
        Topic('service restoration', 'recovery target', 'hours', 4, 'service incidents', 'target time for restoration'),
        Topic('deployment approval', 'reviewer count', 'reviewers', 2, 'change review', 'number of required reviewers')),
        'Service owners review the operational risk of changes.'),
    Domain('security', ('production', 'corporate'), ('human', 'service', 'device'), (
        Topic('credential rotation', 'rotation interval', 'days', 46, 'identity controls', 'frequency for replacing credentials'),
        Topic('signing keys', 'approver count', 'approvers', 2, 'key custody', 'number of people approving key rotation'),
        Topic('device compliance', 'remediation grace', 'days', 7, 'endpoint checks', 'time to remediate a device'),
        Topic('privileged access', 'review interval', 'days', 28, 'access governance', 'frequency of privileged review'),
        Topic('incident notification', 'reporting deadline', 'hours', 5, 'security response', 'time allowed to report an incident')),
        'Security Engineering records control owners and evidence.'),
    Domain('sales', ('domestic', 'international'), ('standard', 'enterprise', 'strategic'), (
        Topic('renewal discounts', 'discount cap', 'percent', 8, 'renewal quotes', 'largest permitted renewal discount'),
        Topic('customer escalation', 'response target', 'hours', 6, 'account support', 'time allowed for escalation response'),
        Topic('customer credits', 'approval ceiling', 'dollars', 400, 'commercial adjustments', 'credit amount needing approval'),
        Topic('quote validity', 'validity period', 'days', 20, 'sales proposals', 'length of time a quote remains valid'),
        Topic('account handoff', 'handoff window', 'days', 5, 'customer success transfer', 'time allowed to transfer an account')),
        'Revenue Operations coordinates commercial approvals.'),
    Domain('legal', ('domestic', 'international'), ('vendor', 'customer', 'partner'), (
        Topic('record retention', 'retention period', 'months', 30, 'records governance', 'length of time records must be kept'),
        Topic('contract review', 'review lead time', 'days', 14, 'agreement intake', 'advance time needed for legal review'),
        Topic('regulatory notices', 'filing deadline', 'days', 11, 'compliance filings', 'time allowed to file a notice'),
        Topic('litigation holds', 'review interval', 'days', 28, 'preservation orders', 'frequency of hold review'),
        Topic('data addenda', 'negotiation period', 'days', 16, 'privacy terms', 'time allowed for addendum negotiation')),
        'Counsel maintains jurisdiction and agreement records.'),
)

FAMILY_COUNTS = {'natural_descriptive': 13, 'paraphrased_policy': 7,
                 'entity_context': 5, 'identifier_bearing': 4,
                 'multi_document': 5, 'negative': 2}
TRANSITIONS = ('UPDATE', 'SUPERSEDE', 'DELETE', 'GRANT', 'REVOKE')
LOGICAL_TIME = 9


@dataclass(frozen=True)
class V2Query:
    query_id: str
    text: str
    domain: str
    family: str
    principal_id: str
    tenant_id: str
    logical_time: int
    logical_document_ids: tuple[str, ...]
    required_document_ids: tuple[str, ...]
    required_atomic_claims: tuple[dict, ...]
    answerable: bool
    precedence_matters: bool
    authorization_sensitive: bool
    version_sensitive: bool
    transition_type: str
    target_document_id: str
    regeneration_attempts: int
    construction_warning: str | None


def logical_id(domain: Domain, topic_index: int, variant: int, kind: str) -> str:
    return f'V2-{domain.name.upper()}-{topic_index:02d}-{variant // 3}-{variant % 3}-{kind}'


def _content(domain: Domain, topic: Topic, topic_index: int, variant: int,
             kind: str, value: int, version: int) -> str:
    axis1, axis2 = domain.axis_one[variant // 3], domain.axis_two[variant % 3]
    role = 'Approved exception' if kind == 'EXC' else 'Current policy'
    title = f'{role}: {topic.subject} for {axis1} {axis2}.'
    metadata = f'Owner: {domain.name.title()} office. Effective cycle: {2024 + version}. Status: approved.'
    if kind == 'BASE':
        fact = (f'For {axis1} {axis2}, the {topic.measure} for {topic.subject} '
                f'is {value} {topic.unit}. This is the default rule in {topic.context}.')
        caveat = (f'An approved exception can alter this amount for qualifying cases. '
                  f'Requests outside this scope follow the relevant local schedule.')
    else:
        fact = (f'When a documented exception is approved for {axis1} {axis2}, '
                f'the {topic.measure} for {topic.subject} becomes {value} {topic.unit}.')
        caveat = (f'This override applies only after the responsible owner records approval. '
                  f'Without approval, the default policy continues to govern.')
    related = ('Related control: consult the paired policy before applying an override. '
               if kind == 'EXC' else 'Related control: a separate exception record may supersede this default. ')
    closing = domain.policy_context if (topic_index + variant) % 2 == 0 else f'{topic.context.title()} staff retain evidence of the decision.'
    return '\n\n'.join((title + ' ' + metadata, fact, caveat + ' ' + related + closing))


def _value(topic: Topic, topic_index: int, variant: int, kind: str, rng: random.Random) -> int:
    # Nearby scope variants intentionally have nearby, distinguishable values.
    base = topic.base + (variant // 3) * 3 + (variant % 3) * 2 + topic_index
    jitter = rng.randrange(2)
    return base + (5 if kind == 'EXC' else 0) + jitter


def generate_v2_world(world_seed: int) -> WorldLedger:
    rng = random.Random(world_seed)
    world = WorldLedger()
    for tenant in ('tenant_a', 'tenant_b'):
        for domain in DOMAINS:
            world.add_principal(Principal(f'v2-{tenant}-{domain.name}-specialist', tenant,
                                          frozenset({'employee', domain.name})))
            world.add_principal(Principal(f'v2-{tenant}-{domain.name}-generalist', tenant,
                                          frozenset({'employee'})))
    for domain in DOMAINS:
        for ti, topic in enumerate(domain.topics):
            for variant in range(6):
                tenant = ('tenant_a', 'tenant_b')[variant // 3]
                for kind in ('BASE', 'EXC'):
                    logical = logical_id(domain, ti, variant, kind)
                    current_value = _value(topic, ti, variant, kind, rng)
                    groups = frozenset({domain.name}) if kind == 'EXC' else frozenset({'employee'})
                    versioned = kind == 'BASE' and variant % 3 == 0
                    if versioned:
                        previous = DocumentVersion(f'{logical}-v1', logical, tenant, 1, 0,
                                                   valid_until=5, authority_level=3,
                                                   content=_content(domain, topic, ti, variant,
                                                                    kind, current_value - 3, 1),
                                                   required_groups=groups)
                        world.add_document(previous)
                    version = 2 if versioned else 1
                    world.add_document(DocumentVersion(
                        f'{logical}-v{version}', logical, tenant, version,
                        5 if versioned else 0,
                        supersedes=f'{logical}-v1' if versioned else None,
                        authority_level=4 if kind == 'EXC' else 3,
                        content=_content(domain, topic, ti, variant, kind, current_value, version),
                        required_groups=groups))
    return world


def _question(domain: Domain, topic: Topic, variant: int, family: str,
              kind: str, template_index: int, logical: str) -> str:
    a, b = domain.axis_one[variant // 3], domain.axis_two[variant % 3]
    scope = f'{a} {b}'
    if family == 'identifier_bearing':
        return f'For record {logical}, what is the currently approved {topic.measure}?'
    if family == 'multi_document':
        options = (
            f'For {scope}, what is the default {topic.measure} for {topic.subject}, and what changes after a documented exception is approved?',
            f'How does an approved exception alter the present {topic.measure} for {topic.subject} affecting {scope}? Include the default rule.',
        )
    elif family == 'paraphrased_policy':
        options = (
            f'For {scope}, what is the present {topic.paraphrase}?',
            f'How is the {topic.paraphrase} currently set for {scope}?',
        )
    elif family == 'entity_context':
        options = (
            f'Within {domain.name.title()} operations for {scope}, how does the current {topic.subject} rule set the {topic.measure}?',
            f'In the {domain.name.title()} {scope} context, what is presently required for {topic.subject}?',
        )
    elif family == 'negative':
        options = (
            f'What is the current {topic.measure} for {topic.subject} affecting {scope}?',
            f'For {scope}, what does the approved {topic.subject} schedule require now?',
        )
    else:
        options = (
            f'What is the current default {topic.measure} for {topic.subject} affecting {scope}?',
            f'How much is the present {topic.subject} {topic.measure} for {scope}?',
            f'For {scope}, what does the latest approved {topic.subject} rule specify?',
        )
    return options[template_index % len(options)]


def generate_v2_queries(world: WorldLedger, query_seed: int) -> list[V2Query]:
    rng = random.Random(query_seed)
    queries: list[V2Query] = []
    seen = set()
    for domain in DOMAINS:
        allocation = [family for family, count in FAMILY_COUNTS.items() for _ in range(count)]
        rng.shuffle(allocation)
        for j, family in enumerate(allocation):
            ti, variant = (j // 6) % 5, j % 6
            topic = domain.topics[ti]
            kind = 'EXC' if family == 'negative' else 'BASE'
            base_logical = logical_id(domain, ti, variant, 'BASE')
            exc_logical = logical_id(domain, ti, variant, 'EXC')
            ids = (base_logical, exc_logical) if family == 'multi_document' else (
                (exc_logical,) if family == 'negative' else (base_logical,))
            tenant = ('tenant_a', 'tenant_b')[variant // 3]
            principal_tenant = ('tenant_b' if tenant == 'tenant_a' else 'tenant_a') if family == 'negative' else tenant
            principal = f'v2-{principal_tenant}-{domain.name}-specialist'
            if family == 'natural_descriptive' and j % 3 == 0:
                principal = f'v2-{principal_tenant}-{domain.name}-generalist'
            text = None
            attempts = 0
            for attempt in range(3):
                candidate = _question(domain, topic, variant, family, kind, attempt + j, base_logical)
                if candidate not in seen:
                    text = candidate
                    attempts = attempt
                    break
            if text is None:
                raise ValueError('Duplicate deterministic query could not be resolved')
            seen.add(text)
            required = tuple(world.document_at(lid, LOGICAL_TIME).document_id for lid in ids
                             if world.allowed(principal, lid, LOGICAL_TIME))
            claims = []
            for lid in ids:
                doc = world.document_at(lid, LOGICAL_TIME)
                if doc and world.allowed(principal, lid, LOGICAL_TIME):
                    match = re.search(r'\b(?:is|becomes) (\d+) ' + re.escape(topic.unit) + r'\b', doc.content)
                    if match is None:
                        raise ValueError(f'Cannot resolve atomic value in {doc.document_id}')
                    claims.append({'logical_document_id': lid, 'document_id': doc.document_id,
                                   'claim_type': 'approved_exception' if lid.endswith('EXC') else 'default_rule',
                                   'measure': topic.measure, 'value': int(match.group(1)),
                                   'unit': topic.unit,
                                   'scope': f'{domain.axis_one[variant // 3]} {domain.axis_two[variant % 3]}',
                                   'precedence': 2 if lid.endswith('EXC') else 1})
            target = world.document_at(ids[0], LOGICAL_TIME)
            queries.append(V2Query(f'v2-q-{domain.name}-{j:02d}', text, domain.name, family,
                                   principal, principal_tenant, LOGICAL_TIME, ids, required,
                                   tuple(claims), bool(required), family == 'multi_document',
                                   family in ('multi_document', 'negative') or j % 4 == 1,
                                   any(world.document_at(lid, LOGICAL_TIME).version > 1 for lid in ids),
                                   TRANSITIONS[j % len(TRANSITIONS)], target.document_id,
                                   attempts, None))
    return queries


def corpus_hash(world: WorldLedger, queries: list[V2Query]) -> str:
    payload = {
        'documents': [asdict(world.documents[k]) for k in sorted(world.documents)],
        'principals': [asdict(world.principals[k]) for k in sorted(world.principals)],
        'queries': [asdict(q) for q in queries],
    }
    def normalize(value):
        if isinstance(value, (set, frozenset)):
            return sorted(value)
        raise TypeError(type(value).__name__)
    return hashlib.sha256(json.dumps(payload, default=normalize, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()
