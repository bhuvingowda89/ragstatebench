"""Narrow, audited capabilities for the two-track transition runtime.

Content lifecycle and authorization are separate planes. Runtime retrieval,
cache, and indexed filtering receive replica views only. The scheduler may
construct canonical transitions; the evaluator may compare against canonical
truth. A4/A5 receive only this module's authorization-only view.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

from ragstate.world.models import PermissionRule


class ForbiddenCanonicalAccess(RuntimeError):
    pass


class AccessAudit:
    """Fail immediately when a component crosses a canonical plane boundary."""

    _allowed = {
        'scheduler': frozenset({'canonical_content', 'canonical_authorization'}),
        'oracle': frozenset({'canonical_content', 'canonical_authorization'}),
        'evaluator': frozenset({'canonical_content', 'canonical_authorization'}),
        'live_authorization': frozenset({'canonical_authorization'}),
    }

    def __init__(self, query_id: str):
        self.query_id = query_id
        self.events: list[dict] = []
        self.forbidden_count = 0

    def record(self, component: str, plane: str, logical_time: int,
               reason: str) -> None:
        allowed = plane in self._allowed.get(component, frozenset())
        event = {'component': component, 'logical_time': logical_time,
                 'query_id': self.query_id, 'accessed_plane': plane,
                 'access_reason': reason, 'allowed': allowed}
        self.events.append(event)
        if not allowed:
            self.forbidden_count += 1
            raise ForbiddenCanonicalAccess(f'{component} may not read {plane}: {reason}')

    def assert_clean(self) -> None:
        if self.forbidden_count:
            raise ForbiddenCanonicalAccess(f'{self.forbidden_count} forbidden canonical accesses')


@dataclass(frozen=True)
class AuthorizationMetadata:
    tenant_id: str
    required_groups: frozenset[str]


class AuthorizationPolicyView:
    """ACL/tenant/group state with no document content or lifecycle API."""

    def __init__(self, principals, metadata, permissions, group_events):
        self.principals = {key: (value.tenant_id, frozenset(value.groups))
                           for key, value in principals.items()}
        self.metadata = dict(metadata)
        self.permissions = list(permissions)
        self.group_events = list(group_events)

    @classmethod
    def from_world(cls, world, tick: int):
        metadata = {}
        for doc in world.documents.values():
            lid = doc.logical_document_id
            if lid not in metadata:
                current = world.document_at(lid, tick)
                if current is not None:
                    metadata[lid] = AuthorizationMetadata(current.tenant_id,
                                                          frozenset(current.required_groups))
        return cls(world.principals, metadata,
                   copy.deepcopy(world.permissions), copy.deepcopy(world.group_events))

    def allowed(self, principal_id: str, logical_document_id: str, tick: int) -> bool:
        principal = self.principals[principal_id]
        resource = self.metadata.get(logical_document_id)
        if resource is None or principal[0] != resource.tenant_id:
            return False
        direct = [rule for rule in self.permissions
                  if rule.principal_id == principal_id
                  and rule.logical_document_id == logical_document_id
                  and rule.active(tick)]
        if direct:
            return direct[-1].allow
        if not resource.required_groups:
            return True
        groups = set(principal[1])
        for at, who, group, add in self.group_events:
            if who == principal_id and at <= tick:
                if add:
                    groups.add(group)
                else:
                    groups.discard(group)
        return bool(groups.intersection(resource.required_groups))

    def apply_authorization_event(self, event) -> bool:
        if event.event_type in ('PERMISSION_GRANT', 'PERMISSION_REVOKE'):
            self.permissions.append(PermissionRule(
                event.payload['principal_id'], event.payload['logical_document_id'],
                event.event_type == 'PERMISSION_GRANT', event.logical_time))
            return True
        if event.event_type in ('GROUP_ADD', 'GROUP_REMOVE'):
            self.group_events.append((event.logical_time, event.payload['principal_id'],
                                      event.payload['group_id'],
                                      event.event_type == 'GROUP_ADD'))
            return True
        return False

    def apply_replica_metadata_event(self, event) -> bool:
        if event.event_type in ('DOCUMENT_CREATE', 'DOCUMENT_UPDATE', 'DOCUMENT_SUPERSEDE'):
            payload = event.payload
            self.metadata[payload['logical_document_id']] = AuthorizationMetadata(
                payload['tenant_id'], frozenset(payload.get('required_groups', ())))
            return True
        # Deletion changes source existence, not the authorization policy.
        return False

    def snapshot(self):
        return {'principals': sorted((key, tenant, sorted(groups))
                                     for key, (tenant, groups) in self.principals.items()),
                'metadata': sorted((key, value.tenant_id, sorted(value.required_groups))
                                   for key, value in self.metadata.items()),
                'permissions': [asdict(rule) for rule in self.permissions],
                'group_events': self.group_events}


class AuthoritativeAuthorizationView:
    """The sole A4/A5 authoritative capability; no content methods exist."""

    def __init__(self, policy: AuthorizationPolicyView, audit: AccessAudit):
        self._policy = policy
        self._audit = audit

    def apply(self, event) -> None:
        if event.event_type in ('PERMISSION_GRANT', 'PERMISSION_REVOKE',
                                'GROUP_ADD', 'GROUP_REMOVE'):
            self._audit.record('scheduler', 'canonical_authorization', event.logical_time,
                               'deliver authoritative authorization event')
            self._policy.apply_authorization_event(event)
        elif event.event_type.startswith('DOCUMENT_'):
            # No content lifecycle event may change this capability.
            return
        else:
            raise ValueError('Unknown event for authorization view')

    def allowed(self, principal_id: str, logical_document_id: str,
                tick: int) -> bool:
        self._audit.record('live_authorization', 'canonical_authorization', tick,
                           'validate current ACL before context')
        return self._policy.allowed(principal_id, logical_document_id, tick)

    def state_hash_payload(self):
        return self._policy.snapshot()


class CanonicalEvaluationView:
    """Evaluator-only wrapper; deliberately not passed to retrieval or cache."""

    def __init__(self, ledger, audit: AccessAudit, tick: int):
        self._ledger = ledger
        self._audit = audit
        self._tick = tick

    def document_at(self, logical_document_id, tick):
        self._audit.record('evaluator', 'canonical_content', tick,
                           'resolve current evidence')
        return self._ledger.document_at(logical_document_id, tick)

    def allowed(self, principal_id, logical_document_id, tick):
        self._audit.record('evaluator', 'canonical_authorization', tick,
                           'evaluate canonical access policy')
        self._audit.record('evaluator', 'canonical_content', tick,
                           'world truth predicate also checks resource existence')
        return self._ledger.allowed(principal_id, logical_document_id, tick)

    @property
    def documents(self):
        self._audit.record('evaluator', 'canonical_content', self._tick,
                           'inspect canonical document records')
        return self._ledger.documents

    @property
    def principals(self):
        self._audit.record('evaluator', 'canonical_authorization', self._tick,
                           'inspect principal tenant/groups')
        return self._ledger.principals

    @property
    def permissions(self):
        self._audit.record('evaluator', 'canonical_authorization', self._tick,
                           'inspect direct permission rules')
        return self._ledger.permissions

    @property
    def group_events(self):
        self._audit.record('evaluator', 'canonical_authorization', self._tick,
                           'inspect group membership events')
        return self._ledger.group_events
