"""Explicit replica planes for the state-isolated and PA2 pilot tracks."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass

from ragstate.retrieval.chunking import chunk_document
from ragstate.retrieval.models import Chunk
from ragstate.world.models import DocumentVersion
from ragstate.campaign.state_capabilities import AuthorizationPolicyView


def _stable(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     default=lambda item: sorted(item) if isinstance(item, frozenset)
                                     else str(item)).encode()).hexdigest()


def _document(payload: dict) -> DocumentVersion:
    values = dict(payload)
    values['required_groups'] = frozenset(values.get('required_groups', ()))
    return DocumentVersion(**values)


class ReplicaContentPlane:
    """A materialized content index or parent-document index, never an oracle."""

    def __init__(self, source_ledger, tick: int, *, chunked: bool):
        self.chunked = chunked
        self.documents: dict[str, DocumentVersion] = {}
        self.chunks: dict[str, list[Chunk]] = {}
        self.epoch = 0
        self._state_hash: str | None = None
        seen = set()
        for record in source_ledger.documents.values():
            lid = record.logical_document_id
            if lid in seen:
                continue
            seen.add(lid)
            current = source_ledger.document_at(lid, tick)
            if current is not None:
                self.documents[lid] = copy.deepcopy(current)
                if chunked:
                    self.chunks[lid] = chunk_document(current, max_words=160)

    def ordered_chunks(self) -> list[Chunk]:
        if not self.chunked:
            raise ValueError('Parent plane does not expose chunk-index records')
        return [chunk for lid in self.documents for chunk in self.chunks.get(lid, ())]

    def ordered_parent_chunks(self) -> list[Chunk]:
        if self.chunked:
            raise ValueError('Chunk plane does not expose parent-index records')
        return [Chunk(doc.document_id + ':document', doc.document_id, lid, doc.version,
                      doc.tenant_id, 0, ' '.join(doc.content.split()))
                for lid, doc in self.documents.items()]

    def apply(self, event, *, fraction: float = 1.0) -> dict:
        """Apply source event payload to replica units; partial applies real chunks."""
        lid = event.payload.get('logical_document_id')
        if event.event_type == 'DOCUMENT_DELETE':
            if lid is None:
                raise ValueError('Delete event needs logical_document_id in pilot payload')
            if self.chunked and fraction < 1 and lid in self.chunks:
                old_chunks = self.chunks[lid]
                count = min(len(old_chunks), max(1, math.floor(len(old_chunks) * fraction)))
                self.chunks[lid] = old_chunks[count:]
                self.epoch += 1
                self._state_hash = None
                return {'applied_units': count, 'total_units': len(old_chunks)}
            self.documents.pop(lid, None)
            self.chunks.pop(lid, None)
            self.epoch += 1
            self._state_hash = None
            return {'applied_units': 1, 'total_units': 1}
        if event.event_type not in ('DOCUMENT_UPDATE', 'DOCUMENT_SUPERSEDE', 'DOCUMENT_CREATE'):
            return {'applied_units': 0, 'total_units': 0}
        new = _document(event.payload)
        lid = new.logical_document_id
        if not 0 < fraction <= 1:
            raise ValueError('Partial fraction must be within (0, 1]')
        if self.chunked:
            new_chunks = chunk_document(new, max_words=160)
            old_chunks = self.chunks.get(lid, [])
            if fraction < 1 and old_chunks:
                count = min(len(new_chunks), max(1, math.floor(len(new_chunks) * fraction)))
                merged = [new_chunks[i] if i < count else old_chunks[i]
                          for i in range(min(len(new_chunks), len(old_chunks)))]
                merged.extend(new_chunks[len(merged):])
                self.chunks[lid] = merged
                # Document metadata remains locally old until all chunks arrive.
                self.epoch += 1
                self._state_hash = None
                return {'applied_units': count, 'total_units': len(new_chunks)}
            self.chunks[lid] = new_chunks
        self.documents[lid] = new
        self.epoch += 1
        self._state_hash = None
        return {'applied_units': len(self.chunks[lid]) if self.chunked else 1,
                'total_units': len(self.chunks[lid]) if self.chunked else 1}

    def state_hash(self) -> str:
        if self._state_hash is None:
            self._state_hash = _stable({'documents': [(lid, asdict(doc)) for lid, doc in self.documents.items()],
                        'chunks': [(lid, [(chunk.chunk_id, chunk.text) for chunk in values])
                                   for lid, values in self.chunks.items()],
                        'epoch': self.epoch})
        return self._state_hash


class ReplicaAuthorizationPlane:
    """Locally indexed ACL/metadata; document existence is a separate plane."""

    def __init__(self, source_ledger, tick: int = 9):
        self.policy = AuthorizationPolicyView.from_world(source_ledger, tick)
        self.chunk_overrides: dict[tuple[str, str], bool] = {}
        self.epoch = 0
        self._state_hash: str | None = None

    def allowed(self, principal_id: str, chunk: Chunk, tick: int) -> bool:
        override = self.chunk_overrides.get((principal_id, chunk.chunk_id))
        return override if override is not None else self.policy.allowed(
            principal_id, chunk.logical_document_id, tick)

    def apply(self, event, *, fraction: float = 1.0, units: list[Chunk] | None = None) -> dict:
        principal = event.payload.get('principal_id')
        if event.event_type in ('PERMISSION_GRANT', 'PERMISSION_REVOKE', 'GROUP_ADD', 'GROUP_REMOVE'):
            if not 0 < fraction <= 1:
                raise ValueError('ACL fraction must be within (0, 1]')
            units = sorted(units or [], key=lambda chunk: chunk.chunk_id)
            old = {(principal, chunk.chunk_id): self.allowed(principal, chunk, event.logical_time)
                   for chunk in units}
            self.policy.apply_authorization_event(event)
            count = len(units) if fraction == 1 else max(1, math.floor(len(units) * fraction))
            selected = {chunk.chunk_id for chunk in units[:count]}
            for chunk in units:
                key = (principal, chunk.chunk_id)
                if chunk.chunk_id in selected:
                    self.chunk_overrides.pop(key, None)
                else:
                    self.chunk_overrides[key] = old[key]
            self.epoch += 1
            self._state_hash = None
            return {'applied_units': count, 'total_units': len(units)}
        if event.event_type.startswith('DOCUMENT_'):
            # An update can deliver new ACL metadata; DELETE alone cannot turn
            # a prior authorization into denial.
            changed = self.policy.apply_replica_metadata_event(event)
            if changed:
                self.epoch += 1
                self._state_hash = None
            return {'applied_units': int(changed), 'total_units': int(changed)}
        return {'applied_units': 0, 'total_units': 0}

    def complete_partial(self, principal_id: str, chunk_ids: list[str]) -> None:
        for chunk_id in chunk_ids:
            self.chunk_overrides.pop((principal_id, chunk_id), None)
        self.epoch += 1
        self._state_hash = None

    def state_hash(self, tick: int) -> str:
        if self._state_hash is None:
            self._state_hash = _stable({
                'authorization': self.policy.snapshot(),
                'partial_acl': sorted((principal, chunk_id, allowed)
                                      for (principal, chunk_id), allowed in self.chunk_overrides.items()),
                'epoch': self.epoch})
        return self._state_hash


@dataclass(frozen=True)
class CandidateCacheEntry:
    cache_type: str
    pipeline_stage: str
    evidence: tuple[tuple[Chunk, float, tuple], ...]
    principal_id: str
    tenant_id: str
    creation_time: int
    chunk_epoch: int
    parent_epoch: int
    auth_epoch: int
    producer_architecture: str
    source_state_hashes: tuple[tuple[str, str], ...]

    @property
    def evidence_ids(self):
        return tuple(item[0].chunk_id for item in self.evidence)

    @property
    def versions(self):
        return tuple(item[0].version for item in self.evidence)


class ReplicaEvidenceCache:
    """C1 candidate cache. A hit re-enters indexed/live filtering every time."""

    def __init__(self):
        self.entries: dict[str, CandidateCacheEntry] = {}
        self.invalidations = 0
        self.stage_audit: list[dict] = []

    def put(self, key: str, evidence, principal_id: str,
            chunk_epoch: int, parent_epoch: int, auth_epoch: int, *,
            tenant_id: str, creation_time: int, producer_architecture: str,
            source_state_hashes: dict[str, str]) -> None:
        from ragstate.retrieval.models import RetrievalHit
        if not all(isinstance(item, RetrievalHit) for item in evidence):
            raise TypeError('Candidate cache accepts retrieval hits, not context chunks')
        ordered = tuple((hit.chunk, hit.score, hit.components) for hit in evidence)
        self.entries[key] = CandidateCacheEntry(
            'C1_candidate', 'candidate_retrieval', ordered, principal_id,
            tenant_id, creation_time, chunk_epoch, parent_epoch, auth_epoch,
            producer_architecture, tuple(sorted(source_state_hashes.items())))
        self.stage_audit.append({'action': 'put', 'key': key, 'cache_type': 'C1_candidate',
                                 'created_stage': 'candidate_retrieval',
                                 'consumed_stage': None, 'valid': True,
                                 'logical_time': creation_time})

    def get(self, key: str, principal_id: str, *, tenant_id: str,
            logical_time: int, consumer_stage: str = 'candidate_retrieval',
            epoch_aware: bool, chunk_epoch: int, parent_epoch: int,
            auth_epoch: int):
        from ragstate.retrieval.models import RetrievalHit
        entry = self.entries.get(key)
        if entry is None:
            return None
        valid_stage = entry.pipeline_stage == consumer_stage == 'candidate_retrieval'
        self.stage_audit.append({'action': 'get', 'key': key,
                                 'cache_type': entry.cache_type,
                                 'created_stage': entry.pipeline_stage,
                                 'consumed_stage': consumer_stage,
                                 'valid': valid_stage,
                                 'logical_time': logical_time})
        if not valid_stage:
            raise ValueError('Candidate cache cannot be consumed as post-filter context')
        if entry.principal_id != principal_id or entry.tenant_id != tenant_id:
            return None
        if epoch_aware and (entry.chunk_epoch, entry.parent_epoch, entry.auth_epoch) != \
                (chunk_epoch, parent_epoch, auth_epoch):
            return None
        return [RetrievalHit(chunk, score, components)
                for chunk, score, components in entry.evidence]

    def invalidate(self, key: str | None = None) -> None:
        if key is None:
            self.invalidations += len(self.entries)
            self.entries.clear()
        elif key in self.entries:
            self.invalidations += 1
            del self.entries[key]

    def state_hash(self) -> str:
        return _stable({'entries': [(key, entry.cache_type, entry.pipeline_stage,
                                     [(chunk.chunk_id, chunk.version, score, components)
                                      for chunk, score, components in entry.evidence],
                                     entry.principal_id, entry.tenant_id,
                                     entry.creation_time, entry.chunk_epoch,
                                     entry.parent_epoch, entry.auth_epoch,
                                     entry.producer_architecture,
                                     entry.source_state_hashes)
                                    for key, entry in sorted(self.entries.items())],
                        'invalidations': self.invalidations})
