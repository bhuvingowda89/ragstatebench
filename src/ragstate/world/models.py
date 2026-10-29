from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, FrozenSet

@dataclass(frozen=True)
class DocumentVersion:
    document_id: str
    logical_document_id: str
    tenant_id: str
    version: int
    valid_from: int
    valid_until: Optional[int] = None
    supersedes: Optional[str] = None
    authority_level: int = 1
    deleted_at: Optional[int] = None
    content: str = ""
    required_groups: FrozenSet[str] = field(default_factory=frozenset)

    def is_temporally_valid(self, t: int) -> bool:
        if t < self.valid_from:
            return False
        if self.valid_until is not None and t >= self.valid_until:
            return False
        if self.deleted_at is not None and t >= self.deleted_at:
            return False
        return True

@dataclass(frozen=True)
class Principal:
    principal_id: str
    tenant_id: str
    groups: FrozenSet[str] = field(default_factory=frozenset)

@dataclass(frozen=True)
class PermissionRule:
    principal_id: str
    logical_document_id: str
    allow: bool
    valid_from: int
    valid_until: Optional[int] = None

    def active(self, t: int) -> bool:
        return self.valid_from <= t and (self.valid_until is None or t < self.valid_until)
