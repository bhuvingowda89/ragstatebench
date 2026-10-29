from __future__ import annotations
from dataclasses import dataclass
from .base import Replica

@dataclass
class CacheEntry:
    key: str
    value: str
    principal_id: str
    source_versions: dict[str, int]
    permission_epoch: int
    index_epoch: int

class DependencyAwareCache(Replica):
    def __init__(self, name: str):
        super().__init__(name)
        self.entries: dict[str, CacheEntry] = {}
        self.permission_epochs: dict[str, int] = {}
        self.index_epoch: int = 0

    def put(self, entry: CacheEntry) -> None:
        self.entries[entry.key] = entry

    def bump_permission_epoch(self, principal_id: str) -> None:
        self.permission_epochs[principal_id] = self.permission_epochs.get(principal_id, 0) + 1

    def bump_index_epoch(self) -> None:
        self.index_epoch += 1

    def reusable(self, key: str, principal_id: str) -> bool:
        entry = self.entries.get(key)
        if entry is None or entry.principal_id != principal_id:
            return False
        return (
            entry.permission_epoch == self.permission_epochs.get(principal_id, 0)
            and entry.index_epoch == self.index_epoch
        )
