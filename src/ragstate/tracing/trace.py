from __future__ import annotations
from dataclasses import dataclass, asdict
import json

@dataclass
class QueryTrace:
    query_id: str
    principal_id: str
    logical_time: int
    canonical_state_hash: str
    index_state_hash: str
    auth_state_hash: str
    cache_state_hash: str
    retrieved: list[str]
    filtered: list[str]
    context: list[str]
    output_claims: list[str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)
