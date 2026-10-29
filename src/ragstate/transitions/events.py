from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class Transition:
    event_id: str
    logical_time: int
    event_type: str
    resource_id: str
    payload: dict[str, Any]
