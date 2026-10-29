from __future__ import annotations
import hashlib, json

class Replica:
    def __init__(self, name: str):
        self.name = name
        self.applied_event_ids: list[str] = []
        self.state: dict = {}

    def apply(self, event) -> None:
        self.applied_event_ids.append(event.event_id)
        self.state[event.resource_id] = {
            "event_type": event.event_type,
            "payload": event.payload,
            "logical_time": event.logical_time,
        }

    def state_hash(self) -> str:
        blob = json.dumps(self.state, sort_keys=True, default=list).encode()
        return hashlib.sha256(blob).hexdigest()

    def version(self) -> int:
        return len(self.applied_event_ids)
