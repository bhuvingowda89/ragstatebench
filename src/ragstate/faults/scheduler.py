from __future__ import annotations
from dataclasses import dataclass, field, replace

@dataclass
class FaultSchedule:
    delays: dict[tuple[str, str], int] = field(default_factory=dict)
    drops: set[tuple[str, str]] = field(default_factory=set)
    payload_overrides: dict[tuple[str, str], dict] = field(default_factory=dict)

    def delivery_time(self, replica_name: str, event) -> int | None:
        key = (replica_name, event.event_id)
        if key in self.drops:
            return None
        return event.logical_time + self.delays.get(key, 0)

    def transform(self, replica_name: str, event):
        key = (replica_name, event.event_id)
        override = self.payload_overrides.get(key)
        if override is None:
            return event
        payload = dict(event.payload)
        payload.update(override)
        return replace(event, payload=payload)

    def schedule_delay(self, replica_name: str, event_id: str, ticks: int) -> None:
        self.delays[(replica_name, event_id)] = ticks

    def schedule_drop(self, replica_name: str, event_id: str) -> None:
        self.drops.add((replica_name, event_id))

    def schedule_payload_override(self, replica_name: str, event_id: str, values: dict) -> None:
        self.payload_overrides[(replica_name, event_id)] = dict(values)
