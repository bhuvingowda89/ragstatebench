from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Delivery:
    at: int
    replica_name: str
    event: object

class ReplicaSimulator:
    def __init__(self, replicas: dict[str, object], fault_schedule):
        self.replicas = replicas
        self.fault_schedule = fault_schedule
        self.queue: list[Delivery] = []
        self.time = 0

    def submit(self, event) -> None:
        for name in self.replicas:
            dt = self.fault_schedule.delivery_time(name, event)
            if dt is not None:
                transformed = self.fault_schedule.transform(name, event)
                self.queue.append(Delivery(dt, name, transformed))
        self.queue.sort(key=lambda x: (x.at, x.replica_name, x.event.event_id))

    def run_until(self, t: int) -> list[Delivery]:
        delivered = []
        remaining = []
        for item in self.queue:
            if item.at <= t:
                self.replicas[item.replica_name].apply(item.event)
                delivered.append(item)
            else:
                remaining.append(item)
        self.queue = remaining
        self.time = max(self.time, t)
        return delivered
