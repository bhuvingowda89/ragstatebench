from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class OracleResult:
    eligible_document_ids: frozenset[str]
    answerable: bool

class QueryOracle:
    def __init__(self, ledger):
        self.ledger = ledger

    def evaluate(self, query_id: str, principal_id: str, t: int) -> OracleResult:
        eligible = frozenset(self.ledger.eligible_evidence(query_id, principal_id, t))
        return OracleResult(
            eligible_document_ids=eligible,
            answerable=bool(eligible),
        )
