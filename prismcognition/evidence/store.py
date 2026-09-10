from __future__ import annotations

import json
from prismcognition.persist.atomic import atomic_write_text
from pathlib import Path
from typing import Dict, List, Optional

from prismcognition.schemas.core import (
    EvidenceRecord,
    GroundingRegime,
    GroundingStatus,
    Polarity,
    StructuredClaim,
    Warrant,
)


class EvidenceStore:
    """Snapshot ledger. Missing rows stay None; they are never coerced to 0.0."""

    def __init__(self, snapshot_id: str = "default"):
        self.snapshot_id = snapshot_id
        self._records: Dict[str, EvidenceRecord] = {}

    def ingest(self, record: EvidenceRecord) -> None:
        self._records[record.record_id] = record

    def records_for(self, domain_key: str, regime: GroundingRegime) -> List[EvidenceRecord]:
        return [
            record
            for record in self._records.values()
            if record.domain_key == domain_key and record.regime == regime
        ]

    async def assess(
        self,
        warrant: Warrant,
        claims_universe: Dict[str, StructuredClaim],
    ) -> tuple[GroundingStatus, Optional[float], tuple[str, ...], tuple[str, ...], str]:
        conclusion = claims_universe.get(warrant.conclusion_claim_id)
        if conclusion is None:
            return (
                GroundingStatus.UNRESOLVED,
                None,
                (),
                (),
                "Conclusion is not in the claims universe; unresolved is not refutation.",
            )

        matches = self.records_for(conclusion.domain_key, warrant.epistemic_regime)
        if not matches:
            return (
                GroundingStatus.INSUFFICIENT_EVIDENCE,
                None,
                (),
                (),
                f"No evidence rows in snapshot '{self.snapshot_id}' for {conclusion.domain_key}/{warrant.epistemic_regime.value}.",
            )

        # Only accepted assertions can support or contradict another claim.
        # A rejected/unresolved assertion is not evidence of its inverse.
        contested = [item for item in matches if item.status == GroundingStatus.CONTESTED]
        if contested:
            return (
                GroundingStatus.CONTESTED, None, (), (),
                "Matched evidence includes contested assertions; support withheld pending review.",
            )
        matches = [item for item in matches if item.status == GroundingStatus.SUPPORTED]
        if not matches:
            return (
                GroundingStatus.INSUFFICIENT_EVIDENCE, None, (), (),
                "No accepted SUPPORTED assertions match; rejected or unresolved rows cannot establish truth.",
            )
        supporting = [item for item in matches if item.polarity == conclusion.polarity]
        opposing = [
            item
            for item in matches
            if item.polarity != Polarity.NEUTRAL and item.polarity != conclusion.polarity
        ]
        support_refs = tuple(item.source_ref for item in supporting)
        counter_refs = tuple(item.source_ref for item in opposing)

        if supporting and opposing:
            return (
                GroundingStatus.CONTESTED,
                _mean_scores(supporting + opposing),
                support_refs,
                counter_refs,
                "Supporting and counterevidence both present.",
            )
        if opposing and not supporting:
            return (
                GroundingStatus.CONTRADICTED,
                _mean_scores(opposing),
                support_refs,
                counter_refs,
                "Only counterevidence is present.",
            )
        if supporting:
            return (
                GroundingStatus.SUPPORTED,
                _mean_scores(supporting),
                support_refs,
                counter_refs,
                "Polarity-aligned evidence is present.",
            )
        return (
            GroundingStatus.UNRESOLVED,
            None,
            support_refs,
            counter_refs,
            "Matched rows are polarity-neutral; score withheld.",
        )

    def save_json(self, path: Path) -> None:
        payload = {
            "snapshot_id": self.snapshot_id,
            "records": [record.model_dump(mode="json") for record in self._records.values()],
        }
        atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True))

    @classmethod
    def load_json(cls, path: Path) -> "EvidenceStore":
        payload = json.loads(path.read_text(encoding="utf-8"))
        store = cls(snapshot_id=str(payload.get("snapshot_id", "default")))
        for item in payload.get("records", []):
            store.ingest(EvidenceRecord.model_validate(item))
        return store


def _mean_scores(records: List[EvidenceRecord]) -> Optional[float]:
    scores = [item.support_score for item in records if item.support_score is not None]
    if not scores:
        return None
    return float(sum(scores) / len(scores))
