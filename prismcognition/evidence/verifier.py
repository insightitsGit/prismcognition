from __future__ import annotations

from typing import Dict, Optional

from prismcognition.identity import artifact_id, new_provenance
from prismcognition.schemas.core import (
    GroundingRegime,
    GroundingStatus,
    StructuredClaim,
    Warrant,
    WarrantGrounding,
)

NOT_GROUNDABLE_REGIMES = {
    GroundingRegime.NORMATIVE,
    GroundingRegime.INTERPRETIVE,
    GroundingRegime.PHENOMENOLOGICAL,
    GroundingRegime.UNVERIFIABLE,
}


class RegimeAwareGroundingVerifier:
    """Verifies warrants only inside their declared regime. Absence is None, not 0.0."""

    def __init__(self, evidence_adapter=None):
        self.evidence = evidence_adapter

    async def verify_warrant(
        self,
        warrant: Warrant,
        claims_universe: Dict[str, StructuredClaim],
    ) -> WarrantGrounding:
        snapshot_id = evidence_snapshot_id(self) or warrant.provenance.evidence_snapshot_id
        provenance = new_provenance(
            artifact_id_value=artifact_id("wground", warrant.warrant_id, warrant.epistemic_regime.value),
            parent_ids=(warrant.provenance.artifact_id,),
            model_provider="internal",
            model_id="grounding_verifier",
            prompt_hash="deterministic-internal",
            created_at=warrant.provenance.created_at,
            evidence_snapshot_id=snapshot_id,
        )

        if warrant.epistemic_regime in NOT_GROUNDABLE_REGIMES:
            return WarrantGrounding(
                warrant_id=warrant.warrant_id,
                epistemic_regime=warrant.epistemic_regime,
                status=GroundingStatus.NOT_GROUNDABLE,
                support_score=None,
                epistemic_reasoning="Native regime is non-groundable; score withheld rather than forced through empirical pipelines.",
                provenance=provenance,
            )

        if self.evidence is not None:
            status, score, support, counter, reason = await self.evidence.assess(warrant, claims_universe)
            if score is not None and not 0.0 <= score <= 1.0:
                raise ValueError("evidence adapter returned a score outside [0, 1]")
            return WarrantGrounding(
                warrant_id=warrant.warrant_id,
                epistemic_regime=warrant.epistemic_regime,
                status=status,
                support_score=score,
                evidence_refs=support,
                counterevidence_refs=counter,
                epistemic_reasoning=reason,
                provenance=provenance,
            )

        missing_premises = [item for item in warrant.premise_claim_ids if item not in claims_universe]
        conclusion = claims_universe.get(warrant.conclusion_claim_id)
        if missing_premises or conclusion is None:
            return WarrantGrounding(
                warrant_id=warrant.warrant_id,
                epistemic_regime=warrant.epistemic_regime,
                status=GroundingStatus.UNRESOLVED,
                support_score=None,
                epistemic_reasoning="Claim graph is incomplete; unresolved is not a refutation.",
                provenance=provenance,
            )

        return WarrantGrounding(
            warrant_id=warrant.warrant_id,
            epistemic_regime=warrant.epistemic_regime,
            status=GroundingStatus.INSUFFICIENT_EVIDENCE,
            support_score=None,
            epistemic_reasoning="No evidence snapshot is bound. Declared confidence is not treated as grounding.",
            provenance=provenance,
        )


def evidence_snapshot_id(verifier) -> Optional[str]:
    if verifier is None:
        return None
    evidence = getattr(verifier, "evidence", None)
    snapshot = getattr(evidence, "snapshot_id", None)
    return str(snapshot) if snapshot else None


class StaticEvidenceAdapter:
    """Test/replay helper: explicit per-warrant scores. Missing warrants stay None."""

    def __init__(
        self,
        scores: Optional[Dict[str, tuple[GroundingStatus, Optional[float]]]] = None,
        snapshot_id: Optional[str] = None,
    ):
        self.scores = scores or {}
        self.snapshot_id = snapshot_id

    async def assess(
        self,
        warrant: Warrant,
        claims_universe: Dict[str, StructuredClaim],
    ) -> tuple[GroundingStatus, Optional[float], tuple[str, ...], tuple[str, ...], str]:
        _ = claims_universe
        if warrant.warrant_id not in self.scores:
            return (
                GroundingStatus.INSUFFICIENT_EVIDENCE,
                None,
                (),
                (),
                "No evidence row for this warrant.",
            )
        status, score = self.scores[warrant.warrant_id]
        return status, score, (f"static:{warrant.warrant_id}",), (), "Static evidence adapter assessment."
