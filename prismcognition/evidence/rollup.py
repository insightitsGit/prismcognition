from __future__ import annotations

from typing import List

from prismcognition.identity import artifact_id, new_provenance
from prismcognition.schemas.core import (
    GroundingProfile,
    GroundingRegime,
    GroundingStatus,
    RegimeAssessment,
    WarrantGrounding,
)


class GroundingRollupEngine:
    @staticmethod
    def compile_profile(groundings: List[WarrantGrounding]) -> GroundingProfile:
        assessments: List[RegimeAssessment] = []
        for regime in GroundingRegime:
            regime_groundings = [item for item in groundings if item.epistemic_regime == regime]
            if not regime_groundings:
                assessments.append(RegimeAssessment(regime=regime, applicable=False, score=None))
                continue

            scores = [item.support_score for item in regime_groundings if item.support_score is not None]
            average = float(sum(scores) / len(scores)) if scores else None
            assessments.append(
                RegimeAssessment(
                    regime=regime,
                    score=average,
                    applicable=True,
                    supported_count=sum(item.status == GroundingStatus.SUPPORTED for item in regime_groundings),
                    contradicted_count=sum(item.status == GroundingStatus.CONTRADICTED for item in regime_groundings),
                    contested_count=sum(item.status == GroundingStatus.CONTESTED for item in regime_groundings),
                    unresolved_count=sum(item.status == GroundingStatus.UNRESOLVED for item in regime_groundings),
                    not_groundable_count=sum(item.status == GroundingStatus.NOT_GROUNDABLE for item in regime_groundings),
                    insufficient_evidence_count=sum(
                        item.status == GroundingStatus.INSUFFICIENT_EVIDENCE for item in regime_groundings
                    ),
                )
            )
        parents = tuple(item.provenance.artifact_id for item in groundings)
        return GroundingProfile(
            regime_assessments=tuple(assessments),
            provenance=new_provenance(
                artifact_id_value=artifact_id("gprof", *(item.warrant_id for item in groundings)),
                parent_ids=parents,
                model_provider="internal",
                model_id="grounding_rollup",
                prompt_hash="deterministic-internal",
            ),
        )
