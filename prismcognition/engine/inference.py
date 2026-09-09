from __future__ import annotations

from typing import Dict

from prismcognition.schemas.core import Polarity, StructuredClaim, Warrant


class InferenceConflictEvaluator:
    @staticmethod
    def compatible_premises(
        w1: Warrant,
        w2: Warrant,
        claims_universe: Dict[str, StructuredClaim],
    ) -> bool:
        missing = [
            premise_id
            for premise_id in tuple(w1.premise_claim_ids) + tuple(w2.premise_claim_ids)
            if premise_id not in claims_universe
        ]
        if missing:
            return False

        left = {
            claims_universe[premise_id].domain_key: claims_universe[premise_id].polarity
            for premise_id in w1.premise_claim_ids
        }
        right = {
            claims_universe[premise_id].domain_key: claims_universe[premise_id].polarity
            for premise_id in w2.premise_claim_ids
        }
        shared = set(left).intersection(right)
        return all(left[key] == right[key] for key in shared)

    @staticmethod
    def contradictory_claims(left: StructuredClaim, right: StructuredClaim) -> bool:
        return (
            left.domain_key == right.domain_key
            and left.polarity != Polarity.NEUTRAL
            and right.polarity != Polarity.NEUTRAL
            and left.polarity != right.polarity
        )

    @staticmethod
    def evaluate_inference_clash(
        w1: Warrant,
        w2: Warrant,
        claims_universe: Dict[str, StructuredClaim],
    ) -> bool:
        conclusion_left = claims_universe.get(w1.conclusion_claim_id)
        conclusion_right = claims_universe.get(w2.conclusion_claim_id)
        if conclusion_left is None or conclusion_right is None:
            return False
        if not InferenceConflictEvaluator.contradictory_claims(conclusion_left, conclusion_right):
            return False
        return InferenceConflictEvaluator.compatible_premises(w1, w2, claims_universe)
