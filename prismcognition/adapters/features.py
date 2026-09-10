from __future__ import annotations

from typing import Dict

from prismcognition.engine.inquiry import extract_inquiry_features


class FeatureClassifier:
    """Scores clusters from typed inquiry features. Missing signal stays at probe depth, not SKIPPED."""

    BASE = 0.18

    async def score_clusters(self, inquiry: str) -> Dict[int, float]:
        features = extract_inquiry_features(inquiry)
        scores = {index: self.BASE for index in range(1, 7)}
        if features.formal_a_priori:
            scores[1] = max(scores[1], 0.86)
        if features.empirical_measurement:
            scores[2] = max(scores[2], 0.86)
        if features.causal_mechanism:
            scores[3] = max(scores[3], 0.82)
        if features.strategic_decision:
            scores[4] = max(scores[4], 0.72)
            scores[5] = max(scores[5], 0.72)
        if features.normative_judgment:
            scores[6] = max(scores[6], 0.86)
        if features.physical_constant_query:
            scores[1] = max(scores[1], 0.70)
            scores[2] = max(scores[2], 0.55)
        if features.catastrophic_stakes:
            scores[3] = max(scores[3], 0.84)
            scores[4] = max(scores[4], 0.90)
            scores[5] = max(scores[5], 0.88)
            scores[6] = max(scores[6], 0.90)
        if features.routine_or_aesthetic:
            scores[1] = min(scores[1], 0.16)
            scores[3] = min(scores[3], 0.16)
            scores[4] = min(scores[4], 0.28)
            scores[5] = max(scores[5], 0.62)
            scores[6] = max(scores[6], 0.48)
        return scores
