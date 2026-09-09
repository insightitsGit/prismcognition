from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from prismcognition.engine.inquiry import extract_inquiry_features, structurally_incompatible
from prismcognition.schemas.core import (
    AllocationScoreSource,
    ClusterRoute,
    ExecutionTier,
    RoutePlan,
)


class DepthAllocatingRouter:
    """Low relevance yields PROBE. SKIPPED requires typed structural incompatibility."""

    def __init__(self, classifier_client, required_threshold: float = 0.40, high_risk_floor: float = 0.50):
        self.classifier = classifier_client
        self.required_threshold = required_threshold
        self.high_risk_floor = high_risk_floor

    async def route(self, inquiry: str, risk_level: str = "HIGH") -> RoutePlan:
        features = extract_inquiry_features(inquiry)
        scores: Dict[int, float] = await self.classifier.score_clusters(inquiry)
        plan: List[ClusterRoute] = []

        for cluster_id in range(1, 8):
            if cluster_id == 7:
                plan.append(
                    ClusterRoute(
                        cluster_id=7,
                        tier=ExecutionTier.REQUIRED,
                        justification="Via Negativa failure analysis is mandatory across all strategic deliberations.",
                        classifier_score=None,
                        allocated_score=None,
                        score_source=AllocationScoreSource.MANDATORY,
                    )
                )
                continue

            if structurally_incompatible(cluster_id, features):
                raw_score, missing = _raw_score(scores, cluster_id)
                plan.append(
                    ClusterRoute(
                        cluster_id=cluster_id,
                        tier=ExecutionTier.SKIPPED,
                        justification=(
                            "Structural domain rule: typed inquiry features are formally "
                            "disjoint from this cluster's ontology."
                        ),
                        classifier_score=None if missing else raw_score,
                        allocated_score=None,
                        score_source=AllocationScoreSource.STRUCTURAL_SKIP,
                    )
                )
                continue

            allocated, classifier_score, source, missing = _allocate_score(
                scores, cluster_id, risk_level, self.high_risk_floor
            )
            if allocated >= self.required_threshold:
                tier = ExecutionTier.REQUIRED
                justification = f"Direct cluster relevance ({allocated:.2f}); allocated full catalog."
            else:
                tier = ExecutionTier.PROBE
                justification = f"Low surface relevance ({allocated:.2f}); allocated primary-method probe."
            if missing:
                justification += (
                    " Classifier key was absent; 0.00 is a compute default for depth only, "
                    "not an empty-cluster or safety finding."
                )
            if source == AllocationScoreSource.HIGH_RISK_FLOOR:
                justification += f" High-risk floor {self.high_risk_floor:.2f} raised clusters 4/5."
            plan.append(
                ClusterRoute(
                    cluster_id=cluster_id,
                    tier=tier,
                    justification=justification,
                    classifier_score=classifier_score,
                    allocated_score=allocated,
                    score_source=source,
                )
            )

        return RoutePlan(routes=tuple(plan), inquiry_features=features)


def _raw_score(scores: Dict[int, float], cluster_id: int) -> Tuple[float, bool]:
    if cluster_id not in scores:
        return 0.0, True
    return float(scores[cluster_id]), False


def _allocate_score(
    scores: Dict[int, float],
    cluster_id: int,
    risk_level: str,
    high_risk_floor: float,
) -> Tuple[float, Optional[float], AllocationScoreSource, bool]:
    raw, missing = _raw_score(scores, cluster_id)
    classifier_score = None if missing else raw
    allocated = 0.0 if missing else raw
    source = AllocationScoreSource.MISSING_COMPUTE_DEFAULT if missing else AllocationScoreSource.CLASSIFIER
    if risk_level == "HIGH" and cluster_id in (4, 5):
        floored = max(allocated, high_risk_floor)
        if floored > allocated:
            allocated = floored
            source = AllocationScoreSource.HIGH_RISK_FLOOR
        else:
            allocated = floored
    return allocated, classifier_score, source, missing
