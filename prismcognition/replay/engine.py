from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

import numpy as np

from prismcognition.engine.coverage import attach_coverage
from prismcognition.engine.derive import assumptions_that_matter, collect_pair_artifacts, select_strongly_supported
from prismcognition.engine.tension import HardenedTensionEngine
from prismcognition.identity import FLOAT_QUANTUM
from prismcognition.schemas.core import (
    DeliberationArtifact,
    FrozenDeliberationBundle,
    GroundingProfile,
    WarrantGrounding,
    index_claims,
)
from prismcognition.subtractive.gate import SubtractiveGate

__all__ = ["ReplayEngine", "FrozenDeliberationBundle", "freeze_profiles"]


class ReplayEngine:
    @staticmethod
    def replay(bundle: FrozenDeliberationBundle) -> DeliberationArtifact:
        gate = SubtractiveGate(tol=bundle.threshold_config.get("subtractive_rank_tol", 1e-6))
        weights = (
            bundle.threshold_config.get("w_pred", 0.40),
            bundle.threshold_config.get("w_warr", 0.30),
            bundle.threshold_config.get("w_assump", 0.20),
            bundle.threshold_config.get("w_norm", 0.10),
        )
        clash_threshold = bundle.threshold_config.get("structural_clash", 0.35)
        tension_engine = HardenedTensionEngine(
            clash_threshold=clash_threshold,
            assumption_spread=bundle.threshold_config.get("assumption_spread", 0.25),
            weights=weights,
        )

        stance_pairs = [
            (stance, np.array(stance.semantic_vector, dtype=float) if stance.semantic_vector else None)
            for stance in bundle.frozen_stances
        ]
        viable_pairs, _diagnostics = gate.apply_subtractive_gate(stance_pairs, bundle.frozen_ruin_result)
        viable_stances = [stance for stance, _vector in viable_pairs]

        profiles: Dict[str, GroundingProfile] = {}
        for stance_id, payload in bundle.frozen_grounding_profiles.items():
            profiles[stance_id] = (
                payload if isinstance(payload, GroundingProfile) else GroundingProfile.model_validate(payload)
            )

        warrant_groundings: Dict[str, tuple[WarrantGrounding, ...]] = {}
        for stance_id, items in bundle.frozen_warrant_groundings.items():
            warrant_groundings[stance_id] = tuple(
                item if isinstance(item, WarrantGrounding) else WarrantGrounding.model_validate(item) for item in items
            )

        claims_universe = index_claims(viable_stances)
        disagreements, resolved, diversities, evidence_needed, irreducible = collect_pair_artifacts(
            viable_stances,
            profiles,
            claims_universe,
            tension_engine,
            clash_threshold,
        )
        supported = select_strongly_supported(
            viable_stances,
            profiles,
            warrant_groundings,
            bundle.threshold_config.get("support_threshold", 0.70),
        )
        artifact = DeliberationArtifact(
            deliberation_id=bundle.deliberation_id,
            inquiry_text=bundle.inquiry_text,
            methods_executed=bundle.methods_executed,
            methods_probed=bundle.methods_probed,
            methods_skipped=bundle.methods_skipped,
            methods_failed=bundle.methods_failed,
            strongly_supported_claims=tuple(supported),
            ruin_analysis=bundle.frozen_ruin_result,
            active_disagreements=tuple(disagreements),
            perspective_diversities=tuple(diversities),
            evidence_needed=evidence_needed,
            assumptions_that_matter=assumptions_that_matter(viable_stances, list(disagreements) + list(resolved)),
            irreducible_tensions=irreducible,
            optional_recommendation=bundle.optional_recommendation,
            route_plan=bundle.frozen_route_plan,
            resolved_disagreements=tuple(resolved),
            thesis_domain_key=bundle.thesis_domain_key,
            execution_mode=bundle.execution_mode,
            live_requested=bundle.live_requested,
            execution_notes=bundle.execution_notes,
            provenance=bundle.provenance,
        )
        return attach_coverage(artifact)

    @staticmethod
    def canonical_hash(artifact: DeliberationArtifact) -> str:
        dump = _quantize(artifact.model_dump(mode="json"))
        canonical = json.dumps(dump, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def freeze_profiles(profiles: Dict[str, GroundingProfile]) -> Dict[str, GroundingProfile]:
    return dict(profiles)


def _quantize(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, FLOAT_QUANTUM)
    if isinstance(value, dict):
        return {key: _quantize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_quantize(item) for item in value]
    return value
