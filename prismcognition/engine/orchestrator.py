from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from prismcognition.engine.clusters import default_cluster_registry
from prismcognition.engine.coverage import attach_coverage
from prismcognition.engine.derive import assumptions_that_matter, collect_pair_artifacts, select_strongly_supported
from prismcognition.engine.recommend import draft_optional_recommendation
from prismcognition.engine.router import DepthAllocatingRouter
from prismcognition.identity import artifact_id, derive_provenance, new_provenance, prompt_digest
from prismcognition.engine.tension import HardenedTensionEngine
from prismcognition.evidence.rollup import GroundingRollupEngine
from prismcognition.evidence.verifier import evidence_snapshot_id
from prismcognition.schemas.core import (
    DeliberationArtifact,
    EpistemicStance,
    ExecutionTier,
    FrozenDeliberationBundle,
    GroundingProfile,
    RuinAnalysisStatus,
    WarrantGrounding,
    index_claims,
)
from prismcognition.subtractive.containment import ChaosRoomSandbox
from prismcognition.subtractive.gate import SubtractiveGate

DEFAULT_THRESHOLDS = {
    "subtractive_rank_tol": 1e-6,
    "structural_clash": 0.35,
    "assumption_spread": 0.25,
    "support_threshold": 0.70,
    "router_required": 0.40,
    "high_risk_floor": 0.50,
    "w_pred": 0.40,
    "w_warr": 0.30,
    "w_assump": 0.20,
    "w_norm": 0.10,
}


class EpistemicOrchestrator:
    def __init__(
        self,
        router: DepthAllocatingRouter,
        chaos_room: ChaosRoomSandbox,
        cluster_registry: Optional[Dict[str, Any]] = None,
        grounding_verifier: Any = None,
        clash_threshold: float = 0.35,
        threshold_config: Optional[Dict[str, float]] = None,
    ):
        self.router = router
        self.chaos_room = chaos_room
        self.clusters = cluster_registry or default_cluster_registry()
        self.verifier = grounding_verifier
        self.thresholds = {**DEFAULT_THRESHOLDS, **(threshold_config or {})}
        self.thresholds["structural_clash"] = clash_threshold
        self.subtractive_gate = SubtractiveGate(tol=self.thresholds["subtractive_rank_tol"])
        self.tension = HardenedTensionEngine(
            clash_threshold=clash_threshold,
            assumption_spread=self.thresholds["assumption_spread"],
            weights=(
                self.thresholds["w_pred"],
                self.thresholds["w_warr"],
                self.thresholds["w_assump"],
                self.thresholds["w_norm"],
            ),
        )
        self._apply_router_thresholds()
        self._last_bundle: Optional[FrozenDeliberationBundle] = None

    def _apply_router_thresholds(self) -> None:
        if hasattr(self.router, "required_threshold"):
            self.router.required_threshold = self.thresholds["router_required"]
        if hasattr(self.router, "high_risk_floor"):
            self.router.high_risk_floor = self.thresholds["high_risk_floor"]

    async def deliberate(
        self,
        inquiry: str,
        risk_level: str = "HIGH",
        emit_recommendation: bool = False,
    ) -> DeliberationArtifact:
        deliberation_id = artifact_id("delib", inquiry, risk_level)
        snapshot_id = evidence_snapshot_id(self.verifier)
        provenance = new_provenance(
            artifact_id_value=deliberation_id,
            model_provider="orchestrator",
            model_id="v2.1-engine",
            prompt_hash=prompt_digest(inquiry),
            evidence_snapshot_id=snapshot_id,
        )

        route_plan = await self.router.route(inquiry, risk_level=risk_level)
        methods_skipped = tuple(
            str(route.cluster_id) for route in route_plan.routes if route.tier == ExecutionTier.SKIPPED
        )

        chaos_task = asyncio.create_task(self.chaos_room.generate_quarantined_boundaries(inquiry))
        exec_tasks: Dict[str, Tuple[ExecutionTier, asyncio.Task]] = {}
        for route in route_plan.routes:
            cluster_id = str(route.cluster_id)
            if route.tier == ExecutionTier.SKIPPED or cluster_id == "7":
                continue
            evaluator = self.clusters.get(cluster_id)
            if evaluator is None:
                continue
            exec_tasks[cluster_id] = (
                route.tier,
                asyncio.create_task(_run_cluster(evaluator, inquiry, route.tier)),
            )

        ruin_result = await chaos_task
        executed_stances: List[EpistemicStance] = []
        methods_executed: List[str] = []
        methods_probed: List[str] = []
        methods_failed: List[str] = []

        if ruin_result.status in (RuinAnalysisStatus.FAILED, RuinAnalysisStatus.TIMEOUT):
            methods_failed.append(f"7: ruin_analysis_{ruin_result.status.value}")

        for cluster_id, (tier, task) in exec_tasks.items():
            try:
                stances = await task
                executed_stances.extend(stances)
                labels = [f"{cluster_id}_{stance.method_id}" for stance in stances]
                if tier == ExecutionTier.PROBE:
                    methods_probed.extend(labels)
                else:
                    methods_executed.extend(labels)
            except Exception as exc:
                methods_failed.append(f"{cluster_id}: {type(exc).__name__}")

        stance_pairs = [
            (stance, np.array(stance.semantic_vector, dtype=float) if stance.semantic_vector else None)
            for stance in executed_stances
        ]
        viable_pairs, dropped = self.subtractive_gate.apply_subtractive_gate(stance_pairs, ruin_result)
        viable_stances = [stance for stance, _vector in viable_pairs]
        if ruin_result.status in (RuinAnalysisStatus.FAILED, RuinAnalysisStatus.TIMEOUT) and not ruin_result.diagnostics:
            ruin_result = ruin_result.model_copy(
                update={"diagnostics": ("Ruin analysis did not complete; absence of boundaries is not safety.",)}
            )
        if dropped:
            ruin_result = ruin_result.model_copy(
                update={"diagnostics": tuple(ruin_result.diagnostics) + tuple(dropped)}
            )

        claims_universe = index_claims(viable_stances)
        grounding_profiles: Dict[str, GroundingProfile] = {}
        warrant_groundings: Dict[str, Tuple[WarrantGrounding, ...]] = {}
        for stance in viable_stances:
            if self.verifier is None or not stance.warrants:
                grounding_profiles[stance.stance_id] = GroundingRollupEngine.compile_profile([])
                warrant_groundings[stance.stance_id] = ()
                continue
            groundings = await asyncio.gather(
                *[self.verifier.verify_warrant(warrant, claims_universe) for warrant in stance.warrants]
            )
            warrant_groundings[stance.stance_id] = tuple(groundings)
            grounding_profiles[stance.stance_id] = GroundingRollupEngine.compile_profile(list(groundings))

        disagreements, diversities, evidence_needed, irreducible = collect_pair_artifacts(
            viable_stances,
            grounding_profiles,
            claims_universe,
            self.tension,
            self.thresholds["structural_clash"],
        )
        supported = select_strongly_supported(
            viable_stances,
            grounding_profiles,
            warrant_groundings,
            self.thresholds["support_threshold"],
        )
        artifact = DeliberationArtifact(
            deliberation_id=deliberation_id,
            inquiry_text=inquiry,
            methods_executed=tuple(methods_executed),
            methods_probed=tuple(methods_probed),
            methods_skipped=methods_skipped,
            methods_failed=tuple(methods_failed),
            strongly_supported_claims=tuple(supported),
            ruin_analysis=ruin_result,
            active_disagreements=tuple(disagreements),
            perspective_diversities=tuple(diversities),
            evidence_needed=evidence_needed,
            assumptions_that_matter=assumptions_that_matter(viable_stances, disagreements),
            irreducible_tensions=irreducible,
            optional_recommendation=None,
            route_plan=route_plan,
            provenance=provenance,
        )
        artifact = attach_coverage(artifact)
        if emit_recommendation:
            artifact = artifact.model_copy(
                update={
                    "optional_recommendation": draft_optional_recommendation(artifact),
                    "provenance": derive_provenance(
                        artifact.provenance,
                        artifact_id_value=artifact_id("delib_rec", deliberation_id),
                        model_provider="internal",
                        model_id="optional_recommendation",
                        prompt_hash="deterministic-internal",
                    ),
                }
            )
            artifact = attach_coverage(artifact)
        self._last_bundle = FrozenDeliberationBundle(
            deliberation_id=artifact.deliberation_id,
            inquiry_text=artifact.inquiry_text,
            frozen_stances=tuple(executed_stances),
            frozen_ruin_result=ruin_result,
            frozen_grounding_profiles=dict(grounding_profiles),
            frozen_warrant_groundings=dict(warrant_groundings),
            methods_executed=artifact.methods_executed,
            methods_probed=artifact.methods_probed,
            methods_skipped=artifact.methods_skipped,
            methods_failed=artifact.methods_failed,
            evidence_needed=artifact.evidence_needed,
            irreducible_tensions=artifact.irreducible_tensions,
            optional_recommendation=artifact.optional_recommendation,
            threshold_config=dict(self.thresholds),
            frozen_route_plan=route_plan,
            provenance=artifact.provenance,
        )
        return artifact

    def last_frozen_bundle(self) -> FrozenDeliberationBundle:
        if self._last_bundle is None:
            raise RuntimeError("deliberate() has not been run")
        return self._last_bundle


async def _run_cluster(evaluator: Any, inquiry: str, depth: ExecutionTier) -> List[EpistemicStance]:
    if hasattr(evaluator, "evaluate_all"):
        return list(await evaluator.evaluate_all(inquiry, depth=depth))
    stance = await evaluator.evaluate(inquiry, depth=depth)
    return [stance]
