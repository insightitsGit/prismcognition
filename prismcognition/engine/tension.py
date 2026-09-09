from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from prismcognition.engine.inference import InferenceConflictEvaluator
from prismcognition.identity import artifact_id
from prismcognition.schemas.core import (
    AssumptionMidpoint,
    EpistemicClash,
    EpistemicPivot,
    EpistemicStance,
    GroundingProfile,
    GroundingRegime,
    PerspectiveDiversity,
    Polarity,
    ProvenanceRef,
    ResolutionStatus,
    StructuredClaim,
    TensionProfile,
)


@dataclass(frozen=True)
class PairEvaluation:
    raw_delta: float
    calibrated: Optional[float]
    profile: TensionProfile
    status: ResolutionStatus
    clashes: List[EpistemicClash]
    diversities: List[PerspectiveDiversity]
    midpoint: Optional[AssumptionMidpoint]
    pivot: Optional[EpistemicPivot]
    midpoints: Tuple[AssumptionMidpoint, ...]
    pivots: Tuple[EpistemicPivot, ...]


class HardenedTensionEngine:
    def __init__(
        self,
        clash_threshold: float = 0.35,
        assumption_spread: float = 0.25,
        weights: Optional[Tuple[float, float, float, float]] = None,
    ):
        self.tau_clash = clash_threshold
        self.assumption_spread = assumption_spread
        self.weights = weights or (0.40, 0.30, 0.20, 0.10)

    def evaluate_pair(
        self,
        s1: EpistemicStance,
        s2: EpistemicStance,
        p1: GroundingProfile,
        p2: GroundingProfile,
        claims_universe: Dict[str, StructuredClaim],
    ) -> PairEvaluation:
        clashes: List[EpistemicClash] = []
        diversities: List[PerspectiveDiversity] = []
        provenance = ProvenanceRef(
            artifact_id=artifact_id("prov_eval", s1.stance_id, s2.stance_id),
            parent_ids=(s1.provenance.artifact_id, s2.provenance.artifact_id),
            model_provider="internal",
            model_id="tension_engine",
            prompt_hash="deterministic-internal",
            created_at=s1.provenance.created_at,
        )

        d_pred = self._eval_predicates(s1, s2, clashes, provenance)
        d_warr = self._eval_warrants(s1, s2, clashes, diversities, claims_universe, provenance)
        d_assump, midpoint, pivot, midpoints, pivots = self._eval_assumptions(s1, s2, clashes, provenance)
        d_norm = self._eval_normative(s1, s2, clashes, diversities, provenance)
        d_causal = self._eval_regime_warrants(
            s1, s2, clashes, claims_universe, provenance, GroundingRegime.CAUSAL, "CAUSAL_CONFLICT"
        )
        self._eval_regime_warrants(
            s1, s2, clashes, claims_universe, provenance, GroundingRegime.DEFINITIONAL, "DEFINITIONAL_CONFLICT"
        )

        if (
            s1.cluster_id != s2.cluster_id
            and not clashes
            and not any(item.diversity_type == "METHODOLOGICAL_COMPLEMENT" for item in diversities)
        ):
            diversities.append(
                PerspectiveDiversity(
                    diversity_id=artifact_id("div_method", s1.stance_id, s2.stance_id),
                    diversity_type="METHODOLOGICAL_COMPLEMENT",
                    stance_id_left=s1.stance_id,
                    stance_id_right=s2.stance_id,
                    explanation="Distinct methodological clusters without adversarial clash.",
                    provenance=provenance,
                )
            )

        weight_pred, weight_warr, weight_assump, weight_norm = self.weights
        raw_delta = weight_pred * d_pred + weight_warr * d_warr + weight_assump * d_assump + weight_norm * d_norm
        profile = self._build_tension_profile(
            d_pred, d_warr, d_assump, d_norm, d_causal, raw_delta, p1, p2, provenance
        )
        calibrated = self._mean_or_none([profile.empirical, profile.formal, profile.causal, profile.assumption])
        status = self._derive_resolution_status(raw_delta, clashes, p1, p2, midpoint is not None)
        return PairEvaluation(
            raw_delta=raw_delta,
            calibrated=calibrated,
            profile=profile,
            status=status,
            clashes=clashes,
            diversities=diversities,
            midpoint=midpoint,
            pivot=pivot,
            midpoints=midpoints,
            pivots=pivots,
        )

    def _eval_predicates(
        self,
        s1: EpistemicStance,
        s2: EpistemicStance,
        clashes: List[EpistemicClash],
        provenance: ProvenanceRef,
    ) -> float:
        left = {item.domain_key: item for item in s1.propositions}
        right = {item.domain_key: item for item in s2.propositions}
        shared = set(left).intersection(right)
        if not shared:
            return 0.0

        conflicts = 0
        for key in shared:
            if left[key].polarity == Polarity.NEUTRAL or right[key].polarity == Polarity.NEUTRAL:
                continue
            if left[key].polarity != right[key].polarity:
                conflicts += 1
                clashes.append(
                    EpistemicClash(
                        clash_id=f"pred_{key}",
                        clash_type="FACTUAL_CONFLICT",
                        clash_regime=GroundingRegime.EMPIRICAL,
                        stance_id_left=s1.stance_id,
                        stance_id_right=s2.stance_id,
                        explanation=f"Direct polarity conflict on predicate '{key}': {left[key].polarity} vs {right[key].polarity}",
                        provenance=provenance,
                    )
                )
        return conflicts / len(shared)

    def _eval_warrants(
        self,
        s1: EpistemicStance,
        s2: EpistemicStance,
        clashes: List[EpistemicClash],
        diversities: List[PerspectiveDiversity],
        claims_universe: Dict[str, StructuredClaim],
        provenance: ProvenanceRef,
    ) -> float:
        if not s1.warrants or not s2.warrants:
            return 0.0

        conflict_count = 0
        compared = 0
        formal_left = [item for item in s1.warrants if item.epistemic_regime == GroundingRegime.FORMAL]
        formal_right = [item for item in s2.warrants if item.epistemic_regime == GroundingRegime.FORMAL]

        for left in formal_left:
            for right in formal_right:
                if left.rule_statement != right.rule_statement:
                    continue
                compared += 1
                if InferenceConflictEvaluator.evaluate_inference_clash(left, right, claims_universe):
                    conflict_count += 1
                    conclusion = claims_universe.get(left.conclusion_claim_id)
                    domain = conclusion.domain_key if conclusion else left.conclusion_claim_id
                    clashes.append(
                        EpistemicClash(
                            clash_id=f"warr_rule_{left.rule_statement[:16]}",
                            clash_type="INFERENCE_RULE_CONFLICT",
                            clash_regime=GroundingRegime.FORMAL,
                            stance_id_left=s1.stance_id,
                            stance_id_right=s2.stance_id,
                            explanation=(
                                f"Compatible premises under rule '{left.rule_statement}' "
                                f"infer mutually exclusive conclusions on '{domain}'."
                            ),
                            provenance=provenance,
                        )
                    )

        if compared == 0:
            diversities.append(
                PerspectiveDiversity(
                    diversity_id=artifact_id("div_warr", s1.stance_id, s2.stance_id),
                    diversity_type="NON_OVERLAPPING_WARRANTS",
                    stance_id_left=s1.stance_id,
                    stance_id_right=s2.stance_id,
                    explanation="Stances utilize disjoint justification rules. Orthogonal perspectives, not adversarial friction.",
                    provenance=provenance,
                )
            )
            return 0.0
        return float(conflict_count / compared)

    def _eval_regime_warrants(
        self,
        s1: EpistemicStance,
        s2: EpistemicStance,
        clashes: List[EpistemicClash],
        claims_universe: Dict[str, StructuredClaim],
        provenance: ProvenanceRef,
        regime: GroundingRegime,
        clash_type: str,
    ) -> float:
        left = [item for item in s1.warrants if item.epistemic_regime == regime]
        right = [item for item in s2.warrants if item.epistemic_regime == regime]
        if not left or not right:
            return 0.0
        conflicts = 0
        pairs = 0
        for w1 in left:
            for w2 in right:
                pairs += 1
                if InferenceConflictEvaluator.evaluate_inference_clash(w1, w2, claims_universe):
                    conflicts += 1
                    clashes.append(
                        EpistemicClash(
                            clash_id=f"{regime.value.lower()}_{w1.warrant_id}_{w2.warrant_id}",
                            clash_type=clash_type,  # type: ignore[arg-type]
                            clash_regime=regime,
                            stance_id_left=s1.stance_id,
                            stance_id_right=s2.stance_id,
                            explanation=f"{regime.value} warrants yield incompatible conclusions from compatible premises.",
                            provenance=provenance,
                        )
                    )
        return float(conflicts / pairs) if pairs else 0.0

    def _eval_assumptions(
        self,
        s1: EpistemicStance,
        s2: EpistemicStance,
        clashes: List[EpistemicClash],
        provenance: ProvenanceRef,
    ) -> Tuple[float, Optional[AssumptionMidpoint], Optional[EpistemicPivot], Tuple[AssumptionMidpoint, ...], Tuple[EpistemicPivot, ...]]:
        left = {item.parameter_name: item for item in s1.assumptions}
        right = {item.parameter_name: item for item in s2.assumptions}
        shared = set(left).intersection(right)
        if not shared:
            return 0.0, None, None, (), ()

        spreads: List[float] = []
        midpoints: List[AssumptionMidpoint] = []
        pivots: List[EpistemicPivot] = []
        for key in sorted(shared):
            v1, v2 = left[key].assumed_value, right[key].assumed_value
            numeric = _as_float(v1) is not None and _as_float(v2) is not None
            if numeric:
                n1, n2 = float(_as_float(v1)), float(_as_float(v2))
                spread = abs(n1 - n2)
                denom = max(abs(n1), abs(n2), 1e-6)
                norm_spread = min(1.0, spread / denom)
                spreads.append(norm_spread)
                if norm_spread > self.assumption_spread:
                    midpoint = AssumptionMidpoint(
                        variable_name=key,
                        disputing_clusters=(s1.cluster_id, s2.cluster_id),
                        values=(n1, n2),
                        midpoint=float((n1 + n2) / 2.0),
                        provenance=provenance,
                    )
                    midpoints.append(midpoint)
                    pivots.append(
                        EpistemicPivot(
                            variable_name=key,
                            disputing_clusters=(s1.cluster_id, s2.cluster_id),
                            current_assumptions=(n1, n2),
                            pivot_threshold=midpoint.midpoint,
                            verification_source="assumption_midpoint_derivation",
                            derivation_trace=f"({n1} + {n2}) / 2",
                            provenance=provenance,
                        )
                    )
                    clashes.append(
                        EpistemicClash(
                            clash_id=f"assump_{key}",
                            clash_type="ASSUMPTION_CONFLICT",
                            clash_regime=GroundingRegime.EMPIRICAL,
                            stance_id_left=s1.stance_id,
                            stance_id_right=s2.stance_id,
                            explanation=f"Divergent parameter assumptions on '{key}': {n1} vs {n2}",
                            provenance=provenance,
                        )
                    )
            elif v1 != v2:
                spreads.append(1.0)
                clashes.append(
                    EpistemicClash(
                        clash_id=f"assump_{key}",
                        clash_type="ASSUMPTION_CONFLICT",
                        clash_regime=GroundingRegime.EMPIRICAL,
                        stance_id_left=s1.stance_id,
                        stance_id_right=s2.stance_id,
                        explanation=f"Divergent non-numeric assumptions on '{key}': {v1!r} vs {v2!r}",
                        provenance=provenance,
                    )
                )
            else:
                spreads.append(0.0)

        midpoint = midpoints[0] if midpoints else None
        pivot = pivots[0] if pivots else None
        midpoint_tuple = tuple(midpoints)
        pivot_tuple = tuple(pivots)
        if not spreads:
            return 0.0, None, None, (), ()
        return sum(spreads) / len(spreads), midpoint, pivot, midpoint_tuple, pivot_tuple

    def _eval_normative(
        self,
        s1: EpistemicStance,
        s2: EpistemicStance,
        clashes: List[EpistemicClash],
        diversities: List[PerspectiveDiversity],
        provenance: ProvenanceRef,
    ) -> float:
        n1 = {item.framework_id: item for item in s1.normative_commitments}
        n2 = {item.framework_id: item for item in s2.normative_commitments}
        if not n1 or not n2:
            return 0.0

        def deontic_keys(commitments, field: str) -> set[tuple[str, Polarity]]:
            keys = set()
            for commitment in commitments.values():
                for claim in getattr(commitment, field):
                    keys.add((claim.domain_key, claim.polarity))
            return keys

        prohibit_1 = deontic_keys(n1, "prohibited_claims")
        obligate_2 = deontic_keys(n2, "obligatory_claims")
        prohibit_2 = deontic_keys(n2, "prohibited_claims")
        obligate_1 = deontic_keys(n1, "obligatory_claims")

        # Same (domain_key, polarity) prohibited on one side and obligatory on the other.
        left_onto_right = prohibit_1.intersection(obligate_2)
        right_onto_left = prohibit_2.intersection(obligate_1)

        if left_onto_right or right_onto_left:
            rank_left = min(item.priority_rank for item in n1.values())
            rank_right = min(item.priority_rank for item in n2.values())
            clashes.append(
                EpistemicClash(
                    clash_id="norm_conflict",
                    clash_type="NORMATIVE_CONFLICT",
                    clash_regime=GroundingRegime.NORMATIVE,
                    stance_id_left=s1.stance_id,
                    stance_id_right=s2.stance_id,
                    explanation=(
                        "Symmetric normative conflict: frameworks declare mutually exclusive mandates "
                        "on the same deontic target and polarity. "
                        f"1 forbids what 2 requires: {left_onto_right}. "
                        f"2 forbids what 1 requires: {right_onto_left}. "
                        f"Priority ranks (left min={rank_left}, right min={rank_right}) are recorded "
                        "and do not dissolve the conflict."
                    ),
                    provenance=provenance,
                )
            )
            return 1.0

        if set(n1) != set(n2):
            diversities.append(
                PerspectiveDiversity(
                    diversity_id=artifact_id("div_norm", s1.stance_id, s2.stance_id),
                    diversity_type="NORMATIVE_FRAMEWORK_DIVERSITY",
                    stance_id_left=s1.stance_id,
                    stance_id_right=s2.stance_id,
                    explanation=f"Distinct normative frameworks ({set(n1.keys())} vs {set(n2.keys())}) without conflicting deontic imperatives.",
                    provenance=provenance,
                )
            )
        return 0.0

    def _build_tension_profile(
        self,
        d_pred: float,
        d_warr: float,
        d_assump: float,
        d_norm: float,
        d_causal: float,
        raw_aggregate: float,
        p1: GroundingProfile,
        p2: GroundingProfile,
        provenance: ProvenanceRef,
    ) -> TensionProfile:
        _ = d_norm
        empirical = _calibrate(d_pred, p1, p2, GroundingRegime.EMPIRICAL)
        formal = _calibrate(d_warr, p1, p2, GroundingRegime.FORMAL)
        causal = _calibrate(d_causal, p1, p2, GroundingRegime.CAUSAL)
        assumption = _calibrate(d_assump, p1, p2, GroundingRegime.EMPIRICAL)
        if assumption is None:
            assumption = _calibrate(d_assump, p1, p2, GroundingRegime.CAUSAL)
        return TensionProfile(
            empirical=empirical,
            formal=formal,
            causal=causal,
            assumption=assumption,
            normative=None,
            aggregate_structural_tension=raw_aggregate,
            provenance=provenance,
        )

    def _derive_resolution_status(
        self,
        raw_delta: float,
        clashes: List[EpistemicClash],
        p1: GroundingProfile,
        p2: GroundingProfile,
        has_midpoint: bool,
    ) -> ResolutionStatus:
        if raw_delta < self.tau_clash:
            return ResolutionStatus.NO_MATERIAL_DISAGREEMENT
        types = {item.clash_type for item in clashes}
        if "DEFINITIONAL_CONFLICT" in types:
            return ResolutionStatus.DEFINITIONALLY_IRREDUCIBLE
        if "NORMATIVE_CONFLICT" in types:
            return ResolutionStatus.NORMATIVELY_IRREDUCIBLE
        if "INFERENCE_RULE_CONFLICT" in types:
            return ResolutionStatus.AXIOMATICALLY_IRREDUCIBLE
        if has_midpoint:
            return ResolutionStatus.CONDITIONALLY_RESOLVABLE
        if self._inaccessible(clashes, p1, p2):
            return ResolutionStatus.INSUFFICIENT_EPISTEMIC_ACCESS
        if self._missing_scores(clashes, p1, p2):
            return ResolutionStatus.CURRENTLY_UNRESOLVED
        if self._evidence_resolves(clashes, p1, p2):
            return ResolutionStatus.RESOLVED_BY_EVIDENCE
        return ResolutionStatus.CURRENTLY_UNRESOLVED

    def _missing_scores(
        self,
        clashes: List[EpistemicClash],
        p1: GroundingProfile,
        p2: GroundingProfile,
    ) -> bool:
        for clash in clashes:
            left = p1.get_assessment(clash.clash_regime)
            right = p2.get_assessment(clash.clash_regime)
            if left is None or right is None or left.score is None or right.score is None:
                return True
        return False

    def _inaccessible(
        self,
        clashes: List[EpistemicClash],
        p1: GroundingProfile,
        p2: GroundingProfile,
    ) -> bool:
        inaccessible = {
            GroundingRegime.UNVERIFIABLE,
            GroundingRegime.INTERPRETIVE,
            GroundingRegime.PHENOMENOLOGICAL,
        }
        for clash in clashes:
            if clash.clash_regime in inaccessible:
                return True
            for profile in (p1, p2):
                assessment = profile.get_assessment(clash.clash_regime)
                if assessment and assessment.not_groundable_count > 0 and assessment.score is None:
                    return True
        return False

    def _evidence_resolves(
        self,
        clashes: List[EpistemicClash],
        p1: GroundingProfile,
        p2: GroundingProfile,
    ) -> bool:
        factual = [item for item in clashes if item.clash_type == "FACTUAL_CONFLICT"]
        if not factual:
            return False
        for clash in factual:
            left = p1.get_assessment(clash.clash_regime)
            right = p2.get_assessment(clash.clash_regime)
            if left is None or right is None or left.score is None or right.score is None:
                return False
            asymmetric = (
                left.supported_count > 0
                and right.contradicted_count > 0
                or right.supported_count > 0
                and left.contradicted_count > 0
            )
            if not asymmetric:
                return False
        return True

    @staticmethod
    def _mean_or_none(values: List[Optional[float]]) -> Optional[float]:
        present = [float(item) for item in values if item is not None]
        if not present:
            return None
        return float(np.mean(present))


def _calibrate(
    divergence: float,
    p1: GroundingProfile,
    p2: GroundingProfile,
    regime: GroundingRegime,
) -> Optional[float]:
    left = p1.get_assessment(regime)
    right = p2.get_assessment(regime)
    if left is None or right is None or left.score is None or right.score is None:
        return None
    return float(divergence * np.sqrt(left.score * right.score))


def _as_float(value: Union[float, str, bool]) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None
