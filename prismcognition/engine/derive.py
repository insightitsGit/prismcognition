from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set, Tuple

from prismcognition.engine.tension import HardenedTensionEngine, PairEvaluation
from prismcognition.identity import artifact_id, derive_provenance
from prismcognition.schemas.core import (
    Assumption,
    AssumptionMidpoint,
    DisagreementArtifact,
    EpistemicClash,
    EpistemicPivot,
    EpistemicStance,
    GroundingProfile,
    GroundingRegime,
    GroundingStatus,
    PerspectiveDiversity,
    ResolutionStatus,
    StructuredClaim,
    WarrantGrounding,
)


def collect_pair_artifacts(
    stances: List[EpistemicStance],
    profiles: Dict[str, GroundingProfile],
    claims_universe: Dict[str, StructuredClaim],
    tension: HardenedTensionEngine,
    clash_threshold: float,
) -> Tuple[List[DisagreementArtifact], List[PerspectiveDiversity], Tuple[str, ...], Tuple[str, ...]]:
    disagreements: List[DisagreementArtifact] = []
    diversities: List[PerspectiveDiversity] = []
    evidence_needed: Set[str] = set()
    irreducible: List[str] = []

    for index, left in enumerate(stances):
        for right in stances[index + 1 :]:
            evaluation = tension.evaluate_pair(
                left,
                right,
                profiles[left.stance_id],
                profiles[right.stance_id],
                claims_universe,
            )
            diversities.extend(evaluation.diversities)
            if evaluation.raw_delta < clash_threshold:
                continue
            for clash in evaluation.clashes:
                midpoint = _midpoint_for_clash(clash, evaluation)
                pivot = _pivot_for_clash(clash, evaluation)
                disagreements.append(
                    DisagreementArtifact(
                        disagreement_id=f"dis_{left.stance_id}_{right.stance_id}_{clash.clash_id}",
                        clash=clash,
                        raw_tension=round(evaluation.raw_delta, 4),
                        calibrated_tension=(
                            round(evaluation.calibrated, 4) if evaluation.calibrated is not None else None
                        ),
                        tension_profile=evaluation.profile,
                        profile_left=profiles[left.stance_id],
                        profile_right=profiles[right.stance_id],
                        resolution_status=evaluation.status,
                        midpoint=midpoint,
                        validated_pivot=pivot,
                        provenance=derive_provenance(
                            clash.provenance,
                            artifact_id_value=artifact_id(
                                "dis", left.stance_id, right.stance_id, clash.clash_id
                            ),
                            model_provider="internal",
                            model_id="derive",
                            prompt_hash="deterministic-internal",
                        ),
                    )
                )
            if evaluation.status in (
                ResolutionStatus.NORMATIVELY_IRREDUCIBLE,
                ResolutionStatus.AXIOMATICALLY_IRREDUCIBLE,
                ResolutionStatus.DEFINITIONALLY_IRREDUCIBLE,
            ):
                irreducible.append(f"{left.cluster_id} vs {right.cluster_id}: {evaluation.status.value}")
            elif evaluation.status == ResolutionStatus.CONDITIONALLY_RESOLVABLE:
                for midpoint in evaluation.midpoints or ((evaluation.midpoint,) if evaluation.midpoint else ()):
                    evidence_needed.add(f"Parameter confirmation: {midpoint.variable_name} near {midpoint.midpoint}")
            elif evaluation.status == ResolutionStatus.CURRENTLY_UNRESOLVED:
                evidence_needed.add(f"Regime grounding for clash {left.cluster_id} vs {right.cluster_id}")

    return disagreements, diversities, tuple(sorted(evidence_needed)), tuple(irreducible)


def select_strongly_supported(
    stances: Iterable[EpistemicStance],
    profiles: Dict[str, GroundingProfile],
    warrant_groundings: Dict[str, Tuple[WarrantGrounding, ...]],
    threshold: float,
) -> List[StructuredClaim]:
    supported: List[StructuredClaim] = []
    seen: Set[str] = set()
    for stance in stances:
        groundings = warrant_groundings.get(stance.stance_id, ())
        if groundings:
            allowed = {
                item.warrant_id
                for item in groundings
                if item.status == GroundingStatus.SUPPORTED
                and item.support_score is not None
                and item.support_score >= threshold
            }
            for warrant in stance.warrants:
                if warrant.warrant_id not in allowed:
                    continue
                claim = _claim_for(stance, warrant.conclusion_claim_id)
                if claim is not None and claim.claim_id not in seen:
                    seen.add(claim.claim_id)
                    supported.append(claim)
            continue

        profile = profiles[stance.stance_id]
        empirical = profile.get_assessment(GroundingRegime.EMPIRICAL)
        formal = profile.get_assessment(GroundingRegime.FORMAL)
        empirical_ok = empirical is not None and empirical.score is not None and empirical.score >= threshold
        formal_ok = formal is not None and formal.score is not None and formal.score >= threshold
        if empirical_ok or formal_ok:
            if stance.conclusion.claim_id not in seen:
                seen.add(stance.conclusion.claim_id)
                supported.append(stance.conclusion)
    return supported


def assumptions_that_matter(
    stances: Iterable[EpistemicStance],
    disagreements: Iterable[DisagreementArtifact],
) -> Tuple[Assumption, ...]:
    names = {
        item.midpoint.variable_name
        for item in disagreements
        if item.midpoint is not None
    }
    names.update(
        item.clash.clash_id[len("assump_") :]
        for item in disagreements
        if item.clash.clash_type == "ASSUMPTION_CONFLICT" and item.clash.clash_id.startswith("assump_")
    )
    if not names:
        return ()
    selected: List[Assumption] = []
    seen: Set[str] = set()
    for stance in stances:
        for assumption in stance.assumptions:
            if assumption.parameter_name not in names:
                continue
            key = f"{assumption.assumption_id}:{assumption.parameter_name}"
            if key in seen:
                continue
            seen.add(key)
            selected.append(assumption)
    return tuple(selected)


def _midpoint_for_clash(clash: EpistemicClash, evaluation: PairEvaluation) -> Optional[AssumptionMidpoint]:
    if clash.clash_type != "ASSUMPTION_CONFLICT":
        return None
    key = clash.clash_id[len("assump_") :] if clash.clash_id.startswith("assump_") else ""
    for midpoint in evaluation.midpoints:
        if midpoint.variable_name == key:
            return midpoint
    return evaluation.midpoint


def _pivot_for_clash(clash: EpistemicClash, evaluation: PairEvaluation) -> Optional[EpistemicPivot]:
    if clash.clash_type != "ASSUMPTION_CONFLICT":
        return None
    key = clash.clash_id[len("assump_") :] if clash.clash_id.startswith("assump_") else ""
    for pivot in evaluation.pivots:
        if pivot.variable_name == key:
            return pivot
    return evaluation.pivot


def _claim_for(stance: EpistemicStance, claim_id: str) -> Optional[StructuredClaim]:
    if stance.conclusion.claim_id == claim_id:
        return stance.conclusion
    for proposition in stance.propositions:
        if proposition.claim_id == claim_id:
            return proposition
    return None
