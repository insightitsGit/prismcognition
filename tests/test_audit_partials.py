from __future__ import annotations

import pytest

from prismcognition.adapters.deterministic import HashEmbedder, ScriptedChaosLLM, UniformClassifier
from prismcognition.engine.catalog import methods_for
from prismcognition.engine.clusters import DeterministicMethodEvaluator, default_cluster_registry
from prismcognition.engine.derive import collect_pair_artifacts
from prismcognition.engine.orchestrator import EpistemicOrchestrator
from prismcognition.engine.router import DepthAllocatingRouter
from prismcognition.engine.tension import HardenedTensionEngine
from prismcognition.schemas.core import (
    Assumption,
    EpistemicStance,
    ExecutionTier,
    GroundingProfile,
    GroundingRegime,
    Polarity,
    ProvenanceRef,
    RegimeAssessment,
    ResolutionStatus,
    StructuredClaim,
    Warrant,
)
from prismcognition.subtractive.containment import ChaosRoomSandbox

PROV = ProvenanceRef(
    artifact_id="prov_audit",
    model_provider="pytest",
    model_id="audit",
    prompt_hash="0x0",
    created_at="2026-09-09T00:00:00Z",
)


def _claim(claim_id: str, domain: str, polarity: Polarity) -> StructuredClaim:
    subject, predicate = domain.split(".", 1)
    return StructuredClaim(
        claim_id=claim_id,
        subject=subject,
        predicate=predicate,
        target_object="True",
        polarity=polarity,
        domain_key=domain,
        raw_statement="",
        provenance=PROV,
    )


def _stance(stance_id: str, cluster: str, conclusion: StructuredClaim, **kwargs) -> EpistemicStance:
    return EpistemicStance(
        stance_id=stance_id,
        cluster_id=cluster,
        method_id=f"{cluster}.1",
        conclusion=conclusion,
        propositions=kwargs.get("propositions", (conclusion,)),
        warrants=kwargs.get("warrants", ()),
        assumptions=kwargs.get("assumptions", ()),
        normative_commitments=kwargs.get("normative_commitments", ()),
        internal_confidence=1.0,
        provenance=PROV,
    )


def test_grounding_and_tension_artifacts_carry_provenance():
    engine = HardenedTensionEngine()
    c1 = _claim("c1", "X.Y", Polarity.POSITIVE)
    c2 = _claim("c2", "X.Y", Polarity.NEGATIVE)
    s1 = _stance("s1", "1", c1)
    s2 = _stance("s2", "2", c2)
    profile = GroundingProfile(regime_assessments=())
    assert profile.provenance.artifact_id
    evaluation = engine.evaluate_pair(s1, s2, profile, profile, {"c1": c1, "c2": c2})
    assert evaluation.profile.provenance.parent_ids == (s1.provenance.artifact_id, s2.provenance.artifact_id)


def test_assumption_pivot_is_returned_not_rebuilt():
    engine = HardenedTensionEngine(assumption_spread=0.25)
    c1 = _claim("c1", "X.Y", Polarity.POSITIVE)
    c2 = _claim("c2", "X.Y", Polarity.POSITIVE)
    a1 = Assumption(assumption_id="a1", parameter_name="cost", assumed_value=0.2, provenance=PROV)
    a2 = Assumption(assumption_id="a2", parameter_name="cost", assumed_value=0.9, provenance=PROV)
    s1 = _stance("s1", "3", c1, assumptions=(a1,))
    s2 = _stance("s2", "5", c2, assumptions=(a2,))
    evaluation = engine.evaluate_pair(s1, s2, GroundingProfile(regime_assessments=()), GroundingProfile(regime_assessments=()), {})
    assert evaluation.midpoint is not None
    assert evaluation.pivot is not None
    assert evaluation.pivot.pivot_threshold == evaluation.midpoint.midpoint
    assert evaluation.pivot.provenance.artifact_id == evaluation.midpoint.provenance.artifact_id

    disagreements, _, _, _, _ = collect_pair_artifacts(
        [s1, s2],
        {"s1": GroundingProfile(regime_assessments=()), "s2": GroundingProfile(regime_assessments=())},
        {},
        engine,
        0.01,
    )
    assert disagreements
    assert disagreements[0].validated_pivot is not None
    assert disagreements[0].validated_pivot.pivot_threshold == evaluation.pivot.pivot_threshold
    assert disagreements[0].provenance.parent_ids[-1] == disagreements[0].clash.provenance.artifact_id


def test_definitional_conflict_is_definitionally_irreducible():
    engine = HardenedTensionEngine()
    premise = _claim("p1", "Term.defined", Polarity.POSITIVE)
    c_pos = _claim("c1", "Term.means", Polarity.POSITIVE)
    c_neg = _claim("c2", "Term.means", Polarity.NEGATIVE)
    w1 = Warrant(
        warrant_id="w1",
        epistemic_regime=GroundingRegime.DEFINITIONAL,
        rule_statement="DefTerm",
        premise_claim_ids=("p1",),
        conclusion_claim_id="c1",
        declared_confidence=1.0,
        provenance=PROV,
    )
    w2 = Warrant(
        warrant_id="w2",
        epistemic_regime=GroundingRegime.DEFINITIONAL,
        rule_statement="DefTerm",
        premise_claim_ids=("p1",),
        conclusion_claim_id="c2",
        declared_confidence=1.0,
        provenance=PROV,
    )
    fact_pos = _claim("f1", "X.Y", Polarity.POSITIVE)
    fact_neg = _claim("f2", "X.Y", Polarity.NEGATIVE)
    s1 = _stance("s1", "1", c_pos, propositions=(fact_pos,), warrants=(w1,))
    s2 = _stance("s2", "2", c_neg, propositions=(fact_neg,), warrants=(w2,))
    universe = {"p1": premise, "c1": c_pos, "c2": c_neg, "f1": fact_pos, "f2": fact_neg}
    evaluation = engine.evaluate_pair(s1, s2, GroundingProfile(regime_assessments=()), GroundingProfile(regime_assessments=()), universe)
    assert any(clash.clash_type == "DEFINITIONAL_CONFLICT" for clash in evaluation.clashes)
    assert evaluation.status == ResolutionStatus.DEFINITIONALLY_IRREDUCIBLE


def test_causal_conflict_is_emitted():
    engine = HardenedTensionEngine()
    premise = _claim("p1", "Cause.present", Polarity.POSITIVE)
    c_pos = _claim("c1", "Effect.follows", Polarity.POSITIVE)
    c_neg = _claim("c2", "Effect.follows", Polarity.NEGATIVE)
    w1 = Warrant(
        warrant_id="w1",
        epistemic_regime=GroundingRegime.CAUSAL,
        rule_statement="Mech",
        premise_claim_ids=("p1",),
        conclusion_claim_id="c1",
        declared_confidence=1.0,
        provenance=PROV,
    )
    w2 = Warrant(
        warrant_id="w2",
        epistemic_regime=GroundingRegime.CAUSAL,
        rule_statement="Mech",
        premise_claim_ids=("p1",),
        conclusion_claim_id="c2",
        declared_confidence=1.0,
        provenance=PROV,
    )
    s1 = _stance("s1", "3", c_pos, propositions=(premise, c_pos), warrants=(w1,))
    s2 = _stance("s2", "3", c_neg, propositions=(premise, c_neg), warrants=(w2,))
    evaluation = engine.evaluate_pair(
        s1, s2, GroundingProfile(regime_assessments=()), GroundingProfile(regime_assessments=()),
        {"p1": premise, "c1": c_pos, "c2": c_neg},
    )
    assert any(clash.clash_type == "CAUSAL_CONFLICT" for clash in evaluation.clashes)


def test_resolved_by_evidence_requires_asymmetric_grounding():
    engine = HardenedTensionEngine()
    c1 = _claim("c1", "X.Y", Polarity.POSITIVE)
    c2 = _claim("c2", "X.Y", Polarity.NEGATIVE)
    s1 = _stance("s1", "2", c1)
    s2 = _stance("s2", "4", c2)
    left = GroundingProfile(
        regime_assessments=(RegimeAssessment(regime=GroundingRegime.EMPIRICAL, score=0.9, supported_count=1),)
    )
    right = GroundingProfile(
        regime_assessments=(RegimeAssessment(regime=GroundingRegime.EMPIRICAL, score=0.1, contradicted_count=1),)
    )
    evaluation = engine.evaluate_pair(s1, s2, left, right, {"c1": c1, "c2": c2})
    assert evaluation.status == ResolutionStatus.RESOLVED_BY_EVIDENCE
    active, resolved, _, _, _ = collect_pair_artifacts(
        [s1, s2], {"s1": left, "s2": right}, {"c1": c1, "c2": c2}, engine, 0.35
    )
    assert not active
    assert resolved and resolved[0].resolution_status == ResolutionStatus.RESOLVED_BY_EVIDENCE


def test_inaccessible_regime_is_insufficient_epistemic_access():
    engine = HardenedTensionEngine()
    c1 = _claim("c1", "X.Y", Polarity.POSITIVE)
    c2 = _claim("c2", "X.Y", Polarity.NEGATIVE)
    s1 = _stance("s1", "2", c1)
    s2 = _stance("s2", "4", c2)
    opaque = GroundingProfile(
        regime_assessments=(
            RegimeAssessment(regime=GroundingRegime.EMPIRICAL, score=None, not_groundable_count=1),
        )
    )
    evaluation = engine.evaluate_pair(s1, s2, opaque, opaque, {"c1": c1, "c2": c2})
    assert evaluation.status == ResolutionStatus.INSUFFICIENT_EPISTEMIC_ACCESS


def test_orchestrator_applies_router_thresholds_from_config():
    router = DepthAllocatingRouter(UniformClassifier(0.50), required_threshold=0.40, high_risk_floor=0.50)
    orchestrator = EpistemicOrchestrator(
        router=router,
        chaos_room=ChaosRoomSandbox(ScriptedChaosLLM(), HashEmbedder()),
        cluster_registry=default_cluster_registry(),
        threshold_config={"router_required": 0.91, "high_risk_floor": 0.82},
        clash_threshold=0.35,
    )
    assert orchestrator.router.required_threshold == 0.91
    assert orchestrator.router.high_risk_floor == 0.82


@pytest.mark.asyncio
async def test_probe_stances_keep_premise_propositions():
    spec = methods_for("2", ExecutionTier.PROBE)[0]
    stance = await DeterministicMethodEvaluator(spec).evaluate("measure rainfall", depth=ExecutionTier.PROBE)
    domains = {item.domain_key for item in stance.propositions}
    assert "Rainfall.observed" in domains
    assert any(key.startswith("Inquiry.is_well_posed.") for key in domains)
    assert stance.warrants[0].premise_claim_ids[0] in {item.claim_id for item in stance.propositions}
