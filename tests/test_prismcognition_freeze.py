import pytest
import numpy as np
from typing import Dict

from prismcognition.schemas.core import (
    EpistemicStance,
    StructuredClaim,
    Warrant,
    Polarity,
    GroundingRegime,
    NormativeCommitment,
    ProvenanceRef,
    RuinBoundary,
    RuinAnalysisResult,
    RuinAnalysisStatus,
    GroundingProfile,
    RegimeAssessment,
    ResolutionStatus,
    WarrantGrounding,
    GroundingStatus,
    GeometricRuinStatus,
)
from prismcognition.subtractive.gate import SubtractiveGate
from prismcognition.engine.tension import HardenedTensionEngine
from prismcognition.engine.router import DepthAllocatingRouter
from prismcognition.engine.inference import InferenceConflictEvaluator
from prismcognition.replay.engine import ReplayEngine, FrozenDeliberationBundle
from prismcognition.schemas.core import ExecutionTier

PROV = ProvenanceRef(
    artifact_id="prov_test",
    model_provider="pytest",
    model_id="unit_test",
    prompt_hash="0x0",
    created_at="2026-09-09T00:00:00Z",
)


def test_svd_subspace_robustness():
    gate = SubtractiveGate(tol=1e-5)
    claim = StructuredClaim(
        claim_id="c",
        subject="A",
        predicate="b",
        target_object="c",
        polarity=Polarity.POSITIVE,
        domain_key="A.b",
        raw_statement="",
        provenance=PROV,
    )
    b1 = [1.0, 0.0, 0.0]
    b2 = [1.0, 0.0, 0.0]
    b3 = [0.0, 0.0, 0.0]
    boundaries = [
        RuinBoundary(
            boundary_id=f"r{i}",
            failure_state=claim,
            trigger_predicates=(),
            reversibility=0.0,
            severity=1.0,
            semantic_vector=tuple(v),
            provenance=PROV,
        )
        for i, v in enumerate([b1, b2, b3])
    ]
    ruin_res = RuinAnalysisResult(
        status=RuinAnalysisStatus.COMPLETED,
        boundaries=tuple(boundaries),
        diagnostics=(),
        execution_time_ms=1.0,
        provenance=PROV,
    )
    z = np.array([1.0, 1.0, 0.0])
    st = EpistemicStance(
        stance_id="s1",
        cluster_id="1",
        method_id="1.1",
        conclusion=claim,
        propositions=(),
        warrants=(),
        assumptions=(),
        normative_commitments=(),
        internal_confidence=1.0,
        semantic_vector=tuple(z.tolist()),
        provenance=PROV,
    )
    surviving, _ = gate.apply_subtractive_gate([(st, z)], ruin_res)
    assert len(surviving) == 1
    safe_v = surviving[0][1]
    assert abs(np.dot(safe_v, [1.0, 0.0, 0.0])) < 1e-5


def test_vanishing_energy_preserves_logical_stance():
    gate = SubtractiveGate(tol=1e-5)
    claim = StructuredClaim(
        claim_id="c",
        subject="A",
        predicate="b",
        target_object="c",
        polarity=Polarity.POSITIVE,
        domain_key="A.b",
        raw_statement="",
        provenance=PROV,
    )
    boundary = RuinBoundary(
        boundary_id="r1",
        failure_state=claim,
        trigger_predicates=(),
        reversibility=0.0,
        severity=1.0,
        semantic_vector=(1.0, 0.0, 0.0),
        provenance=PROV,
    )
    ruin_res = RuinAnalysisResult(
        status=RuinAnalysisStatus.COMPLETED,
        boundaries=(boundary,),
        diagnostics=(),
        execution_time_ms=1.0,
        provenance=PROV,
    )
    z = np.array([1.0, 0.0, 0.0])
    st = EpistemicStance(
        stance_id="s_vanish",
        cluster_id="1",
        method_id="1.1",
        conclusion=claim,
        propositions=(),
        warrants=(),
        assumptions=(),
        normative_commitments=(),
        internal_confidence=1.0,
        semantic_vector=tuple(z.tolist()),
        provenance=PROV,
    )
    surviving, _ = gate.apply_subtractive_gate([(st, z)], ruin_res)
    assert len(surviving) == 1
    assert surviving[0][0].ruin_geometry_status == GeometricRuinStatus.GEOMETRICALLY_COLLAPSED
    assert surviving[0][1] is None
    assert surviving[0][0].propositions == st.propositions
    assert surviving[0][0].conclusion.claim_id == "c"
    assert st.provenance.artifact_id in surviving[0][0].provenance.parent_ids


def test_inference_clash_with_compatible_premises():
    engine = HardenedTensionEngine(clash_threshold=0.35)
    premise = StructuredClaim(
        claim_id="p1",
        subject="Load",
        predicate="exceeds",
        target_object="Limit",
        polarity=Polarity.POSITIVE,
        domain_key="Load.exceeds",
        raw_statement="",
        provenance=PROV,
    )
    c_pos = StructuredClaim(
        claim_id="c1",
        subject="System",
        predicate="fails",
        target_object="True",
        polarity=Polarity.POSITIVE,
        domain_key="System.fails",
        raw_statement="",
        provenance=PROV,
    )
    c_neg = StructuredClaim(
        claim_id="c2",
        subject="System",
        predicate="fails",
        target_object="True",
        polarity=Polarity.NEGATIVE,
        domain_key="System.fails",
        raw_statement="",
        provenance=PROV,
    )
    claims_map = {"p1": premise, "c1": c_pos, "c2": c_neg}
    w1 = Warrant(
        warrant_id="w1",
        epistemic_regime=GroundingRegime.FORMAL,
        rule_statement="RuleOverload",
        premise_claim_ids=("p1",),
        conclusion_claim_id="c1",
        declared_confidence=1.0,
        provenance=PROV,
    )
    w2 = Warrant(
        warrant_id="w2",
        epistemic_regime=GroundingRegime.FORMAL,
        rule_statement="RuleOverload",
        premise_claim_ids=("p1",),
        conclusion_claim_id="c2",
        declared_confidence=1.0,
        provenance=PROV,
    )
    s1 = EpistemicStance(
        stance_id="s1",
        cluster_id="1",
        method_id="1.1",
        conclusion=c_pos,
        propositions=(premise,),
        warrants=(w1,),
        assumptions=(),
        normative_commitments=(),
        internal_confidence=1.0,
        provenance=PROV,
    )
    s2 = EpistemicStance(
        stance_id="s2",
        cluster_id="2",
        method_id="2.1",
        conclusion=c_neg,
        propositions=(premise,),
        warrants=(w2,),
        assumptions=(),
        normative_commitments=(),
        internal_confidence=1.0,
        provenance=PROV,
    )
    profile = GroundingProfile(regime_assessments=())
    evaluation = engine.evaluate_pair(s1, s2, profile, profile, claims_map)
    clashes = evaluation.clashes
    raw_d = evaluation.raw_delta
    assert any(clash.clash_type == "INFERENCE_RULE_CONFLICT" for clash in clashes)
    assert InferenceConflictEvaluator.evaluate_inference_clash(w1, w2, claims_map) is True
    assert raw_d >= 0.0


def test_incompatible_premises_do_not_clash():
    engine = HardenedTensionEngine(clash_threshold=0.35)
    p_pos = StructuredClaim(
        claim_id="p_pos",
        subject="Load",
        predicate="exceeds",
        target_object="Limit",
        polarity=Polarity.POSITIVE,
        domain_key="Load.exceeds",
        raw_statement="",
        provenance=PROV,
    )
    p_neg = StructuredClaim(
        claim_id="p_neg",
        subject="Load",
        predicate="exceeds",
        target_object="Limit",
        polarity=Polarity.NEGATIVE,
        domain_key="Load.exceeds",
        raw_statement="",
        provenance=PROV,
    )
    c_pos = StructuredClaim(
        claim_id="c1",
        subject="System",
        predicate="fails",
        target_object="True",
        polarity=Polarity.POSITIVE,
        domain_key="System.fails",
        raw_statement="",
        provenance=PROV,
    )
    c_neg = StructuredClaim(
        claim_id="c2",
        subject="System",
        predicate="fails",
        target_object="True",
        polarity=Polarity.NEGATIVE,
        domain_key="System.fails",
        raw_statement="",
        provenance=PROV,
    )
    universe = {"p_pos": p_pos, "p_neg": p_neg, "c1": c_pos, "c2": c_neg}
    w1 = Warrant(
        warrant_id="w1",
        epistemic_regime=GroundingRegime.FORMAL,
        rule_statement="RuleOverload",
        premise_claim_ids=("p_pos",),
        conclusion_claim_id="c1",
        declared_confidence=1.0,
        provenance=PROV,
    )
    w2 = Warrant(
        warrant_id="w2",
        epistemic_regime=GroundingRegime.FORMAL,
        rule_statement="RuleOverload",
        premise_claim_ids=("p_neg",),
        conclusion_claim_id="c2",
        declared_confidence=1.0,
        provenance=PROV,
    )
    s1 = EpistemicStance(
        stance_id="s1",
        cluster_id="1",
        method_id="1.1",
        conclusion=c_pos,
        propositions=(p_pos,),
        warrants=(w1,),
        assumptions=(),
        normative_commitments=(),
        internal_confidence=1.0,
        provenance=PROV,
    )
    s2 = EpistemicStance(
        stance_id="s2",
        cluster_id="2",
        method_id="2.1",
        conclusion=c_neg,
        propositions=(p_neg,),
        warrants=(w2,),
        assumptions=(),
        normative_commitments=(),
        internal_confidence=1.0,
        provenance=PROV,
    )
    profile = GroundingProfile(regime_assessments=())
    evaluation = engine.evaluate_pair(s1, s2, profile, profile, universe)
    clashes = evaluation.clashes
    assert not any(clash.clash_type == "INFERENCE_RULE_CONFLICT" for clash in clashes)


def test_tension_profile_preserves_none():
    engine = HardenedTensionEngine(clash_threshold=0.35)
    c1 = StructuredClaim(
        claim_id="c1",
        subject="X",
        predicate="Y",
        target_object="Z",
        polarity=Polarity.POSITIVE,
        domain_key="X.Y",
        raw_statement="",
        provenance=PROV,
    )
    c2 = StructuredClaim(
        claim_id="c2",
        subject="X",
        predicate="Y",
        target_object="Z",
        polarity=Polarity.NEGATIVE,
        domain_key="X.Y",
        raw_statement="",
        provenance=PROV,
    )
    s1 = EpistemicStance(
        stance_id="s1",
        cluster_id="1",
        method_id="1.1",
        conclusion=c1,
        propositions=(c1,),
        warrants=(),
        assumptions=(),
        normative_commitments=(),
        internal_confidence=1.0,
        provenance=PROV,
    )
    s2 = EpistemicStance(
        stance_id="s2",
        cluster_id="2",
        method_id="2.1",
        conclusion=c2,
        propositions=(c2,),
        warrants=(),
        assumptions=(),
        normative_commitments=(),
        internal_confidence=1.0,
        provenance=PROV,
    )
    unres = GroundingProfile(
        regime_assessments=(RegimeAssessment(regime=GroundingRegime.EMPIRICAL, score=None, unresolved_count=1),)
    )
    evaluation = engine.evaluate_pair(s1, s2, unres, unres, {"c1": c1, "c2": c2})
    assert evaluation.raw_delta == 0.40
    assert evaluation.calibrated is None
    assert evaluation.profile.empirical is None
    assert evaluation.status == ResolutionStatus.CURRENTLY_UNRESOLVED
    assert evaluation.profile.provenance.artifact_id


def test_normative_conflict_symmetry():
    engine = HardenedTensionEngine(clash_threshold=0.35)
    claim = StructuredClaim(
        claim_id="c",
        subject="Action",
        predicate="taken",
        target_object="True",
        polarity=Polarity.POSITIVE,
        domain_key="Action.taken",
        raw_statement="",
        provenance=PROV,
    )
    n1 = NormativeCommitment(
        axiom_name="A1",
        framework_id="FW1",
        priority_rank=1,
        prohibited_claims=(claim,),
        provenance=PROV,
    )
    n2 = NormativeCommitment(
        axiom_name="A2",
        framework_id="FW2",
        priority_rank=1,
        obligatory_claims=(claim,),
        provenance=PROV,
    )
    s1 = EpistemicStance(
        stance_id="s1",
        cluster_id="6",
        method_id="6.1",
        conclusion=claim,
        propositions=(),
        warrants=(),
        assumptions=(),
        normative_commitments=(n1,),
        internal_confidence=1.0,
        provenance=PROV,
    )
    s2 = EpistemicStance(
        stance_id="s2",
        cluster_id="6",
        method_id="6.2",
        conclusion=claim,
        propositions=(),
        warrants=(),
        assumptions=(),
        normative_commitments=(n2,),
        internal_confidence=1.0,
        provenance=PROV,
    )
    profile = GroundingProfile(regime_assessments=())
    clashes_12 = engine.evaluate_pair(s1, s2, profile, profile, {}).clashes
    clashes_21 = engine.evaluate_pair(s2, s1, profile, profile, {}).clashes
    assert any(clash.clash_type == "NORMATIVE_CONFLICT" for clash in clashes_12)
    assert any(clash.clash_type == "NORMATIVE_CONFLICT" for clash in clashes_21)


@pytest.mark.asyncio
async def test_router_depth_allocation():
    class MockClassifier:
        async def score_clusters(self, query: str) -> Dict[int, float]:
            return {1: 0.05, 2: 0.05, 3: 0.05, 4: 0.05, 5: 0.05, 6: 0.05}

    router = DepthAllocatingRouter(classifier_client=MockClassifier())
    plan_probe = await router.route("General query", risk_level="LOW")
    routes = {route.cluster_id: route.tier for route in plan_probe.routes}
    assert routes[1] == ExecutionTier.PROBE
    assert routes[7] == ExecutionTier.REQUIRED
    assert ExecutionTier.SKIPPED not in {routes[1], routes[3], routes[4], routes[5]}

    plan_skipped = await router.route("Verify formal axiom definition purely a priori", risk_level="LOW")
    routes_skip = {route.cluster_id: route.tier for route in plan_skipped.routes}
    assert routes_skip[2] == ExecutionTier.SKIPPED


def test_deterministic_replay_hash_parity():
    claim = StructuredClaim(
        claim_id="c",
        subject="A",
        predicate="b",
        target_object="c",
        polarity=Polarity.POSITIVE,
        domain_key="A.b",
        raw_statement="",
        provenance=PROV,
    )
    stance = EpistemicStance(
        stance_id="s1",
        cluster_id="1",
        method_id="1.1",
        conclusion=claim,
        propositions=(claim,),
        warrants=(),
        assumptions=(),
        normative_commitments=(),
        internal_confidence=1.0,
        provenance=PROV,
    )
    ruin_res = RuinAnalysisResult(
        status=RuinAnalysisStatus.COMPLETED,
        boundaries=(),
        diagnostics=(),
        execution_time_ms=5.0,
        provenance=PROV,
    )
    bundle = FrozenDeliberationBundle(
        deliberation_id="delib_replay_001",
        inquiry_text="Test Inquiry",
        frozen_stances=(stance,),
        frozen_ruin_result=ruin_res,
        frozen_grounding_profiles={
            "s1": {"regime_assessments": [{"regime": "EMPIRICAL", "score": 0.9, "applicable": True}]}
        },
        methods_executed=("1_1.1",),
        methods_probed=(),
        methods_skipped=(),
        methods_failed=(),
        evidence_needed=(),
        irreducible_tensions=(),
        optional_recommendation=None,
        threshold_config={"structural_clash": 0.35, "subtractive_rank_tol": 1e-6},
        provenance=PROV,
    )
    art1 = ReplayEngine.replay(bundle)
    art2 = ReplayEngine.replay(bundle)
    assert ReplayEngine.canonical_hash(art1) == ReplayEngine.canonical_hash(art2)
    assert art1.strongly_supported_claims[0].claim_id == "c"


def test_warrant_grounding_preserves_none():
    grounding = WarrantGrounding(
        warrant_id="w_test",
        epistemic_regime=GroundingRegime.NORMATIVE,
        status=GroundingStatus.NOT_GROUNDABLE,
        support_score=None,
        epistemic_reasoning="Axiomatic moral imperative; non-empirical.",
        provenance=PROV,
    )
    assert grounding.support_score is None
    assert grounding.status == GroundingStatus.NOT_GROUNDABLE
