from __future__ import annotations

import pytest

from prismcognition.adapters.deterministic import HashEmbedder, ScriptedChaosLLM, UniformClassifier
from prismcognition.engine.catalog import METHOD_CATALOG
from prismcognition.engine.clusters import default_cluster_registry
from prismcognition.engine.coverage import compile_coverage
from prismcognition.engine.derive import assumptions_that_matter, collect_pair_artifacts
from prismcognition.engine.extractor import StructuredClusterExtractor
from prismcognition.engine.inquiry import extract_inquiry_features
from prismcognition.engine.orchestrator import EpistemicOrchestrator
from prismcognition.engine.router import DepthAllocatingRouter
from prismcognition.engine.tension import HardenedTensionEngine
from prismcognition.evidence.store import EvidenceStore
from prismcognition.evidence.verifier import RegimeAwareGroundingVerifier
from prismcognition.render.artifact import render_deliberation
from prismcognition.replay.engine import ReplayEngine
from prismcognition.schemas.core import (
    AllocationScoreSource,
    Assumption,
    EpistemicStance,
    EvidenceRecord,
    ExecutionTier,
    GroundingProfile,
    GroundingRegime,
    GroundingStatus,
    NormativeCommitment,
    Polarity,
    ProvenanceRef,
    ResolutionStatus,
    StructuredClaim,
    Warrant,
    index_claims,
)
from prismcognition.subtractive.containment import ChaosRoomSandbox

PROV = ProvenanceRef(
    artifact_id="prov_gaps",
    model_provider="pytest",
    model_id="gaps",
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


def test_multi_parameter_midpoints_are_all_kept():
    engine = HardenedTensionEngine(clash_threshold=0.01, assumption_spread=0.25)
    c1 = _claim("c1", "X.Y", Polarity.POSITIVE)
    c2 = _claim("c2", "X.Y", Polarity.POSITIVE)
    s1 = _stance(
        "s1",
        "3",
        c1,
        assumptions=(
            Assumption(assumption_id="a1", parameter_name="cost", assumed_value=0.1, provenance=PROV),
            Assumption(assumption_id="a2", parameter_name="delay", assumed_value=0.2, provenance=PROV),
        ),
    )
    s2 = _stance(
        "s2",
        "5",
        c2,
        assumptions=(
            Assumption(assumption_id="b1", parameter_name="cost", assumed_value=0.9, provenance=PROV),
            Assumption(assumption_id="b2", parameter_name="delay", assumed_value=0.95, provenance=PROV),
        ),
    )
    evaluation = engine.evaluate_pair(
        s1, s2, GroundingProfile(regime_assessments=()), GroundingProfile(regime_assessments=()), {}
    )
    names = {item.variable_name for item in evaluation.midpoints}
    assert names == {"cost", "delay"}
    assert {item.variable_name for item in evaluation.pivots} == names

    disagreements, _, needed, _ = collect_pair_artifacts(
        [s1, s2],
        {"s1": GroundingProfile(regime_assessments=()), "s2": GroundingProfile(regime_assessments=())},
        {},
        engine,
        0.01,
    )
    assumption_dis = [item for item in disagreements if item.clash.clash_type == "ASSUMPTION_CONFLICT"]
    assert {item.midpoint.variable_name for item in assumption_dis if item.midpoint} == names
    assert any("cost" in item for item in needed)
    assert any("delay" in item for item in needed)
    mattering = assumptions_that_matter([s1, s2], disagreements)
    assert {item.parameter_name for item in mattering} == names


def test_normative_opposite_polarity_is_not_a_conflict():
    engine = HardenedTensionEngine()
    forbid_pos = _claim("n1", "Action.taken", Polarity.POSITIVE)
    require_neg = _claim("n2", "Action.taken", Polarity.NEGATIVE)
    s1 = _stance(
        "s1",
        "6",
        forbid_pos,
        normative_commitments=(
            NormativeCommitment(
                axiom_name="A1",
                framework_id="FW1",
                priority_rank=1,
                prohibited_claims=(forbid_pos,),
                provenance=PROV,
            ),
        ),
    )
    s2 = _stance(
        "s2",
        "6",
        require_neg,
        normative_commitments=(
            NormativeCommitment(
                axiom_name="A2",
                framework_id="FW2",
                priority_rank=2,
                obligatory_claims=(require_neg,),
                provenance=PROV,
            ),
        ),
    )
    evaluation = engine.evaluate_pair(
        s1, s2, GroundingProfile(regime_assessments=()), GroundingProfile(regime_assessments=()), {}
    )
    assert not any(item.clash_type == "NORMATIVE_CONFLICT" for item in evaluation.clashes)


def test_normative_same_polarity_conflict_records_priority_without_resolving():
    engine = HardenedTensionEngine()
    claim = _claim("n1", "Action.taken", Polarity.POSITIVE)
    s1 = _stance(
        "s1",
        "6",
        claim,
        propositions=(_claim("f1", "X.Y", Polarity.POSITIVE),),
        normative_commitments=(
            NormativeCommitment(
                axiom_name="A1",
                framework_id="FW1",
                priority_rank=1,
                prohibited_claims=(claim,),
                provenance=PROV,
            ),
        ),
    )
    s2 = _stance(
        "s2",
        "6",
        claim,
        propositions=(_claim("f2", "X.Y", Polarity.NEGATIVE),),
        normative_commitments=(
            NormativeCommitment(
                axiom_name="A2",
                framework_id="FW2",
                priority_rank=9,
                obligatory_claims=(claim,),
                provenance=PROV,
            ),
        ),
    )
    evaluation = engine.evaluate_pair(
        s1, s2, GroundingProfile(regime_assessments=()), GroundingProfile(regime_assessments=()), {}
    )
    clash = next(item for item in evaluation.clashes if item.clash_type == "NORMATIVE_CONFLICT")
    assert "left min=1" in clash.explanation
    assert "right min=9" in clash.explanation
    assert "do not dissolve" in clash.explanation
    assert evaluation.status == ResolutionStatus.NORMATIVELY_IRREDUCIBLE


@pytest.mark.asyncio
async def test_router_missing_score_is_compute_default_not_empty_cluster():
    class PartialClassifier:
        async def score_clusters(self, inquiry: str):
            _ = inquiry
            return {1: 0.91}

    router = DepthAllocatingRouter(PartialClassifier(), required_threshold=0.40)
    plan = await router.route("General query", risk_level="LOW")
    routes = {route.cluster_id: route for route in plan.routes}
    assert routes[1].score_source == AllocationScoreSource.CLASSIFIER
    assert routes[1].classifier_score == 0.91
    assert routes[2].score_source == AllocationScoreSource.MISSING_COMPUTE_DEFAULT
    assert routes[2].classifier_score is None
    assert routes[2].tier == ExecutionTier.PROBE
    assert "compute default" in routes[2].justification
    assert "empty-cluster" in routes[2].justification
    assert plan.inquiry_features is not None


@pytest.mark.asyncio
async def test_artifact_carries_coverage_route_plan_and_snapshot():
    store = EvidenceStore(snapshot_id="snap-gaps")
    orchestrator = EpistemicOrchestrator(
        router=DepthAllocatingRouter(UniformClassifier(0.05)),
        chaos_room=ChaosRoomSandbox(ScriptedChaosLLM(), HashEmbedder()),
        cluster_registry=default_cluster_registry(),
        grounding_verifier=RegimeAwareGroundingVerifier(store),
        clash_threshold=0.35,
    )
    artifact = await orchestrator.deliberate("Should we expand the plant this quarter?", risk_level="LOW")
    assert artifact.coverage is not None
    assert artifact.coverage.executed_method_ids == artifact.methods_executed
    assert artifact.coverage.probed_method_ids == artifact.methods_probed
    assert artifact.route_plan is not None
    assert artifact.provenance.evidence_snapshot_id == "snap-gaps"
    compiled = compile_coverage(artifact)
    assert compiled.coverage_index == artifact.coverage.coverage_index

    bundle = orchestrator.last_frozen_bundle()
    assert bundle.frozen_route_plan is not None
    replayed = ReplayEngine.replay(bundle)
    assert replayed.coverage is not None
    assert replayed.route_plan == bundle.frozen_route_plan
    assert replayed.coverage.coverage_index == artifact.coverage.coverage_index
    rendered = render_deliberation(artifact)
    assert "Coverage index:" in rendered


@pytest.mark.asyncio
async def test_extractor_indexes_premises_and_commitments():
    class Scripted:
        async def generate_json(self, messages, temperature=0.0):
            _ = (messages, temperature)
            return {
                "polarity": 1,
                "subject": "Inquiry",
                "predicate": "thesis_holds",
                "object": "True",
                "rule": "extracted_rule",
                "confidence": 0.61,
                "premise_predicate": "is_well_posed",
            }

    extractor = StructuredClusterExtractor(Scripted(), METHOD_CATALOG["6"][0])
    stance = await extractor.evaluate("Should we expand?", depth=ExecutionTier.PROBE)
    universe = index_claims([stance])
    assert stance.warrants[0].premise_claim_ids[0] in universe
    assert stance.warrants[0].conclusion_claim_id in universe
    assert stance.normative_commitments
    assert stance.semantic_vector is not None


def test_inquiry_ontology_covers_token_and_phrase_forms():
    formal = extract_inquiry_features("Prove the axiom by a closed-form proof")
    assert formal.formal_a_priori
    physical = extract_inquiry_features("Calculate the orbital period using Kepler")
    assert physical.physical_constant_query
    empirical = extract_inquiry_features("Run an experiment and measure the outcome")
    assert empirical.empirical_measurement
    general = extract_inquiry_features("General query")
    assert not general.formal_a_priori
    assert not general.physical_constant_query


@pytest.mark.asyncio
async def test_assumptions_that_do_not_clash_are_not_marked_as_mattering():
    orchestrator = EpistemicOrchestrator(
        router=DepthAllocatingRouter(UniformClassifier(0.9)),
        chaos_room=ChaosRoomSandbox(ScriptedChaosLLM(), HashEmbedder()),
        cluster_registry=default_cluster_registry(),
        clash_threshold=0.99,
    )
    artifact = await orchestrator.deliberate("measure rainfall totals", risk_level="LOW")
    assert artifact.assumptions_that_matter == ()


@pytest.mark.asyncio
async def test_grounding_provenance_carries_store_snapshot():
    store = EvidenceStore(snapshot_id="ledger-9")
    store.ingest(
        EvidenceRecord(
            record_id="e1",
            snapshot_id="ledger-9",
            domain_key="Inquiry.thesis_holds",
            polarity=Polarity.POSITIVE,
            regime=GroundingRegime.EMPIRICAL,
            status=GroundingStatus.SUPPORTED,
            support_score=0.8,
            source_ref="note://1",
        )
    )
    verifier = RegimeAwareGroundingVerifier(store)
    claim = _claim("c1", "Inquiry.thesis_holds", Polarity.POSITIVE)
    warrant = Warrant(
        warrant_id="w1",
        epistemic_regime=GroundingRegime.EMPIRICAL,
        rule_statement="r",
        premise_claim_ids=(),
        conclusion_claim_id="c1",
        declared_confidence=0.4,
        provenance=PROV,
    )
    grounding = await verifier.verify_warrant(warrant, {"c1": claim})
    assert grounding.provenance.evidence_snapshot_id == "ledger-9"
    assert grounding.status == GroundingStatus.SUPPORTED
