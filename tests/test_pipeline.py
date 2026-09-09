import pytest

from prismcognition.adapters.deterministic import HashEmbedder, ScriptedChaosLLM, UniformClassifier
from prismcognition.engine.clusters import default_cluster_registry
from prismcognition.engine.coverage import compile_coverage
from prismcognition.engine.orchestrator import EpistemicOrchestrator
from prismcognition.engine.router import DepthAllocatingRouter
from prismcognition.evidence.rollup import GroundingRollupEngine
from prismcognition.evidence.verifier import RegimeAwareGroundingVerifier, StaticEvidenceAdapter
from prismcognition.replay.engine import ReplayEngine
from prismcognition.schemas.core import (
    GroundingRegime,
    GroundingStatus,
    Polarity,
    ProvenanceRef,
    RuinAnalysisStatus,
    RuinBoundary,
    RuinAnalysisResult,
    StructuredClaim,
    Warrant,
    WarrantGrounding,
)
from prismcognition.subtractive.containment import ChaosRoomSandbox
from prismcognition.subtractive.gate import SubtractiveGate

PROV = ProvenanceRef(
    artifact_id="prov_pipe",
    model_provider="pytest",
    model_id="unit_test",
    prompt_hash="0x0",
    created_at="2026-09-09T00:00:00Z",
)


def _claim(domain: str, polarity: Polarity) -> StructuredClaim:
    subject, predicate = domain.split(".", 1)
    return StructuredClaim(
        claim_id=f"c_{domain}_{polarity.value}",
        subject=subject,
        predicate=predicate,
        target_object="True",
        polarity=polarity,
        domain_key=domain,
        raw_statement="",
        provenance=PROV,
    )


def test_predicate_prune_drops_only_matching_stances():
    trigger = _claim("Risk.ignored", Polarity.POSITIVE)
    kept = _claim("Risk.ignored", Polarity.NEGATIVE)
    doomed_prop = _claim("Risk.ignored", Polarity.POSITIVE)
    doomed = doomed_prop.model_copy(update={"claim_id": "doomed"})
    safe_claim = kept.model_copy(update={"claim_id": "safe"})
    from prismcognition.schemas.core import EpistemicStance

    s_drop = EpistemicStance(
        stance_id="drop",
        cluster_id="2",
        method_id="2.1",
        conclusion=doomed,
        propositions=(doomed,),
        warrants=(),
        assumptions=(),
        normative_commitments=(),
        internal_confidence=1.0,
        provenance=PROV,
    )
    s_keep = EpistemicStance(
        stance_id="keep",
        cluster_id="4",
        method_id="4.1",
        conclusion=safe_claim,
        propositions=(safe_claim,),
        warrants=(),
        assumptions=(),
        normative_commitments=(),
        internal_confidence=1.0,
        provenance=PROV,
    )
    boundary = RuinBoundary(
        boundary_id="b1",
        failure_state=trigger,
        trigger_predicates=(trigger,),
        reversibility=0.0,
        severity=1.0,
        provenance=PROV,
    )
    ruin = RuinAnalysisResult(
        status=RuinAnalysisStatus.COMPLETED,
        boundaries=(boundary,),
        diagnostics=(),
        execution_time_ms=1.0,
        provenance=PROV,
    )
    surviving, diagnostics = SubtractiveGate().apply_subtractive_gate([(s_drop, None), (s_keep, None)], ruin)
    assert [item[0].stance_id for item in surviving] == ["keep"]
    assert any("drop" in item for item in diagnostics)


def test_failed_ruin_does_not_look_like_safety():
    from prismcognition.schemas.core import EpistemicStance

    claim = _claim("X.y", Polarity.POSITIVE)
    stance = EpistemicStance(
        stance_id="s",
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
    ruin = RuinAnalysisResult(
        status=RuinAnalysisStatus.FAILED,
        boundaries=(),
        diagnostics=("Chaos Room quarantine parser failure: structured extract rejected.",),
        execution_time_ms=3.0,
        provenance=PROV,
    )
    surviving, diagnostics = SubtractiveGate().apply_subtractive_gate([(stance, None)], ruin)
    assert len(surviving) == 1
    assert any("ruin analysis status is failed" in item for item in diagnostics)


def test_grounding_rollup_keeps_none():
    grounding = WarrantGrounding(
        warrant_id="w",
        epistemic_regime=GroundingRegime.EMPIRICAL,
        status=GroundingStatus.INSUFFICIENT_EVIDENCE,
        support_score=None,
        epistemic_reasoning="no evidence",
        provenance=PROV,
    )
    profile = GroundingRollupEngine.compile_profile([grounding])
    empirical = profile.get_assessment(GroundingRegime.EMPIRICAL)
    assert empirical is not None
    assert empirical.score is None
    assert empirical.insufficient_evidence_count == 1
    normative = profile.get_assessment(GroundingRegime.NORMATIVE)
    assert normative is not None
    assert normative.applicable is False


@pytest.mark.asyncio
async def test_verifier_does_not_use_declared_confidence_as_score():
    verifier = RegimeAwareGroundingVerifier()
    claim = _claim("X.y", Polarity.POSITIVE)
    warrant = Warrant(
        warrant_id="w",
        epistemic_regime=GroundingRegime.EMPIRICAL,
        rule_statement="r",
        premise_claim_ids=(),
        conclusion_claim_id=claim.claim_id,
        declared_confidence=0.99,
        provenance=PROV,
    )
    result = await verifier.verify_warrant(warrant, {claim.claim_id: claim})
    assert result.support_score is None
    assert result.status == GroundingStatus.INSUFFICIENT_EVIDENCE


@pytest.mark.asyncio
async def test_chaos_timeout_and_failure():
    timeout_room = ChaosRoomSandbox(ScriptedChaosLLM(delay_s=0.2), HashEmbedder(), timeout_seconds=0.01)
    timed = await timeout_room.generate_quarantined_boundaries("x")
    assert timed.status == RuinAnalysisStatus.TIMEOUT
    assert timed.boundaries == ()

    failed_room = ChaosRoomSandbox(ScriptedChaosLLM(fail=True), HashEmbedder(), timeout_seconds=2.0)
    failed = await failed_room.generate_quarantined_boundaries("x")
    assert failed.status == RuinAnalysisStatus.FAILED
    assert all("RuntimeError" not in item for item in failed.diagnostics)


@pytest.mark.asyncio
async def test_chaos_structured_extract_never_stores_prose():
    room = ChaosRoomSandbox(ScriptedChaosLLM(), HashEmbedder(), timeout_seconds=2.0)
    result = await room.generate_quarantined_boundaries("factory siting")
    assert result.status == RuinAnalysisStatus.COMPLETED
    assert result.boundaries
    dumped = result.model_dump_json()
    assert "You are an isolated catastrophic risk analyzer" not in dumped
    assert "Irreversible systemic collapse" not in dumped
    assert result.boundaries[0].failure_state.raw_statement == ""


@pytest.mark.asyncio
async def test_orchestrator_end_to_end_and_replay():
    orchestrator = EpistemicOrchestrator(
        router=DepthAllocatingRouter(UniformClassifier(0.05)),
        chaos_room=ChaosRoomSandbox(ScriptedChaosLLM(), HashEmbedder()),
        cluster_registry=default_cluster_registry(),
        grounding_verifier=RegimeAwareGroundingVerifier(
            StaticEvidenceAdapter({"unused": (GroundingStatus.SUPPORTED, 0.9)})
        ),
        clash_threshold=0.35,
    )
    artifact = await orchestrator.deliberate("Should we expand the plant this quarter?", risk_level="LOW")
    assert artifact.optional_recommendation is None
    assert artifact.ruin_analysis.status == RuinAnalysisStatus.COMPLETED
    assert "7" not in artifact.methods_skipped
    assert artifact.methods_probed
    coverage = compile_coverage(artifact)
    assert 0.0 <= coverage.coverage_index <= 1.0
    assert artifact.coverage is not None
    assert artifact.route_plan is not None
    assert artifact.coverage.coverage_index == coverage.coverage_index

    bundle = orchestrator.last_frozen_bundle()
    replayed = ReplayEngine.replay(bundle)
    assert ReplayEngine.canonical_hash(replayed) == ReplayEngine.canonical_hash(ReplayEngine.replay(bundle))
    assert replayed.deliberation_id == artifact.deliberation_id
    assert replayed.evidence_needed == artifact.evidence_needed
    assert replayed.irreducible_tensions == artifact.irreducible_tensions
    assert replayed.coverage is not None
    assert replayed.assumptions_that_matter == artifact.assumptions_that_matter


@pytest.mark.asyncio
async def test_catalog_probe_runs_primary_only():
    from prismcognition.engine.catalog import methods_for
    from prismcognition.schemas.core import ExecutionTier

    probe = methods_for("1", ExecutionTier.PROBE)
    required = methods_for("1", ExecutionTier.REQUIRED)
    assert [item.method_id for item in probe] == ["1.1"]
    assert [item.method_id for item in required] == ["1.1", "1.2"]

    group = default_cluster_registry()["2"]
    probed = await group.evaluate_all("measure rainfall totals", depth=ExecutionTier.PROBE)
    full = await group.evaluate_all("measure rainfall totals", depth=ExecutionTier.REQUIRED)
    assert [item.method_id for item in probed] == ["2.1"]
    assert [item.method_id for item in full] == ["2.1", "2.2"]


@pytest.mark.asyncio
async def test_evidence_store_assess_and_persist(tmp_path):
    from prismcognition.evidence.store import EvidenceStore
    from prismcognition.schemas.core import EvidenceRecord

    store = EvidenceStore(snapshot_id="snap-1")
    store.ingest(
        EvidenceRecord(
            record_id="e1",
            snapshot_id="snap-1",
            domain_key="Inquiry.thesis_holds",
            polarity=Polarity.POSITIVE,
            regime=GroundingRegime.EMPIRICAL,
            status=GroundingStatus.SUPPORTED,
            support_score=0.82,
            source_ref="note://field-1",
        )
    )
    claim = _claim("Inquiry.thesis_holds", Polarity.POSITIVE)
    warrant = Warrant(
        warrant_id="w",
        epistemic_regime=GroundingRegime.EMPIRICAL,
        rule_statement="r",
        premise_claim_ids=(),
        conclusion_claim_id=claim.claim_id,
        declared_confidence=0.4,
        provenance=PROV,
    )
    status, score, support, _counter, _reason = await store.assess(warrant, {claim.claim_id: claim})
    assert status == GroundingStatus.SUPPORTED
    assert score == 0.82
    assert support == ("note://field-1",)

    path = tmp_path / "evidence.json"
    store.save_json(path)
    reloaded = EvidenceStore.load_json(path)
    assert reloaded.snapshot_id == "snap-1"
    assert reloaded.records_for("Inquiry.thesis_holds", GroundingRegime.EMPIRICAL)[0].support_score == 0.82


def test_renderer_distinguishes_none_from_zero():
    from prismcognition.render.artifact import UNGROUNDED, render_score

    assert render_score(None) == UNGROUNDED
    assert render_score(0.0) == "0.0000"
    assert render_score(None) != render_score(0.0)
