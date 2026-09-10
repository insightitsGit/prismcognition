"""Offline scaffolding is content-derived; evidence closes factual clashes only."""
from __future__ import annotations

import pytest

from prismcognition.adapters.features import FeatureClassifier
from prismcognition.engine.derive import collect_pair_artifacts
from prismcognition.engine.inquiry import extract_inquiry_features
from prismcognition.engine.tension import HardenedTensionEngine
from prismcognition.engine.thesis import extract_thesis
from prismcognition.evidence.store import EvidenceStore
from prismcognition.factory import build_default_orchestrator
from prismcognition.render.artifact import render_deliberation
from prismcognition.schemas.core import (
    EvidenceRecord,
    GroundingProfile,
    GroundingRegime,
    GroundingStatus,
    NormativeCommitment,
    Polarity,
    ProvenanceRef,
    RegimeAssessment,
    ResolutionStatus,
    StructuredClaim,
    EpistemicStance,
)


PROV = ProvenanceRef(
    artifact_id="prov_content",
    model_provider="pytest",
    model_id="content",
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


def test_thesis_keys_follow_inquiry_content():
    paint = extract_thesis("What color should we paint the office?")
    bankrupt = extract_thesis("Should we file for bankruptcy immediately?")
    expand = extract_thesis("Should we expand the plant this quarter?")
    rain = extract_thesis("measure rainfall totals")
    assert paint.domain_key == "Office.paint_color"
    assert paint.assumption_name == "aesthetic_weight"
    assert paint.severity == "routine"
    assert bankrupt.domain_key == "Company.file_bankruptcy"
    assert bankrupt.assumption_name == "solvency_buffer"
    assert bankrupt.severity == "catastrophic"
    assert expand.domain_key == "Plant.expand"
    assert rain.domain_key == "Rainfall.observed"


def test_inquiry_flags_distinguish_trivial_from_catastrophic():
    paint = extract_inquiry_features("What color should we paint the office?")
    bankrupt = extract_inquiry_features("Should we file for bankruptcy immediately?")
    assert paint.routine_or_aesthetic
    assert not paint.catastrophic_stakes
    assert bankrupt.catastrophic_stakes
    assert not bankrupt.routine_or_aesthetic


@pytest.mark.asyncio
async def test_classifier_scores_diverge_for_paint_and_bankruptcy():
    classifier = FeatureClassifier()
    paint = await classifier.score_clusters("What color should we paint the office?")
    bankrupt = await classifier.score_clusters("Should we file for bankruptcy immediately?")
    expand = await classifier.score_clusters("Should we expand the plant this quarter?")
    assert paint != bankrupt
    assert bankrupt[4] > paint[4]
    assert expand[6] >= 0.8


@pytest.mark.asyncio
async def test_offline_paint_and_bankruptcy_are_not_the_same_skeleton():
    # Isolated assumption/normative contributions are below the default 0.35
    # aggregate cutoff; use 0.1 to inspect those lower-tension rows explicitly.
    engine = build_default_orchestrator(evidence_store=EvidenceStore(snapshot_id="empty-content"), clash_threshold=0.1)
    paint = await engine.deliberate("What color should we paint the office?", risk_level="LOW")
    bankrupt = await engine.deliberate("Should we file for bankruptcy immediately?", risk_level="LOW")

    assert paint.thesis_domain_key == "Office.paint_color"
    assert bankrupt.thesis_domain_key == "Company.file_bankruptcy"
    assert {item.parameter_name for item in paint.assumptions_that_matter} == {"aesthetic_weight"}
    assert {item.parameter_name for item in bankrupt.assumptions_that_matter} == {"solvency_buffer"}

    paint_types = {item.clash.clash_type for item in paint.active_disagreements}
    bankrupt_types = {item.clash.clash_type for item in bankrupt.active_disagreements}
    assert "NORMATIVE_CONFLICT" in bankrupt_types
    assert "NORMATIVE_CONFLICT" not in paint_types
    assert len(paint.active_disagreements) != len(bankrupt.active_disagreements)

    rendered = render_deliberation(paint)
    assert "Execution mode: OFFLINE" in rendered
    assert "Thesis domain key: Office.paint_color" in rendered
    assert "structural demo" in rendered.lower()
    assert "not expert" in rendered.lower()


def test_factual_row_resolves_while_normative_row_stays_active():
    engine = HardenedTensionEngine()
    fact_pos = _claim("f1", "X.Y", Polarity.POSITIVE)
    fact_neg = _claim("f2", "X.Y", Polarity.NEGATIVE)
    deontic = _claim("n1", "Action.taken", Polarity.POSITIVE)
    s1 = _stance(
        "s1",
        "2",
        fact_pos,
        propositions=(fact_pos,),
        normative_commitments=(
            NormativeCommitment(
                axiom_name="A1",
                framework_id="FW1",
                priority_rank=1,
                prohibited_claims=(deontic,),
                provenance=PROV,
            ),
        ),
    )
    s2 = _stance(
        "s2",
        "6",
        fact_neg,
        propositions=(fact_neg,),
        normative_commitments=(
            NormativeCommitment(
                axiom_name="A2",
                framework_id="FW2",
                priority_rank=9,
                obligatory_claims=(deontic,),
                provenance=PROV,
            ),
        ),
    )
    left = GroundingProfile(
        regime_assessments=(RegimeAssessment(regime=GroundingRegime.EMPIRICAL, score=0.95, supported_count=1),)
    )
    right = GroundingProfile(
        regime_assessments=(RegimeAssessment(regime=GroundingRegime.EMPIRICAL, score=0.95, contradicted_count=1),)
    )
    evaluation = engine.evaluate_pair(s1, s2, left, right, {"f1": fact_pos, "f2": fact_neg, "n1": deontic})
    assert evaluation.status == ResolutionStatus.NORMATIVELY_IRREDUCIBLE

    active, resolved, _, _, irreducible = collect_pair_artifacts(
        [s1, s2], {"s1": left, "s2": right}, {"f1": fact_pos, "f2": fact_neg, "n1": deontic}, engine, 0.35
    )
    assert {item.clash.clash_type for item in resolved} == {"FACTUAL_CONFLICT"}
    assert all(item.resolution_status == ResolutionStatus.RESOLVED_BY_EVIDENCE for item in resolved)
    assert {item.clash.clash_type for item in active} == {"NORMATIVE_CONFLICT"}
    assert any("normatively_irreducible" in item for item in irreducible)


@pytest.mark.asyncio
async def test_empirical_evidence_closes_factual_clashes_and_not_the_rest():
    inquiry = "Should we expand the plant this quarter?"
    before_engine = build_default_orchestrator(evidence_store=EvidenceStore(snapshot_id="pre-evidence"), clash_threshold=0.1)
    before = await before_engine.deliberate(inquiry, risk_level="LOW")
    assert before.thesis_domain_key == "Plant.expand"
    assert before.resolved_disagreements == ()
    factual_before = [item for item in before.active_disagreements if item.clash.clash_type == "FACTUAL_CONFLICT"]
    other_before = [item for item in before.active_disagreements if item.clash.clash_type != "FACTUAL_CONFLICT"]
    assert factual_before
    assert other_before

    store = EvidenceStore(snapshot_id="post-evidence")
    for index in range(5):
        store.ingest(
            EvidenceRecord(
                record_id=f"row-{index}",
                snapshot_id="post-evidence",
                domain_key="Plant.expand",
                polarity=Polarity.POSITIVE,
                regime=GroundingRegime.EMPIRICAL,
                status=GroundingStatus.SUPPORTED,
                support_score=0.95,
                source_ref=f"measurement://strong/{index}",
            )
        )
    after_engine = build_default_orchestrator(evidence_store=store, clash_threshold=0.1)
    after = await after_engine.deliberate(inquiry, risk_level="LOW")

    assert after.strongly_supported_claims
    assert after.resolved_disagreements
    assert all(
        item.clash.clash_type in {"FACTUAL_CONFLICT", "CAUSAL_CONFLICT"}
        and item.resolution_status == ResolutionStatus.RESOLVED_BY_EVIDENCE
        for item in after.resolved_disagreements
    )
    assert len(after.active_disagreements) < len(before.active_disagreements)
    assert not any(
        item.resolution_status == ResolutionStatus.RESOLVED_BY_EVIDENCE for item in after.active_disagreements
    )
    remaining_types = {item.clash.clash_type for item in after.active_disagreements}
    assert remaining_types & {"ASSUMPTION_CONFLICT", "NORMATIVE_CONFLICT", "DEFINITIONAL_CONFLICT", "INFERENCE_RULE_CONFLICT"}
    rendered = render_deliberation(after)
    assert "Resolved by evidence:" in rendered
    assert after.thesis_domain_key == "Plant.expand"
