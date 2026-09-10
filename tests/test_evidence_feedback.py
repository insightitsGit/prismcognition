"""Evidence must affect public results, and uncertainty must not resolve clashes."""
import pytest
from fastapi.testclient import TestClient

from prismcognition.api.app import create_app
from prismcognition.persist.store import ArtifactStore
from prismcognition.replay.engine import ReplayEngine
from prismcognition.settings import load_settings


def row(identifier, *, status="SUPPORTED", polarity=1, domain="Plant.expand", regime="EMPIRICAL"):
    return dict(record_id=identifier, domain_key=domain, polarity=polarity,
                regime=regime, status=status, support_score=0.95,
                source_ref=f"synthetic://{identifier}")


def deliberate(client):
    response = client.post("/api/deliberations", json={
        "inquiry": "Should we expand the plant this quarter?", "risk_level": "LOW",
    })
    assert response.status_code == 200
    return response.json()["artifact"]


@pytest.mark.parametrize("status", ["UNRESOLVED", "INSUFFICIENT_EVIDENCE", "NOT_GROUNDABLE", "CONTRADICTED", "CONTESTED"])
def test_unaccepted_rows_cannot_strengthen_or_resolve(tmp_path, status):
    with TestClient(create_app(load_settings(data_dir=str(tmp_path), live=False))) as client:
        assert client.post("/api/evidence", json=row("uncertain", status=status)).status_code == 200
        artifact = deliberate(client)
        assert not artifact["strongly_supported_claims"]
        assert not artifact["resolved_disagreements"]
        assert artifact["active_disagreements"]


@pytest.mark.parametrize("change", [{"domain": "Other.claim"}, {"regime": "NORMATIVE"}])
def test_unrelated_evidence_does_not_resolve_empirical_conflict(tmp_path, change):
    with TestClient(create_app(load_settings(data_dir=str(tmp_path), live=False))) as client:
        assert client.post("/api/evidence", json=row("unrelated", **change)).status_code == 200
        assert not deliberate(client)["resolved_disagreements"]


@pytest.mark.parametrize("counter", [row("negative", polarity=-1), row("disputed", status="CONTESTED")])
def test_api_evidence_resolves_then_counterevidence_reopens_and_frozen_replay_stays_fixed(tmp_path, counter):
    settings = load_settings(data_dir=str(tmp_path), live=False)
    with TestClient(create_app(settings)) as client:
        before = deliberate(client)
        for index in range(5):
            assert client.post("/api/evidence", json=row(f"support-{index}")).status_code == 200
        after = deliberate(client)
        assert after["strongly_supported_claims"]
        assert after["resolved_disagreements"]
        assert len(after["active_disagreements"]) < len(before["active_disagreements"])
        assert {x["clash"]["clash_type"] for x in after["resolved_disagreements"]} == {"FACTUAL_CONFLICT"}
        bundle = ArtifactStore(settings).load_bundle(after["deliberation_id"])
        assert client.post("/api/evidence", json=counter).status_code == 200
        reopened = deliberate(client)
        assert not reopened["resolved_disagreements"]
        assert not reopened["strongly_supported_claims"]
        assert len(reopened["active_disagreements"]) == len(before["active_disagreements"])
        replayed = ReplayEngine.replay(bundle).model_dump(mode="json")
        for field in ("active_disagreements", "resolved_disagreements"):
            assert [x["disagreement_id"] for x in replayed[field]] == [x["disagreement_id"] for x in after[field]]


async def test_mixed_profile_cannot_claim_evidence_resolution():
    from prismcognition.engine.tension import HardenedTensionEngine
    from prismcognition.schemas.core import EpistemicClash, GroundingProfile, RegimeAssessment, GroundingRegime, ResolutionStatus

    clash = EpistemicClash(clash_id="pred_Plant.expand", clash_type="FACTUAL_CONFLICT",
                          clash_regime=GroundingRegime.EMPIRICAL, stance_id_left="a", stance_id_right="b",
                          explanation="test", provenance=GroundingProfile(regime_assessments=()).provenance)
    left = GroundingProfile(regime_assessments=(RegimeAssessment(regime=GroundingRegime.EMPIRICAL,
                            score=0.95, supported_count=1, contested_count=1),))
    right = GroundingProfile(regime_assessments=(RegimeAssessment(regime=GroundingRegime.EMPIRICAL,
                             score=0.95, contradicted_count=1),))
    assert HardenedTensionEngine().resolve_clash(clash, left, right, has_midpoint=False, raw_delta=1) == ResolutionStatus.CURRENTLY_UNRESOLVED


async def test_fallback_warning_does_not_leak_into_next_successful_run():
    from prismcognition.engine.hybrid import HybridClusterGroup
    from prismcognition.engine.clusters import default_cluster_registry
    from prismcognition.schemas.core import ExecutionTier

    group = default_cluster_registry()["1"]
    evaluator = group.evaluators[0]

    class OnceUnavailable:
        spec = evaluator.spec
        failed = False

        async def evaluate(self, inquiry, *, depth):
            if not self.failed:
                self.failed = True
                raise RuntimeError("private provider detail")
            return await evaluator.evaluate(inquiry, depth=depth)

    hybrid = HybridClusterGroup(group, [OnceUnavailable()])
    await hybrid.evaluate_all("q", depth=ExecutionTier.PROBE)
    assert len(hybrid.fallback_notes) == 1
    assert "private provider detail" not in hybrid.fallback_notes[0]
    await hybrid.evaluate_all("q", depth=ExecutionTier.PROBE)
    assert hybrid.fallback_notes == []
