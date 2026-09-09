"""Repeatable local acceptance scenarios; no real provider calls or credentials."""
import asyncio

import pytest
from fastapi.testclient import TestClient

from prismcognition.api.app import create_app
from prismcognition.factory import build_default_orchestrator
from prismcognition.persist.store import ArtifactStore
from prismcognition.replay.engine import ReplayEngine
from prismcognition.settings import load_settings


@pytest.mark.parametrize("risk", ["LOW", "HIGH"])
def test_evidence_deliberation_restart_and_offline_replay(tmp_path, monkeypatch, risk):
    settings = load_settings(data_dir=str(tmp_path), live=False)
    with TestClient(create_app(settings)) as client:
        evidence = client.post("/api/evidence", json={
            "record_id": "measurement-1", "domain_key": "Inquiry.thesis_holds",
            "polarity": 1, "regime": "EMPIRICAL", "support_score": 0.85,
            "source_ref": "measurement://reviewed/1",
        })
        assert evidence.status_code == 200
        created = client.post("/api/deliberations", json={
            "inquiry": "Should we expand the plant this quarter?", "risk_level": risk,
        })
        assert created.status_code == 200
        original = created.json()["artifact"]
        assert original["coverage"] is not None
        assert original["optional_recommendation"] is None
        identifier = original["deliberation_id"]

    def no_generation(**kwargs):
        raise AssertionError("Replay must not generate new model output")

    monkeypatch.setattr("prismcognition.api.app.build_default_orchestrator", no_generation)
    with TestClient(create_app(settings)) as restarted:
        assert identifier in restarted.get("/api/deliberations").json()["ids"]
        assert restarted.get(f"/api/deliberations/{identifier}").json()["artifact"] == original
        replay = restarted.post(f"/api/deliberations/{identifier}/replay")
        assert replay.status_code == 200
        replayed = replay.json()["artifact"]
        for field in ("inquiry_text", "methods_executed", "methods_probed", "methods_failed", "route_plan", "optional_recommendation"):
            assert replayed[field] == original[field]
        assert replayed["coverage"]["coverage_index"] == original["coverage"]["coverage_index"]


async def test_concurrent_independent_inquiries_keep_matching_bundles(tmp_path):
    settings = load_settings(data_dir=str(tmp_path), live=False)
    store = ArtifactStore(settings)

    async def run(index):
        engine = build_default_orchestrator(settings=settings)
        artifact = await engine.deliberate(f"Measure rainfall for district {index}", risk_level="LOW")
        store.save_artifact(artifact)
        store.save_bundle(engine.last_frozen_bundle())
        return artifact

    artifacts = await asyncio.gather(*(run(index) for index in range(8)))
    assert len({item.deliberation_id for item in artifacts}) == 8
    assert len(store.list_deliberation_ids()) == 8
    for artifact in artifacts:
        bundle = store.load_bundle(artifact.deliberation_id)
        assert bundle.inquiry_text == artifact.inquiry_text
        assert ReplayEngine.replay(bundle).inquiry_text == artifact.inquiry_text


async def test_cluster_outage_is_visible_and_other_methods_survive(tmp_path):
    class Unavailable:
        async def evaluate_all(self, *args, **kwargs):
            raise RuntimeError("secret provider diagnostic")

    engine = build_default_orchestrator(settings=load_settings(data_dir=str(tmp_path), live=False))
    engine.clusters["2"] = Unavailable()
    artifact = await engine.deliberate("Measure the outcome of the experiment", risk_level="HIGH")
    assert "2: RuntimeError" in artifact.methods_failed
    assert artifact.methods_executed or artifact.methods_probed
    assert "secret provider diagnostic" not in artifact.model_dump_json()
    assert ReplayEngine.replay(engine.last_frozen_bundle()).methods_failed == artifact.methods_failed
