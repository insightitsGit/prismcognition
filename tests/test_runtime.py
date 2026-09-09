from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from prismcognition.adapters.features import FeatureClassifier
from prismcognition.adapters.openai_compat import OpenAICompatibleEmbedder, OpenAICompatibleLLM
from prismcognition.api.app import create_app
from prismcognition.engine.catalog import METHOD_CATALOG
from prismcognition.engine.clusters import default_cluster_registry
from prismcognition.engine.hybrid import HybridClusterGroup
from prismcognition.engine.recommend import draft_optional_recommendation
from prismcognition.persist.store import ArtifactStore
from prismcognition.schemas.core import (
    DeliberationArtifact,
    ExecutionTier,
    ProvenanceRef,
    RuinAnalysisResult,
    RuinAnalysisStatus,
)
from prismcognition.settings import load_settings


@pytest.mark.asyncio
async def test_feature_classifier_raises_relevant_clusters():
    classifier = FeatureClassifier()
    formal = await classifier.score_clusters("Prove that this is a formal axiom definition purely a priori")
    assert formal[1] >= 0.8
    moral = await classifier.score_clusters("Should we expand the plant this quarter?")
    assert moral[6] >= 0.8
    assert moral[4] >= 0.7


@pytest.mark.asyncio
async def test_openai_compat_parses_json_only():
    def transport(url, headers, body, timeout):
        assert "Authorization" in headers
        assert url.endswith("/chat/completions")
        return {
            "choices": [
                {"message": {"content": json.dumps({"boundaries": [], "ok": True})}}
            ]
        }

    llm = OpenAICompatibleLLM(api_key="sk-test", transport=transport)
    payload = await llm.generate_json([{"role": "user", "content": "x"}], temperature=0.1)
    assert payload["ok"] is True


@pytest.mark.asyncio
async def test_openai_compat_rejects_non_json_content():
    def transport(url, headers, body, timeout):
        return {"choices": [{"message": {"content": "not json"}}]}

    llm = OpenAICompatibleLLM(api_key="sk-test", transport=transport)
    with pytest.raises(ValueError):
        await llm.generate_json([{"role": "user", "content": "x"}])


@pytest.mark.asyncio
async def test_openai_embedder_reads_vector():
    def transport(url, headers, body, timeout):
        return {"data": [{"embedding": [0.0, 1.0, 0.0]}]}

    embedder = OpenAICompatibleEmbedder(api_key="sk-test", transport=transport)
    vector = await embedder.embed("trigger")
    assert np.allclose(vector, [0.0, 1.0, 0.0])


def test_artifact_store_roundtrip(tmp_path):
    settings = load_settings(data_dir=str(tmp_path))
    store = ArtifactStore(settings)
    prov = ProvenanceRef(
        artifact_id="p",
        model_provider="t",
        model_id="t",
        prompt_hash="0",
        created_at="2026-09-09T00:00:00Z",
    )
    ruin = RuinAnalysisResult(
        status=RuinAnalysisStatus.COMPLETED,
        boundaries=(),
        diagnostics=(),
        execution_time_ms=1.0,
        provenance=prov,
    )
    artifact = DeliberationArtifact(
        deliberation_id="delib_store_1",
        inquiry_text="q",
        methods_executed=(),
        methods_probed=(),
        methods_skipped=(),
        methods_failed=(),
        strongly_supported_claims=(),
        ruin_analysis=ruin,
        active_disagreements=(),
        perspective_diversities=(),
        evidence_needed=(),
        assumptions_that_matter=(),
        irreducible_tensions=(),
        provenance=prov,
    )
    store.save_artifact(artifact)
    loaded = store.load_artifact("delib_store_1")
    assert loaded.inquiry_text == "q"
    assert store.list_deliberation_ids() == ["delib_store_1"]


def test_recommend_does_not_claim_consensus():
    prov = ProvenanceRef(
        artifact_id="p",
        model_provider="t",
        model_id="t",
        prompt_hash="0",
        created_at="2026-09-09T00:00:00Z",
    )
    artifact = DeliberationArtifact(
        deliberation_id="d",
        inquiry_text="q",
        methods_executed=(),
        methods_probed=(),
        methods_skipped=(),
        methods_failed=("7: ruin_analysis_failed",),
        strongly_supported_claims=(),
        ruin_analysis=RuinAnalysisResult(
            status=RuinAnalysisStatus.FAILED,
            boundaries=(),
            diagnostics=(),
            execution_time_ms=1.0,
            provenance=prov,
        ),
        active_disagreements=(),
        perspective_diversities=(),
        evidence_needed=("Need measurement of X",),
        assumptions_that_matter=(),
        irreducible_tensions=("1 vs 6: normatively_irreducible",),
        provenance=prov,
    )
    note = draft_optional_recommendation(artifact)
    assert "not a verdict" in note.lower()
    assert "do not treat missing boundaries as safety" in note
    assert "Need measurement of X" in note


@pytest.mark.asyncio
async def test_hybrid_falls_back_when_extractor_fails():
    class Boom:
        spec = METHOD_CATALOG["1"][0]

        async def evaluate(self, inquiry, *, depth):
            raise RuntimeError("provider down")

    group = HybridClusterGroup(default_cluster_registry()["1"], [Boom()])
    stances = await group.evaluate_all("formal check", depth=ExecutionTier.PROBE)
    assert stances[0].method_id == "1.1"


def test_api_deliberate_and_list(tmp_path):
    settings = load_settings(data_dir=str(tmp_path), live=False)
    client = TestClient(create_app(settings))
    home = client.get("/")
    assert home.status_code == 200
    assert "PrismCognition" in home.text

    created = client.post(
        "/api/deliberations",
        json={"inquiry": "Should we expand the plant this quarter?", "risk_level": "LOW"},
    )
    assert created.status_code == 200
    body = created.json()
    assert body["artifact"]["optional_recommendation"] is None
    assert "UNGROUNDED" in body["rendered"] or "calibrated" in body["rendered"]
    deliberation_id = body["artifact"]["deliberation_id"]

    listed = client.get("/api/deliberations")
    assert deliberation_id in listed.json()["ids"]

    fetched = client.get(f"/api/deliberations/{deliberation_id}")
    assert fetched.status_code == 200
    replayed = client.post(f"/api/deliberations/{deliberation_id}/replay")
    assert replayed.status_code == 200


def test_cli_shorthand_help():
    from prismcognition.cli import main

    assert main([]) == 2
