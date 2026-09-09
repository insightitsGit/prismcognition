"""Offline regression tests for untrusted inputs and runtime failures."""
import asyncio
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from prismcognition.adapters.openai_compat import OpenAICompatibleEmbedder, OpenAICompatibleLLM
from prismcognition.api.app import create_app
from prismcognition.persist.atomic import atomic_write_text
from prismcognition.persist.store import ArtifactStore
from prismcognition.settings import load_settings


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(load_settings(data_dir=str(tmp_path), live=False))) as instance:
        yield instance


@pytest.mark.parametrize("payload", [
    {}, {"inquiry": ""}, {"inquiry": " \n\t"}, {"inquiry": "x" * 20001},
    {"inquiry": "question", "risk_level": "UNKNOWN"},
])
def test_invalid_deliberation_does_not_persist(client, tmp_path, payload):
    assert client.post("/api/deliberations", json=payload).status_code == 422
    assert not list(tmp_path.rglob("*.json"))


@pytest.mark.parametrize("field,value", [
    ("polarity", 2), ("regime", "BOGUS"), ("status", "BOGUS"),
    ("support_score", -0.01), ("support_score", 1.01),
    ("record_id", ""), ("source_ref", ""),
])
def test_invalid_evidence_is_client_error(client, tmp_path, field, value):
    payload = dict(record_id="e1", domain_key="Inquiry.thesis_holds", polarity=1,
                   regime="EMPIRICAL", source_ref="measurement://1")
    payload[field] = value
    assert client.post("/api/evidence", json=payload).status_code == 422
    assert not (tmp_path / "evidence.json").exists()


@pytest.mark.parametrize("inquiry,risk", [("   ", "HIGH"), ("q", "TYPO"), ("x" * 20001, "LOW")])
def test_form_has_same_input_limits(client, inquiry, risk):
    assert client.post("/deliberate", data={"inquiry": inquiry, "risk_level": risk}).status_code == 422


@pytest.mark.parametrize("identifier", ["../escape", "..\\escape", "C:\\escape", "/tmp/escape", "x:y", "", "x" * 129])
@pytest.mark.parametrize("operation", ["load_artifact", "load_bundle", "save_artifact", "save_bundle"])
def test_store_rejects_unsafe_ids(tmp_path, identifier, operation):
    store = ArtifactStore(load_settings(data_dir=str(tmp_path), live=False))
    argument = SimpleNamespace(deliberation_id=identifier) if operation.startswith("save") else identifier
    with pytest.raises(ValueError, match="invalid deliberation ID"):
        getattr(store, operation)(argument)
    assert not list(tmp_path.rglob("*.json"))


def test_failed_replace_preserves_previous_file_and_removes_temp(tmp_path, monkeypatch):
    path = tmp_path / "evidence.json"
    path.write_text('{"old": true}', encoding="utf-8")

    def disk_failure(*args):
        raise OSError("simulated disk failure")

    monkeypatch.setattr("prismcognition.persist.atomic.os.replace", disk_failure)
    with pytest.raises(OSError, match="simulated"):
        atomic_write_text(path, '{"new": true}')
    assert path.read_text(encoding="utf-8") == '{"old": true}'
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("vector", [[], [[1, 2]], 1, [float("nan")], [float("inf")], [None]])
async def test_invalid_embedding_is_rejected(vector):
    adapter = OpenAICompatibleEmbedder("test", transport=lambda *args: {"data": [{"embedding": vector}]})
    with pytest.raises(ValueError):
        await adapter.embed("query")


@pytest.mark.parametrize("adapter_type", [OpenAICompatibleLLM, OpenAICompatibleEmbedder])
async def test_provider_transport_does_not_block_event_loop(adapter_type):
    loop_thread = threading.get_ident()
    transport_threads = []

    def transport(*args):
        transport_threads.append(threading.get_ident())
        return {"choices": [{"message": {"content": "{}"}}], "data": [{"embedding": [1, 0]}]}

    adapter = adapter_type("test", transport=transport)
    if adapter_type is OpenAICompatibleLLM:
        await adapter.generate_json([])
    else:
        await adapter.embed("query")
    assert transport_threads and transport_threads[0] != loop_thread


async def test_provider_wait_can_be_cancelled_without_waiting_for_transport():
    entered = threading.Event()
    release = threading.Event()

    def transport(*args):
        entered.set()
        release.wait(5)
        return {"choices": [{"message": {"content": "{}"}}]}

    task = asyncio.create_task(OpenAICompatibleLLM("test", transport=transport).generate_json([]))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        assert not release.is_set()
    finally:
        release.set()


@pytest.mark.parametrize("suffix", ["", "/replay"])
def test_missing_artifact_returns_404(client, suffix):
    response = client.post("/api/deliberations/missing" + suffix) if suffix else client.get("/api/deliberations/missing")
    assert response.status_code == 404
