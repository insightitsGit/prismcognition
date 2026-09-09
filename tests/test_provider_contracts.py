import io
import json
import urllib.error

import pytest

from prismcognition.adapters.openai_compat import OpenAICompatibleEmbedder, OpenAICompatibleLLM


@pytest.mark.parametrize("adapter", [OpenAICompatibleLLM, OpenAICompatibleEmbedder])
def test_missing_credentials_rejected(adapter):
    with pytest.raises(ValueError, match="api_key"):
        adapter("")


@pytest.mark.parametrize("payload", [{}, {"choices": []}, {"choices": [{"message": {"content": ""}}]},
                                    {"choices": [{"message": {"content": "[]"}}]}])
async def test_malformed_completion_rejected(payload):
    adapter = OpenAICompatibleLLM("test", transport=lambda *args: payload)
    with pytest.raises(ValueError):
        await adapter.generate_json([])


@pytest.mark.parametrize("payload", [{}, {"data": []}, {"data": [None]}])
async def test_missing_embedding_rejected(payload):
    adapter = OpenAICompatibleEmbedder("test", transport=lambda *args: payload)
    with pytest.raises(ValueError, match="non-embedding"):
        await adapter.embed("q")


async def test_default_http_transport_encodes_request(monkeypatch):
    def urlopen(request, timeout):
        assert request.full_url == "https://provider.invalid/v1/chat/completions"
        assert request.get_header("Authorization") == "Bearer test"
        assert request.method == "POST"
        body = json.loads(request.data)
        assert body["messages"] == [{"role": "user", "content": "café"}]
        assert body["model"] == "test-model"
        assert timeout == 7
        return io.BytesIO(b'{"choices":[{"message":{"content":"{\\"ok\\":true}"}}]}')

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    adapter = OpenAICompatibleLLM("test", base_url="https://provider.invalid/v1/", model="test-model", timeout_s=7)
    assert await adapter.generate_json([{"role": "user", "content": "café"}]) == {"ok": True}


@pytest.mark.parametrize("raw,match", [(b"not json", "non-JSON"), (b"[]", "non-object")])
async def test_http_invalid_json_rejected(monkeypatch, raw, match):
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: io.BytesIO(raw))
    with pytest.raises(ValueError, match=match):
        await OpenAICompatibleLLM("test").generate_json([])


@pytest.mark.parametrize("error,match", [
    (urllib.error.HTTPError("https://provider.invalid", 429, "sensitive provider body", {}, None), "provider HTTP 429"),
    (urllib.error.URLError("connection refused"), "provider transport failure"),
])
async def test_http_failure_has_safe_message(monkeypatch, error, match):
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr("urllib.request.urlopen", fail)
    with pytest.raises(RuntimeError, match=match) as exc:
        await OpenAICompatibleLLM("test").generate_json([])
    assert "sensitive" not in str(exc.value)
