from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np

Transport = Callable[[str, Dict[str, str], bytes, float], Dict[str, Any]]


class OpenAICompatibleLLM:
    """OpenAI-compatible chat completions. Returns parsed JSON only."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        timeout_s: float = 30.0,
        transport: Optional[Transport] = None,
    ):
        if not api_key:
            raise ValueError("api_key is required for live LLM calls")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self._transport = transport or _http_json_post

    async def generate_json(
        self,
        messages: Sequence[Mapping[str, str]],
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        body = {
            "model": self.model,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "messages": [{"role": item["role"], "content": item["content"]} for item in messages],
        }
        payload = await asyncio.to_thread(self._transport,
            f"{self.base_url}/chat/completions",
            self._headers(),
            json.dumps(body).encode("utf-8"),
            self.timeout_s,
        )
        content = _message_content(payload)
        parsed = _parse_json_object(content)
        return parsed

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }


class OpenAICompatibleEmbedder:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "text-embedding-3-small",
        timeout_s: float = 30.0,
        transport: Optional[Transport] = None,
    ):
        if not api_key:
            raise ValueError("api_key is required for live embeddings")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self._transport = transport or _http_json_post

    async def embed(self, text: str) -> np.ndarray:
        body = {"model": self.model, "input": text}
        payload = await asyncio.to_thread(self._transport,
            f"{self.base_url}/embeddings",
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json.dumps(body).encode("utf-8"),
            self.timeout_s,
        )
        try:
            vector = payload["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("embedding adapter received a non-embedding payload") from exc
        array = np.asarray(vector, dtype=float)
        if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
            raise ValueError("embedding adapter requires a finite, nonempty one-dimensional vector")
        return array


def _http_json_post(url: str, headers: Dict[str, str], body: bytes, timeout_s: float) -> Dict[str, Any]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"provider HTTP {exc.code}") from None
    except urllib.error.URLError as exc:
        raise RuntimeError("provider transport failure") from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("provider returned non-JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("provider returned a non-object")
    return payload


def _message_content(payload: Dict[str, Any]) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("chat adapter received a non-completion payload") from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("chat adapter received empty content")
    return content


def _parse_json_object(content: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("model content was not JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("model content was not a JSON object")
    return parsed
