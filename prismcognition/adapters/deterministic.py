from __future__ import annotations

import asyncio
import hashlib
from typing import Any, Dict, Mapping, Sequence

import numpy as np


class HashEmbedder:
    async def embed(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = np.frombuffer(digest[:32], dtype=np.uint8).astype(float)
        vector = raw - raw.mean()
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else vector


class UniformClassifier:
    def __init__(self, score: float = 0.05):
        self.score = score

    async def score_clusters(self, inquiry: str) -> Dict[int, float]:
        _ = inquiry
        return {index: self.score for index in range(1, 7)}


class ScriptedChaosLLM:
    def __init__(self, payload: Dict[str, Any] | None = None, delay_s: float = 0.0, fail: bool = False):
        self.payload = payload or {
            "boundaries": [
                {
                    "id": "absorb_1",
                    "failure_claim": {
                        "sub": "System",
                        "pred": "collapses",
                        "obj": "True",
                        "polarity": 1,
                        "statement": "Irreversible systemic collapse.",
                    },
                    "triggers": [{"sub": "System", "pred": "overheats", "obj": "True", "polarity": 1}],
                    "reversibility": 0.1,
                    "severity": 0.9,
                }
            ]
        }
        self.delay_s = delay_s
        self.fail = fail

    async def generate_json(
        self,
        messages: Sequence[Mapping[str, str]],
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        _ = (messages, temperature)
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.fail:
            raise RuntimeError("chaos adapter refused to emit structured JSON")
        return self.payload
