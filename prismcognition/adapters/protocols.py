"""Model-agnostic adapter contracts. No provider is hard-wired into the engine."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from prismcognition.schemas.core import (
    EpistemicStance,
    ExecutionTier,
    GroundingStatus,
    StructuredClaim,
    Warrant,
    WarrantGrounding,
)


@runtime_checkable
class LLMAdapter(Protocol):
    async def generate_json(
        self,
        messages: Sequence[Mapping[str, str]],
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Return parsed JSON only. Never leak raw completion text to callers."""


@runtime_checkable
class EmbedderAdapter(Protocol):
    async def embed(self, text: str) -> np.ndarray:
        """Return a 1-D float vector. Used only as an auxiliary geometric container."""


@runtime_checkable
class ClusterClassifier(Protocol):
    async def score_clusters(self, inquiry: str) -> Dict[int, float]:
        """Map cluster IDs 1–7 to relevance in [0, 1]. Missing keys are treated as 0.0 compute depth, not epistemic emptiness."""


@runtime_checkable
class ClusterEvaluator(Protocol):
    async def evaluate(self, inquiry: str, *, depth: ExecutionTier) -> EpistemicStance:
        """Emit one frozen epistemic stance for a cluster/method."""


@runtime_checkable
class EvidenceAdapter(Protocol):
    async def assess(
        self,
        warrant: Warrant,
        claims_universe: Dict[str, StructuredClaim],
    ) -> tuple[GroundingStatus, Optional[float], tuple[str, ...], tuple[str, ...], str]:
        """Native-regime assessment. Absence must be score=None, never 0.0."""


@runtime_checkable
class GroundingVerifier(Protocol):
    async def verify_warrant(
        self,
        warrant: Warrant,
        claims_universe: Dict[str, StructuredClaim],
    ) -> WarrantGrounding:
        ...
