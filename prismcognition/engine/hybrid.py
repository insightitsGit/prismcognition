from __future__ import annotations

import asyncio
from typing import List, Sequence

from prismcognition.engine.catalog import methods_for
from prismcognition.engine.clusters import ClusterGroup
from prismcognition.engine.extractor import StructuredClusterExtractor
from prismcognition.schemas.core import EpistemicStance, ExecutionTier


class HybridClusterGroup:
    """Live JSON extractors with deterministic fallback. Failures never look like empty consensus."""

    def __init__(self, deterministic: ClusterGroup, extractors: Sequence[StructuredClusterExtractor] = ()):
        self.deterministic = deterministic
        self.extractors = list(extractors)
        self.fallback_notes: List[str] = []

    async def evaluate(self, inquiry: str, *, depth: ExecutionTier) -> EpistemicStance:
        stances = await self.evaluate_all(inquiry, depth=depth)
        return stances[0]

    async def evaluate_all(self, inquiry: str, *, depth: ExecutionTier) -> List[EpistemicStance]:
        self.fallback_notes.clear()
        if not self.extractors:
            return await self.deterministic.evaluate_all(inquiry, depth=depth)
        cluster_id = self.deterministic.evaluators[0].spec.cluster_id
        allowed = {item.method_id for item in methods_for(cluster_id, depth)}
        selected = [item for item in self.extractors if item.spec.method_id in allowed]
        if not selected:
            return await self.deterministic.evaluate_all(inquiry, depth=depth)
        try:
            return list(await asyncio.gather(*[item.evaluate(inquiry, depth=depth) for item in selected]))
        except Exception as exc:
            self.fallback_notes.append(
                f"LIVE FALLBACK: cluster {cluster_id} used deterministic adapters after "
                f"{type(exc).__name__}. This is not live model output for that cluster."
            )
            return await self.deterministic.evaluate_all(inquiry, depth=depth)
