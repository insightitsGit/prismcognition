from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from prismcognition.identity import artifact_id, new_provenance, prompt_digest
from prismcognition.schemas.core import (
    ChaosExtractionEnvelope,
    Polarity,
    RuinAnalysisResult,
    RuinAnalysisStatus,
    RuinBoundary,
    StructuredClaim,
)


class ChaosRoomSandbox:
    """Isolated Via Negativa generator. Raw catastrophic prose never leaves this boundary."""

    def __init__(self, llm_adapter, embedder_adapter, timeout_seconds: float = 8.0):
        self.llm = llm_adapter
        self.embedder = embedder_adapter
        self.timeout = timeout_seconds

    async def generate_quarantined_boundaries(self, inquiry_text: str) -> RuinAnalysisResult:
        started = time.monotonic()
        session_id = artifact_id("chaos", inquiry_text)
        provenance = new_provenance(
            artifact_id_value=artifact_id("prov_chaos", inquiry_text),
            model_provider="sandboxed_llm",
            model_id="chaos_room_v1",
            prompt_hash=prompt_digest(("chaos_room_v1", inquiry_text)),
            evidence_snapshot_id=session_id,
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an isolated catastrophic risk analyzer. Output valid JSON adhering strictly to: "
                    '{"boundaries": [{"id": str, "failure_claim": {"sub": str, "pred": str, "obj": str, '
                    '"polarity": int, "statement": str}, "triggers": [{"sub": str, "pred": str, "obj": str, '
                    '"polarity": int}], "reversibility": float, "severity": float}]}'
                ),
            },
            {"role": "user", "content": f"Analyze absorbing failure modes: {inquiry_text}"},
        ]

        try:
            response_json = await asyncio.wait_for(
                self._generate_structured(messages),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            return RuinAnalysisResult(
                status=RuinAnalysisStatus.TIMEOUT,
                boundaries=(),
                diagnostics=("Chaos Room breached strict latency SLA; no safe boundaries constructed.",),
                execution_time_ms=_elapsed_ms(started),
                provenance=provenance,
            )
        except Exception:
            return RuinAnalysisResult(
                status=RuinAnalysisStatus.FAILED,
                boundaries=(),
                diagnostics=("Chaos Room quarantine parser failure: structured extract rejected.",),
                execution_time_ms=_elapsed_ms(started),
                provenance=provenance,
            )

        raw_boundaries = response_json.get("boundaries", [])
        if not isinstance(raw_boundaries, list):
            return RuinAnalysisResult(
                status=RuinAnalysisStatus.FAILED,
                boundaries=(),
                diagnostics=("Chaos Room quarantine parser failure: 'boundaries' was not a list.",),
                execution_time_ms=_elapsed_ms(started),
                provenance=provenance,
            )

        envelope = ChaosExtractionEnvelope(
            session_id=session_id,
            raw_response_digest=prompt_digest(response_json),
            ruin_primitives=tuple(item for item in raw_boundaries if isinstance(item, dict)),
        )
        _ = envelope  # digest-only envelope; raw completion text is never retained

        parsed: List[RuinBoundary] = []
        dropped = 0
        for raw_boundary in envelope.ruin_primitives:
            try:
                parsed.append(await self._translate_primitive(raw_boundary, provenance))
            except Exception:
                dropped += 1

        if parsed:
            status = RuinAnalysisStatus.COMPLETED
        else:
            status = RuinAnalysisStatus.PARTIAL

        diagnostics = []
        if dropped:
            diagnostics.append(f"Dropped {dropped} malformed ruin primitives inside the quarantine envelope.")
        if not parsed:
            diagnostics.append("No well-formed ruin boundaries survived containment translation.")

        return RuinAnalysisResult(
            status=status,
            boundaries=tuple(parsed),
            diagnostics=tuple(diagnostics),
            execution_time_ms=_elapsed_ms(started),
            provenance=provenance,
        )

    async def _generate_structured(self, messages: Sequence[Mapping[str, str]]) -> Dict[str, Any]:
        payload = await self.llm.generate_json(messages, temperature=0.95)
        if not isinstance(payload, dict):
            raise TypeError("adapter_did_not_return_object")
        return payload

    async def _translate_primitive(self, raw_boundary: Dict[str, Any], provenance) -> RuinBoundary:
        failure = raw_boundary["failure_claim"]
        triggers = raw_boundary["triggers"]
        if not isinstance(triggers, list) or not triggers:
            raise ValueError("missing_triggers")

        trigger_expr = " AND ".join(f"{item['sub']}.{item['pred']}=={item['polarity']}" for item in triggers)
        vector = await self.embedder.embed(trigger_expr)
        array = np.asarray(vector, dtype=float).reshape(-1)
        norm = float(np.linalg.norm(array))
        semantic = tuple(float(x) for x in (array / norm).tolist()) if norm > 1e-7 else None

        boundary_id = str(raw_boundary["id"])
        return RuinBoundary(
            boundary_id=boundary_id,
            failure_state=StructuredClaim(
                claim_id=f"ruin_{boundary_id}",
                subject=str(failure["sub"]),
                predicate=str(failure["pred"]),
                target_object=str(failure["obj"]),
                polarity=Polarity(int(failure["polarity"])),
                domain_key=f"{failure['sub']}.{failure['pred']}",
                raw_statement="",
                provenance=provenance,
            ),
            trigger_predicates=tuple(
                StructuredClaim(
                    claim_id=f"trig_{index}_{boundary_id}",
                    subject=str(item["sub"]),
                    predicate=str(item["pred"]),
                    target_object=str(item["obj"]),
                    polarity=Polarity(int(item["polarity"])),
                    domain_key=f"{item['sub']}.{item['pred']}",
                    raw_statement="",
                    provenance=provenance,
                )
                for index, item in enumerate(triggers)
            ),
            reversibility=float(raw_boundary["reversibility"]),
            severity=float(raw_boundary["severity"]),
            semantic_vector=semantic,
            provenance=provenance,
        )


def _elapsed_ms(started: float) -> float:
    return (time.monotonic() - started) * 1000.0
