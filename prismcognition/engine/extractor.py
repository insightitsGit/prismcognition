from __future__ import annotations

from typing import Any, Dict

from prismcognition.engine.catalog import MethodSpec
from prismcognition.engine.clusters import _commitments, _framework, _hash_vector
from prismcognition.engine.thesis import extract_thesis
from prismcognition.identity import artifact_id, new_provenance, prompt_digest
from prismcognition.schemas.core import (
    Assumption,
    EpistemicStance,
    ExecutionTier,
    GroundingRegime,
    Polarity,
    StructuredClaim,
    Warrant,
)


class StructuredClusterExtractor:
    """Optional live path: provider JSON → frozen EpistemicStance. No raw prose is retained."""

    def __init__(self, llm_adapter, spec: MethodSpec):
        self.llm = llm_adapter
        self.spec = spec

    async def evaluate(self, inquiry: str, *, depth: ExecutionTier) -> EpistemicStance:
        messages = [
            {
                "role": "system",
                "content": (
                    "Emit JSON only: {\"polarity\": int, \"subject\": str, \"predicate\": str, "
                    "\"object\": str, \"rule\": str, \"confidence\": float, "
                    "\"premise_predicate\": str, \"assumption_name\": str, "
                    "\"assumption_value\": \"number|bool|string\", "
                    "\"framework_id\": str, \"axiom_name\": str, \"priority_rank\": int}"
                ),
            },
            {"role": "user", "content": f"Cluster {self.spec.cluster_id} method {self.spec.method_id}: {inquiry}"},
        ]
        payload = await self.llm.generate_json(messages, temperature=0.0)
        if not isinstance(payload, dict):
            raise TypeError("cluster extractor received a non-object")
        return self._to_stance(inquiry, depth, payload)

    def _to_stance(self, inquiry: str, depth: ExecutionTier, payload: Dict[str, Any]) -> EpistemicStance:
        provenance = new_provenance(
            artifact_id_value=artifact_id("stance_x", self.spec.method_id, inquiry),
            model_provider="structured_extractor",
            model_id=f"extract_{self.spec.method_id}",
            prompt_hash=prompt_digest((self.spec.method_id, inquiry, depth.value)),
        )
        polarity = Polarity(int(payload.get("polarity", 1)))
        claim = StructuredClaim(
            claim_id=artifact_id("claim_x", self.spec.method_id, inquiry),
            subject=str(payload.get("subject", "Inquiry")),
            predicate=str(payload.get("predicate", "thesis_holds")),
            target_object=str(payload.get("object", "True")),
            polarity=polarity,
            domain_key=f"{payload.get('subject', 'Inquiry')}.{payload.get('predicate', 'thesis_holds')}",
            raw_statement=inquiry,
            provenance=provenance,
        )
        premise = StructuredClaim(
            claim_id=artifact_id("prem_x", self.spec.method_id, inquiry),
            subject="Inquiry",
            predicate=str(payload.get("premise_predicate", "is_well_posed")),
            target_object="True",
            polarity=Polarity.POSITIVE,
            domain_key=f"Inquiry.is_well_posed.{self.spec.cluster_id}.{self.spec.method_id}",
            raw_statement="Inquiry is treated as well-posed for structured extraction.",
            provenance=provenance,
        )
        warrant = Warrant(
            warrant_id=artifact_id("warr_x", self.spec.method_id, inquiry),
            epistemic_regime=self.spec.regime if isinstance(self.spec.regime, GroundingRegime) else GroundingRegime.EMPIRICAL,
            rule_statement=str(payload.get("rule", self.spec.name)),
            premise_claim_ids=(premise.claim_id,),
            conclusion_claim_id=claim.claim_id,
            declared_confidence=float(payload.get("confidence", 0.5)),
            normative_framework_id=str(payload["framework_id"]) if payload.get("framework_id") else _framework(self.spec),
            provenance=provenance,
        )
        assumptions: tuple[Assumption, ...] = ()
        if "assumption_name" in payload and "assumption_value" in payload:
            assumptions = (
                Assumption(
                    assumption_id=artifact_id("assump_x", self.spec.method_id, inquiry),
                    parameter_name=str(payload["assumption_name"]),
                    assumed_value=payload["assumption_value"],
                    provenance=provenance,
                ),
            )
        commitments = _commitments(self.spec, claim, extract_thesis(inquiry), provenance)
        if payload.get("framework_id") and not commitments:
            from prismcognition.schemas.core import NormativeCommitment

            commitments = (
                NormativeCommitment(
                    axiom_name=str(payload.get("axiom_name", "extracted")),
                    framework_id=str(payload["framework_id"]),
                    priority_rank=int(payload.get("priority_rank", 1)),
                    obligatory_claims=(claim,),
                    prohibited_claims=(),
                    provenance=provenance,
                ),
            )
        return EpistemicStance(
            stance_id=artifact_id("sid_x", self.spec.method_id, inquiry),
            cluster_id=self.spec.cluster_id,
            method_id=self.spec.method_id,
            conclusion=claim,
            propositions=(claim, premise),
            warrants=(warrant,),
            assumptions=assumptions,
            normative_commitments=commitments,
            internal_confidence=warrant.declared_confidence,
            semantic_vector=_hash_vector(inquiry, self.spec.cluster_id, self.spec.method_id),
            provenance=provenance,
        )
