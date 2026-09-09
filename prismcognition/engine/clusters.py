from __future__ import annotations

import asyncio
import hashlib
from typing import Dict, List, Sequence

from prismcognition.engine.catalog import METHOD_CATALOG, MethodSpec, methods_for
from prismcognition.identity import artifact_id, new_provenance, prompt_digest
from prismcognition.schemas.core import (
    Assumption,
    EpistemicStance,
    ExecutionTier,
    NormativeCommitment,
    Polarity,
    StructuredClaim,
    Warrant,
)


class DeterministicMethodEvaluator:
    """Emits one typed stance for a single catalog method. No model call."""

    def __init__(self, spec: MethodSpec):
        self.spec = spec

    async def evaluate(self, inquiry: str, *, depth: ExecutionTier) -> EpistemicStance:
        spec = self.spec
        provenance = new_provenance(
            artifact_id_value=artifact_id("stance", spec.cluster_id, spec.method_id, inquiry),
            model_provider="internal",
            model_id=f"deterministic_{spec.cluster_id}_{spec.method_id}",
            prompt_hash=prompt_digest((spec.cluster_id, spec.method_id, inquiry, depth.value)),
        )
        polarity = Polarity.NEGATIVE if spec.cluster_id == "4" else Polarity.POSITIVE
        claim = StructuredClaim(
            claim_id=artifact_id("claim", spec.cluster_id, spec.method_id, inquiry),
            subject="Inquiry",
            predicate="thesis_holds",
            target_object="True",
            polarity=polarity,
            domain_key="Inquiry.thesis_holds",
            raw_statement=inquiry,
            provenance=provenance,
        )
        premise = StructuredClaim(
            claim_id=artifact_id("prem", spec.cluster_id, spec.method_id, inquiry),
            subject="Inquiry",
            predicate="is_well_posed",
            target_object="True",
            polarity=Polarity.POSITIVE,
            domain_key=f"Inquiry.is_well_posed.{spec.cluster_id}.{spec.method_id}",
            raw_statement="Inquiry is treated as well-posed for structured extraction.",
            provenance=provenance,
        )
        warrant = Warrant(
            warrant_id=artifact_id("warr", spec.cluster_id, spec.method_id, inquiry),
            epistemic_regime=spec.regime,
            rule_statement=f"{spec.name}_rule",
            premise_claim_ids=(premise.claim_id,),
            conclusion_claim_id=claim.claim_id,
            declared_confidence=0.55 if depth == ExecutionTier.PROBE else 0.75,
            normative_framework_id=_framework(spec),
            provenance=provenance,
        )
        assumptions = _assumptions(spec, inquiry, provenance)
        commitments = _commitments(spec, claim, provenance)
        propositions = (claim, premise)
        return EpistemicStance(
            stance_id=artifact_id("sid", spec.cluster_id, spec.method_id, inquiry),
            cluster_id=spec.cluster_id,
            method_id=spec.method_id,
            conclusion=claim,
            propositions=propositions,
            warrants=(warrant,),
            assumptions=assumptions,
            normative_commitments=commitments,
            internal_confidence=warrant.declared_confidence,
            semantic_vector=_hash_vector(inquiry, spec.cluster_id, spec.method_id),
            provenance=provenance,
        )


class ClusterGroup:
    def __init__(self, evaluators: Sequence[DeterministicMethodEvaluator]):
        self.evaluators = list(evaluators)

    async def evaluate(self, inquiry: str, *, depth: ExecutionTier) -> EpistemicStance:
        selected = methods_for(self.evaluators[0].spec.cluster_id, depth)
        primary = next(item for item in self.evaluators if item.spec.method_id == selected[0].method_id)
        return await primary.evaluate(inquiry, depth=depth)

    async def evaluate_all(self, inquiry: str, *, depth: ExecutionTier) -> List[EpistemicStance]:
        allowed = {item.method_id for item in methods_for(self.evaluators[0].spec.cluster_id, depth)}
        selected = [item for item in self.evaluators if item.spec.method_id in allowed]
        return list(await asyncio.gather(*[item.evaluate(inquiry, depth=depth) for item in selected]))


def default_cluster_registry() -> Dict[str, ClusterGroup]:
    registry: Dict[str, ClusterGroup] = {}
    for cluster_id, specs in METHOD_CATALOG.items():
        registry[cluster_id] = ClusterGroup([DeterministicMethodEvaluator(spec) for spec in specs])
    return registry


def _framework(spec: MethodSpec) -> str | None:
    if spec.method_id == "6.1":
        return "deontology"
    if spec.method_id == "6.2":
        return "utilitarian"
    return None


def _assumptions(spec: MethodSpec, inquiry: str, provenance) -> tuple[Assumption, ...]:
    if spec.method_id == "3.1":
        value = 0.30
        name = "implementation_cost"
    elif spec.method_id == "3.2":
        value = 0.35
        name = "implementation_cost"
    elif spec.method_id == "5.1":
        value = 0.80
        name = "implementation_cost"
    elif spec.method_id == "5.2":
        value = 0.85
        name = "implementation_cost"
    else:
        return ()
    return (
        Assumption(
            assumption_id=artifact_id("assump", spec.method_id, inquiry),
            parameter_name=name,
            assumed_value=value,
            range_lower=0.0,
            range_upper=1.0,
            provenance=provenance,
        ),
    )


def _commitments(spec: MethodSpec, claim: StructuredClaim, provenance) -> tuple[NormativeCommitment, ...]:
    if spec.method_id == "6.1":
        return (
            NormativeCommitment(
                axiom_name="NonMaleficence",
                framework_id="deontology",
                priority_rank=1,
                obligatory_claims=(claim,),
                prohibited_claims=(),
                provenance=provenance,
            ),
        )
    if spec.method_id == "6.2":
        return (
            NormativeCommitment(
                axiom_name="MaxBeneficence",
                framework_id="utilitarian",
                priority_rank=1,
                obligatory_claims=(claim,),
                prohibited_claims=(),
                provenance=provenance,
            ),
        )
    return ()


def _hash_vector(inquiry: str, cluster_id: str, method_id: str, dim: int = 8) -> tuple[float, ...]:
    seed = hashlib.sha256(f"{cluster_id}:{method_id}:{inquiry}".encode("utf-8")).digest()
    values = [(seed[index] - 127.5) / 127.5 for index in range(dim)]
    norm = sum(item * item for item in values) ** 0.5 or 1.0
    return tuple(item / norm for item in values)
