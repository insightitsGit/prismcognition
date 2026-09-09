from __future__ import annotations

from typing import Dict, List, Tuple

from prismcognition.schemas.core import ExecutionTier, GroundingRegime, ImmutableBase


class MethodSpec(ImmutableBase):
    cluster_id: str
    method_id: str
    name: str
    regime: GroundingRegime
    primary: bool = True


METHOD_CATALOG: Dict[str, Tuple[MethodSpec, ...]] = {
    "1": (
        MethodSpec(cluster_id="1", method_id="1.1", name="propositional_closure", regime=GroundingRegime.FORMAL, primary=True),
        MethodSpec(cluster_id="1", method_id="1.2", name="definitional_analysis", regime=GroundingRegime.DEFINITIONAL, primary=False),
    ),
    "2": (
        MethodSpec(cluster_id="2", method_id="2.1", name="observational_claim", regime=GroundingRegime.EMPIRICAL, primary=True),
        MethodSpec(cluster_id="2", method_id="2.2", name="measurement_bound", regime=GroundingRegime.EMPIRICAL, primary=False),
    ),
    "3": (
        MethodSpec(cluster_id="3", method_id="3.1", name="mechanism", regime=GroundingRegime.CAUSAL, primary=True),
        MethodSpec(cluster_id="3", method_id="3.2", name="intervention", regime=GroundingRegime.CAUSAL, primary=False),
    ),
    "4": (
        MethodSpec(cluster_id="4", method_id="4.1", name="opposing_counsel", regime=GroundingRegime.EMPIRICAL, primary=True),
        MethodSpec(cluster_id="4", method_id="4.2", name="incentive_failure", regime=GroundingRegime.EMPIRICAL, primary=False),
    ),
    "5": (
        MethodSpec(cluster_id="5", method_id="5.1", name="implementation_cost", regime=GroundingRegime.EMPIRICAL, primary=True),
        MethodSpec(cluster_id="5", method_id="5.2", name="institutional_constraint", regime=GroundingRegime.EMPIRICAL, primary=False),
    ),
    "6": (
        MethodSpec(cluster_id="6", method_id="6.1", name="deontology", regime=GroundingRegime.NORMATIVE, primary=True),
        MethodSpec(cluster_id="6", method_id="6.2", name="utilitarianism", regime=GroundingRegime.NORMATIVE, primary=False),
    ),
}


def methods_for(cluster_id: str, depth: ExecutionTier) -> List[MethodSpec]:
    catalog = list(METHOD_CATALOG.get(str(cluster_id), ()))
    if depth == ExecutionTier.PROBE:
        return [item for item in catalog if item.primary] or catalog[:1]
    return catalog
