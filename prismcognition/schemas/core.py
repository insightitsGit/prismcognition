from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field


class ImmutableBase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)


class Polarity(int, Enum):
    POSITIVE = 1
    NEUTRAL = 0
    NEGATIVE = -1


class GroundingRegime(str, Enum):
    EMPIRICAL = "EMPIRICAL"
    FORMAL = "FORMAL"
    CAUSAL = "CAUSAL"
    NORMATIVE = "NORMATIVE"
    INTERPRETIVE = "INTERPRETIVE"
    PHENOMENOLOGICAL = "PHENOMENOLOGICAL"
    DEFINITIONAL = "DEFINITIONAL"
    UNVERIFIABLE = "UNVERIFIABLE"


class GroundingStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    CONTESTED = "CONTESTED"
    UNRESOLVED = "UNRESOLVED"
    NOT_GROUNDABLE = "NOT_GROUNDABLE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class ResolutionStatus(str, Enum):
    NO_MATERIAL_DISAGREEMENT = "no_material_disagreement"
    RESOLVED_BY_EVIDENCE = "resolved_by_evidence"
    CONDITIONALLY_RESOLVABLE = "conditionally_resolvable"
    CURRENTLY_UNRESOLVED = "currently_unresolved"
    NORMATIVELY_IRREDUCIBLE = "normatively_irreducible"
    AXIOMATICALLY_IRREDUCIBLE = "axiomatically_irreducible"
    DEFINITIONALLY_IRREDUCIBLE = "definitionally_irreducible"
    INSUFFICIENT_EPISTEMIC_ACCESS = "insufficient_epistemic_access"


class GeometricRuinStatus(str, Enum):
    PRESERVED = "PRESERVED"
    GEOMETRICALLY_COLLAPSED = "GEOMETRICALLY_COLLAPSED"
    NOT_CALCULATED = "NOT_CALCULATED"


class RuinAnalysisStatus(str, Enum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"


class ExecutionTier(str, Enum):
    REQUIRED = "REQUIRED"
    PROBE = "PROBE"
    SKIPPED = "SKIPPED"


class AllocationScoreSource(str, Enum):
    CLASSIFIER = "classifier"
    MISSING_COMPUTE_DEFAULT = "missing_compute_default"
    HIGH_RISK_FLOOR = "high_risk_floor"
    STRUCTURAL_SKIP = "structural_skip"
    MANDATORY = "mandatory"


class ProvenanceRef(ImmutableBase):
    artifact_id: str
    parent_ids: Tuple[str, ...] = ()
    engine_version: str = "2.1.0-final-freeze"
    schema_version: str = "2.1"
    model_provider: str
    model_id: str
    prompt_hash: str
    created_at: str
    evidence_snapshot_id: Optional[str] = None


def default_provenance() -> ProvenanceRef:
    created = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return ProvenanceRef(
        artifact_id=f"prov_auto_{uuid.uuid4().hex[:16]}",
        model_provider="internal",
        model_id="schema_default",
        prompt_hash="deterministic-internal",
        created_at=created,
    )


class StructuredClaim(ImmutableBase):
    claim_id: str
    subject: str
    predicate: str
    target_object: str
    polarity: Polarity
    domain_key: str
    raw_statement: str
    provenance: ProvenanceRef


class Warrant(ImmutableBase):
    warrant_id: str
    epistemic_regime: GroundingRegime
    rule_statement: str
    premise_claim_ids: Tuple[str, ...]
    conclusion_claim_id: str
    declared_confidence: float = Field(ge=0.0, le=1.0)
    normative_framework_id: Optional[str] = None
    provenance: ProvenanceRef


class WarrantGrounding(ImmutableBase):
    warrant_id: str
    epistemic_regime: GroundingRegime
    status: GroundingStatus
    support_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    evidence_refs: Tuple[str, ...] = ()
    counterevidence_refs: Tuple[str, ...] = ()
    epistemic_reasoning: str
    provenance: ProvenanceRef


class Assumption(ImmutableBase):
    assumption_id: str
    parameter_name: str
    assumed_value: Union[bool, float, str]
    range_lower: Optional[float] = None
    range_upper: Optional[float] = None
    provenance: ProvenanceRef


class NormativeCommitment(ImmutableBase):
    axiom_name: str
    framework_id: str
    priority_rank: int
    obligatory_claims: Tuple[StructuredClaim, ...] = ()
    prohibited_claims: Tuple[StructuredClaim, ...] = ()
    provenance: ProvenanceRef


class EpistemicStance(ImmutableBase):
    stance_id: str
    cluster_id: str
    method_id: str
    conclusion: StructuredClaim
    propositions: Tuple[StructuredClaim, ...]
    warrants: Tuple[Warrant, ...]
    assumptions: Tuple[Assumption, ...]
    normative_commitments: Tuple[NormativeCommitment, ...]
    internal_confidence: float = Field(ge=0.0, le=1.0)
    semantic_vector: Optional[Tuple[float, ...]] = None
    ruin_orthogonal_vector: Optional[Tuple[float, ...]] = None
    ruin_geometry_status: GeometricRuinStatus = GeometricRuinStatus.NOT_CALCULATED
    provenance: ProvenanceRef


class RegimeAssessment(ImmutableBase):
    regime: GroundingRegime
    score: Optional[float] = None
    applicable: bool = True
    supported_count: int = 0
    contradicted_count: int = 0
    contested_count: int = 0
    unresolved_count: int = 0
    not_groundable_count: int = 0
    insufficient_evidence_count: int = 0


class GroundingProfile(ImmutableBase):
    regime_assessments: Tuple[RegimeAssessment, ...]
    provenance: ProvenanceRef = Field(default_factory=default_provenance)

    def get_assessment(self, regime: GroundingRegime) -> Optional[RegimeAssessment]:
        for item in self.regime_assessments:
            if item.regime == regime:
                return item
        return None


class EpistemicClash(ImmutableBase):
    clash_id: str
    clash_type: Literal[
        "FACTUAL_CONFLICT",
        "INFERENCE_RULE_CONFLICT",
        "ASSUMPTION_CONFLICT",
        "NORMATIVE_CONFLICT",
        "CAUSAL_CONFLICT",
        "DEFINITIONAL_CONFLICT",
    ]
    clash_regime: GroundingRegime
    stance_id_left: str
    stance_id_right: str
    explanation: str
    provenance: ProvenanceRef


class PerspectiveDiversity(ImmutableBase):
    diversity_id: str
    diversity_type: Literal[
        "NON_OVERLAPPING_WARRANTS",
        "NORMATIVE_FRAMEWORK_DIVERSITY",
        "METHODOLOGICAL_COMPLEMENT",
    ]
    stance_id_left: str
    stance_id_right: str
    explanation: str
    provenance: ProvenanceRef


class AssumptionMidpoint(ImmutableBase):
    variable_name: str
    disputing_clusters: Tuple[str, str]
    values: Tuple[float, float]
    midpoint: float
    provenance: ProvenanceRef = Field(default_factory=default_provenance)


class EpistemicPivot(ImmutableBase):
    variable_name: str
    disputing_clusters: Tuple[str, str]
    current_assumptions: Tuple[float, float]
    pivot_threshold: float
    verification_source: str
    derivation_trace: str
    provenance: ProvenanceRef = Field(default_factory=default_provenance)


class TensionProfile(ImmutableBase):
    empirical: Optional[float] = None
    formal: Optional[float] = None
    causal: Optional[float] = None
    assumption: Optional[float] = None
    normative: Optional[float] = None
    aggregate_structural_tension: float
    provenance: ProvenanceRef = Field(default_factory=default_provenance)


class DisagreementArtifact(ImmutableBase):
    disagreement_id: str
    clash: EpistemicClash
    raw_tension: float
    calibrated_tension: Optional[float]
    tension_profile: TensionProfile
    profile_left: GroundingProfile
    profile_right: GroundingProfile
    resolution_status: ResolutionStatus
    midpoint: Optional[AssumptionMidpoint] = None
    validated_pivot: Optional[EpistemicPivot] = None
    provenance: ProvenanceRef = Field(default_factory=default_provenance)


class RuinBoundary(ImmutableBase):
    boundary_id: str
    failure_state: StructuredClaim
    trigger_predicates: Tuple[StructuredClaim, ...]
    reversibility: float = Field(ge=0.0, le=1.0)
    severity: float = Field(ge=0.0, le=1.0)
    semantic_vector: Optional[Tuple[float, ...]] = None
    provenance: ProvenanceRef


class RuinAnalysisResult(ImmutableBase):
    status: RuinAnalysisStatus
    boundaries: Tuple[RuinBoundary, ...]
    diagnostics: Tuple[str, ...]
    execution_time_ms: float
    provenance: ProvenanceRef


class InquiryFeatures(ImmutableBase):
    formal_a_priori: bool = False
    physical_constant_query: bool = False
    empirical_measurement: bool = False
    normative_judgment: bool = False
    causal_mechanism: bool = False
    strategic_decision: bool = False
    token_set: Tuple[str, ...] = ()


class ClusterRoute(ImmutableBase):
    cluster_id: int
    tier: ExecutionTier
    justification: str
    classifier_score: Optional[float] = None
    allocated_score: Optional[float] = None
    score_source: AllocationScoreSource = AllocationScoreSource.CLASSIFIER


class RoutePlan(ImmutableBase):
    routes: Tuple[ClusterRoute, ...]
    inquiry_features: Optional[InquiryFeatures] = None


class CoverageReport(ImmutableBase):
    executed_cluster_ids: Tuple[str, ...]
    probed_cluster_ids: Tuple[str, ...]
    skipped_cluster_ids: Tuple[str, ...]
    failed_cluster_ids: Tuple[str, ...]
    executed_method_ids: Tuple[str, ...] = ()
    probed_method_ids: Tuple[str, ...] = ()
    failed_method_ids: Tuple[str, ...] = ()
    diversity_types: Tuple[str, ...]
    coverage_index: float
    explanation: str
    provenance: ProvenanceRef = Field(default_factory=default_provenance)


class DeliberationArtifact(ImmutableBase):
    deliberation_id: str
    inquiry_text: str
    methods_executed: Tuple[str, ...]
    methods_probed: Tuple[str, ...]
    methods_skipped: Tuple[str, ...]
    methods_failed: Tuple[str, ...]
    strongly_supported_claims: Tuple[StructuredClaim, ...]
    ruin_analysis: RuinAnalysisResult
    active_disagreements: Tuple[DisagreementArtifact, ...]
    perspective_diversities: Tuple[PerspectiveDiversity, ...]
    evidence_needed: Tuple[str, ...]
    assumptions_that_matter: Tuple[Assumption, ...]
    irreducible_tensions: Tuple[str, ...]
    optional_recommendation: Optional[str] = None
    route_plan: Optional[RoutePlan] = None
    coverage: Optional[CoverageReport] = None
    provenance: ProvenanceRef


class EvidenceRecord(ImmutableBase):
    record_id: str
    snapshot_id: str
    domain_key: str
    polarity: Polarity
    regime: GroundingRegime
    status: GroundingStatus
    support_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    source_ref: str


class FrozenDeliberationBundle(ImmutableBase):
    deliberation_id: str
    inquiry_text: str
    frozen_stances: Tuple[EpistemicStance, ...]
    frozen_ruin_result: RuinAnalysisResult
    frozen_grounding_profiles: Dict[str, GroundingProfile]
    frozen_warrant_groundings: Dict[str, Tuple[WarrantGrounding, ...]] = Field(default_factory=dict)
    methods_executed: Tuple[str, ...]
    methods_probed: Tuple[str, ...]
    methods_skipped: Tuple[str, ...]
    methods_failed: Tuple[str, ...]
    evidence_needed: Tuple[str, ...]
    irreducible_tensions: Tuple[str, ...]
    optional_recommendation: Optional[str] = None
    threshold_config: Dict[str, float]
    frozen_route_plan: Optional[RoutePlan] = None
    provenance: ProvenanceRef


class ChaosExtractionEnvelope(ImmutableBase):
    """Structured extract only. Raw Chaos Room prose is never stored here."""

    session_id: str
    raw_response_digest: str
    ruin_primitives: Tuple[Dict[str, Any], ...]


def iter_stance_claims(stance: EpistemicStance):
    yield stance.conclusion
    yield from stance.propositions
    for commitment in stance.normative_commitments:
        yield from commitment.obligatory_claims
        yield from commitment.prohibited_claims


def index_claims(stances: List[EpistemicStance]) -> Dict[str, StructuredClaim]:
    universe: Dict[str, StructuredClaim] = {}
    for stance in stances:
        for claim in iter_stance_claims(stance):
            universe[claim.claim_id] = claim
    return universe
