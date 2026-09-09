"""PrismCognition v2.1 — disagreement-preserving epistemic orchestrator."""

from prismcognition.factory import build_default_orchestrator
from prismcognition.schemas.core import (
    DeliberationArtifact,
    EpistemicStance,
    GroundingProfile,
    ProvenanceRef,
    RuinAnalysisResult,
)

__version__ = "2.1.0"

__all__ = [
    "DeliberationArtifact",
    "EpistemicStance",
    "GroundingProfile",
    "ProvenanceRef",
    "RuinAnalysisResult",
    "build_default_orchestrator",
    "__version__",
]
