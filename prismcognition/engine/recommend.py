from __future__ import annotations

from prismcognition.schemas.core import DeliberationArtifact, RuinAnalysisStatus


def draft_optional_recommendation(artifact: DeliberationArtifact) -> str:
    """Subordinate action note. Never claims consensus or absence of ruin."""
    lines = [
        "Optional action note (not a verdict). The deliberation artifact remains the deliverable.",
    ]
    if artifact.ruin_analysis.status in (RuinAnalysisStatus.FAILED, RuinAnalysisStatus.TIMEOUT):
        lines.append(
            f"Ruin analysis is {artifact.ruin_analysis.status.value}; do not treat missing boundaries as safety."
        )
    elif artifact.ruin_analysis.boundaries:
        lines.append(
            f"{len(artifact.ruin_analysis.boundaries)} absorbing ruin boundary(ies) were isolated; "
            "any matching trigger set is a hard stop, not a tradeoff."
        )
    if artifact.irreducible_tensions:
        lines.append("Irreducible tensions remain and are valid terminal states:")
        lines.extend(f"- {item}" for item in artifact.irreducible_tensions)
    if artifact.evidence_needed:
        lines.append("Do not decide as if the following absences were refutations:")
        lines.extend(f"- {item}" for item in artifact.evidence_needed)
    if artifact.active_disagreements and not artifact.irreducible_tensions:
        lines.append(
            f"{len(artifact.active_disagreements)} material disagreement(s) are still open; "
            "do not compress them into a single go/no-go scalar."
        )
    if artifact.strongly_supported_claims:
        lines.append(
            f"{len(artifact.strongly_supported_claims)} claim(s) meet the strong-support rule; "
            "that is not license to ignore remaining friction."
        )
    if artifact.methods_failed:
        lines.append(f"Failed nodes: {', '.join(artifact.methods_failed)}.")
    if len(lines) == 1:
        lines.append("No additional action is implied.")
    return "\n".join(lines)
