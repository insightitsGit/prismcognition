from __future__ import annotations

from typing import Optional

from prismcognition.engine.coverage import compile_coverage
from prismcognition.schemas.core import DeliberationArtifact


UNGROUNDED = "UNGROUNDED"


def render_score(score: Optional[float]) -> str:
    if score is None:
        return UNGROUNDED
    return f"{float(score):.4f}"


def render_deliberation(artifact: DeliberationArtifact) -> str:
    coverage = artifact.coverage or compile_coverage(artifact)
    lines = [
        f"Deliberation {artifact.deliberation_id}",
        f"Inquiry: {artifact.inquiry_text}",
        f"Executed: {', '.join(artifact.methods_executed) or 'none'}",
        f"Probed: {', '.join(artifact.methods_probed) or 'none'}",
        f"Skipped: {', '.join(artifact.methods_skipped) or 'none'}",
        f"Failed: {', '.join(artifact.methods_failed) or 'none'}",
        f"Coverage index: {render_score(coverage.coverage_index)}",
        f"Ruin status: {artifact.ruin_analysis.status.value}",
        f"Strongly supported claims: {len(artifact.strongly_supported_claims)}",
        f"Active disagreements: {len(artifact.active_disagreements)}",
        f"Perspective diversities: {len(artifact.perspective_diversities)}",
    ]
    if artifact.route_plan:
        missing = [
            str(route.cluster_id)
            for route in artifact.route_plan.routes
            if route.score_source.value == "missing_compute_default"
        ]
        if missing:
            lines.append(
                "Missing classifier keys used compute-default depth only "
                f"(not empty-cluster findings): {', '.join(missing)}."
            )
    if artifact.ruin_analysis.status.value in {"failed", "timeout"}:
        lines.append("Ruin analysis did not complete; empty boundaries are not a safety finding.")
    for disagreement in artifact.active_disagreements:
        lines.append(
            f"- {disagreement.clash.clash_type} raw={render_score(disagreement.raw_tension)} "
            f"calibrated={render_score(disagreement.calibrated_tension)} "
            f"status={disagreement.resolution_status.value}"
        )
    if artifact.evidence_needed:
        lines.append("Evidence needed:")
        lines.extend(f"  - {item}" for item in artifact.evidence_needed)
    if artifact.irreducible_tensions:
        lines.append("Irreducible tensions:")
        lines.extend(f"  - {item}" for item in artifact.irreducible_tensions)
    if artifact.optional_recommendation is None:
        lines.append("Optional recommendation: none (deliberation artifact is the deliverable).")
    else:
        lines.append(f"Optional recommendation: {artifact.optional_recommendation}")
    return "\n".join(lines)
