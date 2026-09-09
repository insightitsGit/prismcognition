from __future__ import annotations

from typing import Iterable, Tuple

from prismcognition.identity import artifact_id, derive_provenance
from prismcognition.schemas.core import CoverageReport, DeliberationArtifact


def compile_coverage(artifact: DeliberationArtifact) -> CoverageReport:
    executed_methods = tuple(artifact.methods_executed)
    probed_methods = tuple(artifact.methods_probed)
    failed_methods = tuple(artifact.methods_failed)
    skipped_clusters = tuple(_cluster_id(label) for label in artifact.methods_skipped)
    executed_clusters = _unique(_cluster_id(label) for label in executed_methods)
    probed_clusters = _unique(_cluster_id(label) for label in probed_methods)
    failed_clusters = _unique(_cluster_id(label) for label in failed_methods)
    diversity_types = tuple(sorted({item.diversity_type for item in artifact.perspective_diversities}))

    addressed = len(set(executed_methods) | set(probed_methods))
    withheld = len(set(skipped_clusters) | set(failed_methods))
    diversity_bonus = 0.05 * len(diversity_types)
    denominator = max(addressed + withheld, 1)
    index = min(1.0, (addressed / denominator) + diversity_bonus)

    ruin_note = artifact.ruin_analysis.status.value
    explanation = (
        f"Coverage is computed from executed/probed methods versus skipped/failed methods. "
        f"Diversity types {diversity_types or ('none',)} do not add adversarial tension. "
        f"Ruin analysis status={ruin_note}."
    )
    return CoverageReport(
        executed_cluster_ids=executed_clusters,
        probed_cluster_ids=probed_clusters,
        skipped_cluster_ids=skipped_clusters,
        failed_cluster_ids=failed_clusters,
        executed_method_ids=executed_methods,
        probed_method_ids=probed_methods,
        failed_method_ids=failed_methods,
        diversity_types=diversity_types,
        coverage_index=index,
        explanation=explanation,
        provenance=derive_provenance(
            artifact.provenance,
            artifact_id_value=artifact_id("coverage", artifact.deliberation_id),
            model_provider="internal",
            model_id="coverage",
            prompt_hash="deterministic-internal",
        ),
    )


def attach_coverage(artifact: DeliberationArtifact) -> DeliberationArtifact:
    return artifact.model_copy(update={"coverage": compile_coverage(artifact)})


def _cluster_id(label: str) -> str:
    head = label.split(":", 1)[0]
    return head.split("_", 1)[0]


def _unique(values: Iterable[str]) -> Tuple[str, ...]:
    seen = []
    for item in values:
        if item not in seen:
            seen.append(item)
    return tuple(seen)
