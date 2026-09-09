from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from prismcognition.identity import artifact_id, derive_provenance
from prismcognition.schemas.core import (
    EpistemicStance,
    GeometricRuinStatus,
    RuinAnalysisResult,
    RuinAnalysisStatus,
)


class SubtractiveGate:
    def __init__(self, tol: float = 1e-6):
        self.tol = tol

    def apply_subtractive_gate(
        self,
        stance_pairs: List[Tuple[EpistemicStance, Optional[np.ndarray]]],
        ruin_result: RuinAnalysisResult,
    ) -> Tuple[List[Tuple[EpistemicStance, Optional[np.ndarray]]], List[str]]:
        """Predicate AST pruning, then SVD nullspace annotation only."""
        if ruin_result.status not in (RuinAnalysisStatus.COMPLETED, RuinAnalysisStatus.PARTIAL):
            return list(stance_pairs), [
                f"Subtractive geometry skipped: ruin analysis status is {ruin_result.status.value}."
            ]

        surviving_pairs: List[Tuple[EpistemicStance, Optional[np.ndarray]]] = []
        dropped_diagnostics: List[str] = []

        for stance, vec in stance_pairs:
            stance_preds = {proposition.domain_key: proposition.polarity for proposition in stance.propositions}
            eliminated = False
            for boundary in ruin_result.boundaries:
                triggers = boundary.trigger_predicates
                if triggers and all(stance_preds.get(trigger.domain_key) == trigger.polarity for trigger in triggers):
                    dropped_diagnostics.append(
                        f"Stance {stance.stance_id} eliminated: Triggered absorbing ruin boundary '{boundary.boundary_id}'"
                    )
                    eliminated = True
                    break
            if not eliminated:
                surviving_pairs.append((stance, vec))

        ruin_vectors = [boundary.semantic_vector for boundary in ruin_result.boundaries if boundary.semantic_vector is not None]
        if not ruin_vectors or not surviving_pairs:
            return surviving_pairs, dropped_diagnostics

        widths = {len(vector) for vector in ruin_vectors}
        if len(widths) != 1:
            dropped_diagnostics.append("SVD skipped: ruin embedding dimensions are inconsistent.")
            return surviving_pairs, dropped_diagnostics

        try:
            matrix = np.array(ruin_vectors, dtype=float).T
        except ValueError:
            dropped_diagnostics.append("SVD skipped: ruin vectors could not form a matrix.")
            return surviving_pairs, dropped_diagnostics

        basis, diagnostics = self._ruin_basis(matrix)
        dropped_diagnostics.extend(diagnostics)
        projected: List[Tuple[EpistemicStance, Optional[np.ndarray]]] = []

        for stance, vec in surviving_pairs:
            if vec is None:
                projected.append((stance, None))
                continue
            vector = np.asarray(vec, dtype=float).reshape(-1)
            if vector.size != matrix.shape[0]:
                dropped_diagnostics.append(
                    f"Stance {stance.stance_id}: embedding dimension {vector.size} "
                    f"!= ruin subspace {matrix.shape[0]}; geometry left NOT_CALCULATED."
                )
                projected.append((stance, vec))
                continue

            safe = vector - basis @ (basis.T @ vector)
            norm = float(np.linalg.norm(safe))
            if norm < self.tol:
                projected.append((self._annotate(stance, None, GeometricRuinStatus.GEOMETRICALLY_COLLAPSED), None))
            else:
                unit = safe / norm
                projected.append(
                    (
                        self._annotate(stance, tuple(float(x) for x in unit.tolist()), GeometricRuinStatus.PRESERVED),
                        unit,
                    )
                )

        return projected, dropped_diagnostics

    def _ruin_basis(self, matrix: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        if matrix.size == 0:
            return np.empty((matrix.shape[0], 0)), []
        try:
            left, singular_values, _vh = np.linalg.svd(matrix, full_matrices=False)
        except np.linalg.LinAlgError:
            return np.empty((matrix.shape[0], 0)), ["SVD failed; geometric annotation skipped."]
        if singular_values.size == 0 or float(singular_values[0]) <= 0.0:
            return np.empty((matrix.shape[0], 0)), []
        rank = int(np.sum(singular_values > self.tol * singular_values[0]))
        return left[:, :rank], []

    def _annotate(
        self,
        stance: EpistemicStance,
        orthogonal: Optional[Tuple[float, ...]],
        status: GeometricRuinStatus,
    ) -> EpistemicStance:
        derived = derive_provenance(
            stance.provenance,
            artifact_id_value=artifact_id("stance_geo", stance.stance_id, status.value),
            model_provider="internal",
            model_id="subtractive_gate",
            prompt_hash="deterministic-internal",
        )
        return stance.model_copy(
            update={
                "ruin_orthogonal_vector": orthogonal,
                "ruin_geometry_status": status,
                "provenance": derived,
            }
        )
