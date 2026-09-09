"""Stable identifiers and derived provenance (Invariant 7, 10)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from prismcognition.schemas.core import ProvenanceRef

ENGINE_VERSION = "2.1.0-final-freeze"
SCHEMA_VERSION = "2.1"
FLOAT_QUANTUM = 8


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_hash(*parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def artifact_id(prefix: str, *parts: Any) -> str:
    return f"{prefix}_{stable_hash(*parts)[:16]}"


def prompt_digest(payload: Any) -> str:
    if isinstance(payload, bytes):
        raw = payload
    elif isinstance(payload, str):
        raw = payload.encode("utf-8")
    else:
        raw = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def derive_provenance(
    parent: ProvenanceRef,
    *,
    artifact_id_value: str,
    model_provider: Optional[str] = None,
    model_id: Optional[str] = None,
    prompt_hash: Optional[str] = None,
    created_at: Optional[str] = None,
    extra_parents: Iterable[str] = (),
) -> ProvenanceRef:
    parents = tuple(parent.parent_ids) + (parent.artifact_id,) + tuple(extra_parents)
    return ProvenanceRef(
        artifact_id=artifact_id_value,
        parent_ids=parents,
        engine_version=ENGINE_VERSION,
        schema_version=SCHEMA_VERSION,
        model_provider=model_provider or parent.model_provider,
        model_id=model_id or parent.model_id,
        prompt_hash=prompt_hash or parent.prompt_hash,
        created_at=created_at or utc_now(),
        evidence_snapshot_id=parent.evidence_snapshot_id,
    )


def new_provenance(
    *,
    artifact_id_value: str,
    model_provider: str,
    model_id: str,
    prompt_hash: str,
    created_at: Optional[str] = None,
    parent_ids: tuple[str, ...] = (),
    evidence_snapshot_id: Optional[str] = None,
) -> ProvenanceRef:
    return ProvenanceRef(
        artifact_id=artifact_id_value,
        parent_ids=parent_ids,
        engine_version=ENGINE_VERSION,
        schema_version=SCHEMA_VERSION,
        model_provider=model_provider,
        model_id=model_id,
        prompt_hash=prompt_hash,
        created_at=created_at or utc_now(),
        evidence_snapshot_id=evidence_snapshot_id,
    )


def quantize_float(value: float) -> float:
    return float(round(float(value), FLOAT_QUANTUM))
