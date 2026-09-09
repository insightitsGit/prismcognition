from __future__ import annotations

from pathlib import Path
import re
from typing import List

from prismcognition.schemas.core import DeliberationArtifact, FrozenDeliberationBundle
from prismcognition.settings import RuntimeSettings
from prismcognition.persist.atomic import atomic_write_text


def _artifact_path(directory: Path, deliberation_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", deliberation_id):
        raise ValueError("invalid deliberation ID")
    path = directory / f"{deliberation_id}.json"
    if path.resolve().parent != directory.resolve():
        raise ValueError("artifact path escapes storage directory")
    return path


class ArtifactStore:
    def __init__(self, settings: RuntimeSettings):
        self.settings = settings
        settings.deliberations_dir.mkdir(parents=True, exist_ok=True)
        settings.bundles_dir.mkdir(parents=True, exist_ok=True)
        settings.data_dir.mkdir(parents=True, exist_ok=True)

    def save_artifact(self, artifact: DeliberationArtifact) -> Path:
        path = _artifact_path(self.settings.deliberations_dir, artifact.deliberation_id)
        atomic_write_text(path, artifact.model_dump_json(indent=2))
        return path

    def save_bundle(self, bundle: FrozenDeliberationBundle) -> Path:
        path = _artifact_path(self.settings.bundles_dir, bundle.deliberation_id)
        atomic_write_text(path, bundle.model_dump_json(indent=2))
        return path

    def load_artifact(self, deliberation_id: str) -> DeliberationArtifact:
        path = _artifact_path(self.settings.deliberations_dir, deliberation_id)
        return DeliberationArtifact.model_validate_json(path.read_text(encoding="utf-8"))

    def load_bundle(self, deliberation_id: str) -> FrozenDeliberationBundle:
        path = _artifact_path(self.settings.bundles_dir, deliberation_id)
        return FrozenDeliberationBundle.model_validate_json(path.read_text(encoding="utf-8"))

    def list_deliberation_ids(self) -> List[str]:
        return sorted(path.stem for path in self.settings.deliberations_dir.glob("*.json"))
