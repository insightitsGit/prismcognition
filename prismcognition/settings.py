from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class RuntimeSettings:
    data_dir: Path
    llm_api_key: Optional[str]
    llm_base_url: str
    llm_model: str
    embed_model: str
    live_llm: bool
    request_timeout_s: float

    @property
    def deliberations_dir(self) -> Path:
        return self.data_dir / "deliberations"

    @property
    def bundles_dir(self) -> Path:
        return self.data_dir / "bundles"

    @property
    def evidence_path(self) -> Path:
        return self.data_dir / "evidence.json"


def load_settings(
    *,
    data_dir: Optional[str] = None,
    live: Optional[bool] = None,
) -> RuntimeSettings:
    key = os.environ.get("PRISM_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    requested_live = live if live is not None else os.environ.get("PRISM_LIVE_LLM", "").lower() in {"1", "true", "yes"}
    root = Path(data_dir or os.environ.get("PRISM_DATA_DIR") or ".prismcognition")
    return RuntimeSettings(
        data_dir=root,
        llm_api_key=key,
        llm_base_url=os.environ.get("PRISM_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1",
        llm_model=os.environ.get("PRISM_LLM_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini",
        embed_model=os.environ.get("PRISM_EMBED_MODEL") or "text-embedding-3-small",
        live_llm=bool(requested_live and key),
        request_timeout_s=float(os.environ.get("PRISM_HTTP_TIMEOUT", "30")),
    )
