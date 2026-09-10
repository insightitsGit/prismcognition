from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


LIVE_MODE_UNAVAILABLE = (
    "LIVE MODE WAS REQUESTED BUT NEVER ACTIVATED.\n"
    "\n"
    "No provider credential is configured, so PrismCognition refused to run. "
    "This is not a live deliberation and is not the same as offline mode.\n"
    "\n"
    "Set PRISM_LLM_API_KEY or OPENAI_API_KEY, then retry --live "
    "(or PRISM_LIVE_LLM=1 / load_settings(live=True)).\n"
    "Or omit --live to run the deterministic adapters on purpose.\n"
)


class LiveModeUnavailableError(RuntimeError):
    """Live mode was requested but no provider credential is configured."""


@dataclass(frozen=True)
class RuntimeSettings:
    data_dir: Path
    llm_api_key: Optional[str]
    llm_base_url: str
    llm_model: str
    embed_model: str
    live_requested: bool
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

    def require_live_credentials(self) -> None:
        if self.live_requested and not self.llm_api_key:
            raise LiveModeUnavailableError(LIVE_MODE_UNAVAILABLE)


def load_settings(
    *,
    data_dir: Optional[str] = None,
    live: Optional[bool] = None,
) -> RuntimeSettings:
    key = _first_env("PRISM_LLM_API_KEY", "OPENAI_API_KEY")
    requested_live = live if live is not None else _env_flag("PRISM_LIVE_LLM")
    root = Path(data_dir or os.environ.get("PRISM_DATA_DIR") or ".prismcognition")
    return RuntimeSettings(
        data_dir=root,
        llm_api_key=key,
        llm_base_url=_first_env("PRISM_LLM_BASE_URL", "OPENAI_BASE_URL") or "https://api.openai.com/v1",
        llm_model=_first_env("PRISM_LLM_MODEL", "OPENAI_MODEL") or "gpt-4o-mini",
        embed_model=os.environ.get("PRISM_EMBED_MODEL") or "text-embedding-3-small",
        live_requested=bool(requested_live),
        live_llm=bool(requested_live and key),
        request_timeout_s=float(os.environ.get("PRISM_HTTP_TIMEOUT", "30")),
    )


def _first_env(*names: str) -> Optional[str]:
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes"}
