from __future__ import annotations

import html
from typing import Literal, Optional

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator

from prismcognition.evidence.store import EvidenceStore
from prismcognition.factory import build_artifact_store, build_default_orchestrator
from prismcognition.render.artifact import render_deliberation
from prismcognition.replay.engine import ReplayEngine
from prismcognition.schemas.core import EvidenceRecord, GroundingRegime, GroundingStatus, Polarity
from prismcognition.settings import LIVE_MODE_UNAVAILABLE, LiveModeUnavailableError, RuntimeSettings, load_settings


class DeliberateRequest(BaseModel):
    inquiry: str = Field(min_length=1, max_length=20000)
    risk_level: Literal["HIGH", "LOW"] = "HIGH"
    emit_recommendation: bool = False

    @field_validator("inquiry")
    @classmethod
    def nonblank_inquiry(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("inquiry must not be blank")
        return value


class EvidenceIn(BaseModel):
    record_id: str = Field(min_length=1, max_length=256)
    domain_key: str = Field(min_length=1, max_length=1024)
    polarity: Polarity
    regime: GroundingRegime
    status: GroundingStatus = GroundingStatus.SUPPORTED
    support_score: Optional[float] = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    source_ref: str = Field(min_length=1, max_length=4096)


def create_app(settings: Optional[RuntimeSettings] = None) -> FastAPI:
    runtime = settings or load_settings()
    if runtime.live_requested and not runtime.llm_api_key:
        raise LiveModeUnavailableError(LIVE_MODE_UNAVAILABLE)
    app = FastAPI(title="PrismCognition", version="2.1.0")
    app.state.settings = runtime

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "live_requested": runtime.live_requested,
            "live_llm": runtime.live_llm,
            "live_ready": bool(runtime.live_llm and runtime.llm_api_key),
        }

    @app.get("/", response_class=HTMLResponse)
    async def home():
        return HTML_PAGE

    @app.post("/api/deliberations")
    async def create_deliberation(payload: DeliberateRequest):
        orchestrator = build_default_orchestrator(settings=runtime)
        artifact = await orchestrator.deliberate(
            payload.inquiry,
            risk_level=payload.risk_level,
            emit_recommendation=payload.emit_recommendation,
        )
        store = build_artifact_store(runtime)
        store.save_artifact(artifact)
        store.save_bundle(orchestrator.last_frozen_bundle())
        return {
            "artifact": artifact.model_dump(mode="json"),
            "rendered": render_deliberation(artifact),
        }

    @app.post("/deliberate", response_class=HTMLResponse)
    async def deliberate_form(
        inquiry: str = Form(..., min_length=1, max_length=20000, pattern=r"\S"),
        risk_level: Literal["HIGH", "LOW"] = Form("HIGH"),
        emit_recommendation: str = Form(""),
    ):
        orchestrator = build_default_orchestrator(settings=runtime)
        artifact = await orchestrator.deliberate(
            inquiry,
            risk_level=risk_level,
            emit_recommendation=bool(emit_recommendation),
        )
        store = build_artifact_store(runtime)
        store.save_artifact(artifact)
        store.save_bundle(orchestrator.last_frozen_bundle())
        rendered = html.escape(render_deliberation(artifact))
        return HTML_PAGE.replace(
            "<!--RESULT-->",
            f"<section class='result'><h2>Deliberation</h2><pre>{rendered}</pre></section>",
        )

    @app.get("/api/deliberations")
    async def list_deliberations():
        return {"ids": build_artifact_store(runtime).list_deliberation_ids()}

    @app.get("/api/deliberations/{deliberation_id}")
    async def get_deliberation(deliberation_id: str):
        try:
            artifact = build_artifact_store(runtime).load_artifact(deliberation_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="deliberation not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid deliberation or stored artifact") from exc
        return {"artifact": artifact.model_dump(mode="json"), "rendered": render_deliberation(artifact)}

    @app.post("/api/deliberations/{deliberation_id}/replay")
    async def replay_deliberation(deliberation_id: str):
        store = build_artifact_store(runtime)
        try:
            bundle = store.load_bundle(deliberation_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="bundle not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid deliberation or stored bundle") from exc
        replayed = ReplayEngine.replay(bundle)
        store.save_artifact(replayed)
        return {"artifact": replayed.model_dump(mode="json"), "rendered": render_deliberation(replayed)}

    @app.post("/api/evidence")
    async def ingest_evidence(payload: EvidenceIn):
        store = (
            EvidenceStore.load_json(runtime.evidence_path)
            if runtime.evidence_path.exists()
            else EvidenceStore(snapshot_id="runtime")
        )
        store.ingest(
            EvidenceRecord(
                record_id=payload.record_id,
                snapshot_id=store.snapshot_id,
                domain_key=payload.domain_key,
                polarity=Polarity(payload.polarity),
                regime=GroundingRegime(payload.regime),
                status=GroundingStatus(payload.status),
                support_score=payload.support_score,
                source_ref=payload.source_ref,
            )
        )
        runtime.evidence_path.parent.mkdir(parents=True, exist_ok=True)
        store.save_json(runtime.evidence_path)
        return {"ok": True, "path": str(runtime.evidence_path)}

    return app


def serve(host: str = "127.0.0.1", port: int = 8765, data_dir: Optional[str] = None, live: bool = False) -> None:
    import uvicorn

    settings = load_settings(data_dir=data_dir, live=live)
    uvicorn.run(create_app(settings), host=host, port=port, log_level="info")


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>PrismCognition</title>
  <style>
    body { font-family: Georgia, serif; margin: 2rem auto; max-width: 880px; color: #111; }
    textarea, select, button { font: inherit; width: 100%; }
    textarea { min-height: 7rem; }
    label { display: block; margin: 0.8rem 0 0.3rem; }
    button { margin-top: 1rem; padding: 0.6rem 1rem; }
    pre { white-space: pre-wrap; background: #f4f1ea; padding: 1rem; }
    .note { color: #444; }
  </style>
</head>
<body>
  <h1>PrismCognition</h1>
  <p class="note">Disagreement-preserving deliberation. Recommendations are optional and never a verdict.</p>
  <form method="post" action="/deliberate">
    <label for="inquiry">Inquiry</label>
    <textarea id="inquiry" name="inquiry" required placeholder="Should we expand the plant this quarter?"></textarea>
    <label for="risk_level">Risk level</label>
    <select id="risk_level" name="risk_level">
      <option value="HIGH">HIGH</option>
      <option value="LOW">LOW</option>
    </select>
    <label><input type="checkbox" name="emit_recommendation" value="1"/> Attach optional action note</label>
    <button type="submit">Deliberate</button>
  </form>
  <!--RESULT-->
</body>
</html>
"""
