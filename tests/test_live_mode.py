from __future__ import annotations

import pytest

from prismcognition.cli import main
from prismcognition.factory import build_default_orchestrator
from prismcognition.render.artifact import render_deliberation
from prismcognition.settings import LIVE_MODE_UNAVAILABLE, LiveModeUnavailableError, load_settings


@pytest.fixture
def no_provider_keys(monkeypatch):
    monkeypatch.delenv("PRISM_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PRISM_LIVE_LLM", raising=False)


def test_live_request_without_key_stays_explicit(no_provider_keys):
    settings = load_settings(live=True)
    assert settings.live_requested is True
    assert settings.live_llm is False
    assert settings.llm_api_key is None


def test_factory_refuses_live_without_credentials(no_provider_keys):
    with pytest.raises(LiveModeUnavailableError, match="LIVE MODE WAS REQUESTED"):
        build_default_orchestrator(live=True)


def test_cli_live_without_key_does_not_run_offline(tmp_path, capsys, no_provider_keys):
    assert main(["deliberate", "Should we expand the plant?", "--live", "--data-dir", str(tmp_path)]) == 2
    captured = capsys.readouterr()
    assert "LIVE MODE WAS REQUESTED BUT NEVER ACTIVATED" in captured.err
    assert "PRISM_LLM_API_KEY" in captured.err
    assert captured.out == ""
    assert not list(tmp_path.glob("deliberations/*.json"))


def test_cli_shorthand_live_without_key_is_refused(tmp_path, capsys, no_provider_keys):
    assert main(["Should we expand the plant?", "--live", "--data-dir", str(tmp_path)]) == 2
    assert "NEVER ACTIVATED" in capsys.readouterr().err


def test_blank_api_key_does_not_count_as_configured(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("PRISM_LLM_API_KEY", "   ")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main(["deliberate", "q", "--live", "--data-dir", str(tmp_path)]) == 2
    assert "NEVER ACTIVATED" in capsys.readouterr().err


def test_offline_renderer_names_execution_mode(tmp_path):
    from prismcognition.engine.orchestrator import EpistemicOrchestrator
    from prismcognition.engine.router import DepthAllocatingRouter
    from prismcognition.adapters.deterministic import HashEmbedder, ScriptedChaosLLM, UniformClassifier
    from prismcognition.engine.clusters import default_cluster_registry
    from prismcognition.subtractive.containment import ChaosRoomSandbox
    import asyncio

    orchestrator = EpistemicOrchestrator(
        router=DepthAllocatingRouter(UniformClassifier(0.05)),
        chaos_room=ChaosRoomSandbox(ScriptedChaosLLM(), HashEmbedder()),
        cluster_registry=default_cluster_registry(),
    )
    artifact = asyncio.run(orchestrator.deliberate("Measure rainfall", risk_level="LOW"))
    rendered = render_deliberation(artifact)
    assert "Execution mode: OFFLINE" in rendered
    assert "not live model output" in rendered
    assert artifact.execution_mode == "offline"


def test_create_app_refuses_live_without_key(no_provider_keys):
    from prismcognition.api.app import create_app

    with pytest.raises(LiveModeUnavailableError, match="LIVE MODE WAS REQUESTED"):
        create_app(load_settings(live=True))


def test_unavailable_message_is_actionable():
    assert "omit --live" in LIVE_MODE_UNAVAILABLE
    assert "OPENAI_API_KEY" in LIVE_MODE_UNAVAILABLE
