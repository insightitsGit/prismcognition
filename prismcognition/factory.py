from __future__ import annotations

from typing import Optional

from prismcognition.adapters.deterministic import HashEmbedder, ScriptedChaosLLM
from prismcognition.adapters.features import FeatureClassifier
from prismcognition.adapters.openai_compat import OpenAICompatibleEmbedder, OpenAICompatibleLLM
from prismcognition.engine.catalog import METHOD_CATALOG
from prismcognition.engine.clusters import default_cluster_registry
from prismcognition.engine.extractor import StructuredClusterExtractor
from prismcognition.engine.hybrid import HybridClusterGroup
from prismcognition.engine.orchestrator import DEFAULT_THRESHOLDS, EpistemicOrchestrator
from prismcognition.engine.router import DepthAllocatingRouter
from prismcognition.evidence.store import EvidenceStore
from prismcognition.evidence.verifier import RegimeAwareGroundingVerifier
from prismcognition.persist.store import ArtifactStore
from prismcognition.settings import RuntimeSettings, load_settings
from prismcognition.subtractive.containment import ChaosRoomSandbox


def build_default_orchestrator(
    *,
    settings: Optional[RuntimeSettings] = None,
    evidence_store: Optional[EvidenceStore] = None,
    classifier=None,
    chaos_llm=None,
    embedder=None,
    clash_threshold: float = 0.35,
    live: Optional[bool] = None,
) -> EpistemicOrchestrator:
    runtime = settings or load_settings(live=live)
    store = evidence_store or _load_or_create_evidence(runtime)
    llm = chaos_llm
    vectorizer = embedder
    registry = default_cluster_registry()

    if runtime.live_llm and runtime.llm_api_key and llm is None:
        llm = OpenAICompatibleLLM(
            api_key=runtime.llm_api_key,
            base_url=runtime.llm_base_url,
            model=runtime.llm_model,
            timeout_s=runtime.request_timeout_s,
        )
        vectorizer = vectorizer or OpenAICompatibleEmbedder(
            api_key=runtime.llm_api_key,
            base_url=runtime.llm_base_url,
            model=runtime.embed_model,
            timeout_s=runtime.request_timeout_s,
        )
        live_registry = {}
        for cluster_id, group in registry.items():
            extractors = [
                StructuredClusterExtractor(llm, spec)
                for spec in METHOD_CATALOG.get(cluster_id, ())
            ]
            live_registry[cluster_id] = HybridClusterGroup(group, extractors)
        registry = live_registry

    return EpistemicOrchestrator(
        router=DepthAllocatingRouter(
            classifier or FeatureClassifier(),
            required_threshold=DEFAULT_THRESHOLDS["router_required"],
            high_risk_floor=DEFAULT_THRESHOLDS["high_risk_floor"],
        ),
        chaos_room=ChaosRoomSandbox(llm or ScriptedChaosLLM(), vectorizer or HashEmbedder()),
        cluster_registry=registry,
        grounding_verifier=RegimeAwareGroundingVerifier(store),
        clash_threshold=clash_threshold,
        threshold_config=dict(DEFAULT_THRESHOLDS),
    )


def build_artifact_store(settings: Optional[RuntimeSettings] = None) -> ArtifactStore:
    return ArtifactStore(settings or load_settings())


def _load_or_create_evidence(settings: RuntimeSettings) -> EvidenceStore:
    if settings.evidence_path.exists():
        return EvidenceStore.load_json(settings.evidence_path)
    store = EvidenceStore(snapshot_id=settings.evidence_path.stem)
    return store
