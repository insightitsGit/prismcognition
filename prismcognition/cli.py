from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Optional

from prismcognition.factory import build_artifact_store, build_default_orchestrator
from prismcognition.persist.store import ArtifactStore
from prismcognition.render.artifact import render_deliberation
from prismcognition.replay.engine import ReplayEngine
from prismcognition.schemas.core import EvidenceRecord, GroundingRegime, GroundingStatus, Polarity
from prismcognition.settings import LiveModeUnavailableError, load_settings


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="PrismCognition v2.1 deliberation engine.")
    parser.add_argument("--risk-level", default="HIGH", choices=("LOW", "HIGH"))
    parser.add_argument("--json", action="store_true", help="Emit artifact JSON instead of the renderer.")
    parser.add_argument("--recommend", action="store_true", help="Attach a subordinate optional action note.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Require a live provider. Refuses to run unless PRISM_LLM_API_KEY or OPENAI_API_KEY is set.",
    )
    parser.add_argument("--data-dir", default=None)

    # Parse options first to distinguish shorthand from an explicit command.
    # An optional positional before subparsers consumes the command name.
    arguments = list(sys.argv[1:] if argv is None else argv)
    commands = {"deliberate", "replay", "show", "list", "ingest-evidence", "serve"}
    # Do not consume --help here: let the final parser show its full help.
    if not any(item in {"-h", "--help"} for item in arguments):
        _, remaining = parser.parse_known_args(arguments)
        if remaining and remaining[0] not in commands:
            parser.add_argument("inquiry", help="Inquiry text (shorthand for deliberate).")
            args = parser.parse_args(arguments)
            return _deliberate(args.inquiry, risk_level=args.risk_level, as_json=args.json,
                               recommend=args.recommend, live=args.live, data_dir=args.data_dir)

    sub = parser.add_subparsers(dest="command")

    deliberate_cmd = sub.add_parser("deliberate", help="Run a deliberation.")
    deliberate_cmd.add_argument("inquiry")
    deliberate_cmd.add_argument("--risk-level", default="HIGH", choices=("LOW", "HIGH"))
    deliberate_cmd.add_argument("--json", action="store_true")
    deliberate_cmd.add_argument("--recommend", action="store_true")
    deliberate_cmd.add_argument("--live", action="store_true")
    deliberate_cmd.add_argument("--data-dir", default=None)

    replay_cmd = sub.add_parser("replay", help="Replay a frozen bundle by deliberation id.")
    replay_cmd.add_argument("deliberation_id")
    replay_cmd.add_argument("--json", action="store_true")
    replay_cmd.add_argument("--data-dir", default=None)

    show_cmd = sub.add_parser("show", help="Render a stored deliberation.")
    show_cmd.add_argument("deliberation_id")
    show_cmd.add_argument("--json", action="store_true")
    show_cmd.add_argument("--data-dir", default=None)

    list_cmd = sub.add_parser("list", help="List stored deliberation ids.")
    list_cmd.add_argument("--data-dir", default=None)

    evidence_cmd = sub.add_parser("ingest-evidence", help="Add a row to the evidence snapshot.")
    evidence_cmd.add_argument("--record-id", required=True)
    evidence_cmd.add_argument("--domain-key", required=True)
    evidence_cmd.add_argument("--polarity", required=True, type=int, choices=(-1, 0, 1))
    evidence_cmd.add_argument("--regime", required=True)
    evidence_cmd.add_argument("--status", default="SUPPORTED")
    evidence_cmd.add_argument("--score", type=float, default=None)
    evidence_cmd.add_argument("--source-ref", required=True)
    evidence_cmd.add_argument("--data-dir", default=None)

    serve_cmd = sub.add_parser("serve", help="Start the local HTTP API and UI.")
    serve_cmd.add_argument("--host", default="127.0.0.1")
    serve_cmd.add_argument("--port", type=int, default=8765)
    serve_cmd.add_argument("--data-dir", default=None)
    serve_cmd.add_argument("--live", action="store_true")

    args = parser.parse_args(arguments)
    command = args.command
    if command is None:
        parser.print_help()
        return 2
    if command == "deliberate":
        return _deliberate(
            args.inquiry,
            risk_level=args.risk_level,
            as_json=args.json,
            recommend=args.recommend,
            live=args.live,
            data_dir=args.data_dir,
        )
    if command == "replay":
        return _replay(args.deliberation_id, as_json=args.json, data_dir=args.data_dir)
    if command == "show":
        return _show(args.deliberation_id, as_json=args.json, data_dir=args.data_dir)
    if command == "list":
        settings = load_settings(data_dir=args.data_dir)
        store = ArtifactStore(settings)
        for item in store.list_deliberation_ids():
            print(item)
        return 0
    if command == "ingest-evidence":
        return _ingest_evidence(args)
    if command == "serve":
        from prismcognition.api.app import serve

        try:
            serve(host=args.host, port=args.port, data_dir=args.data_dir, live=args.live)
        except LiveModeUnavailableError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        return 0
    parser.print_help()
    return 2


def _deliberate(
    inquiry: str,
    *,
    risk_level: str,
    as_json: bool,
    recommend: bool,
    live: bool,
    data_dir: Optional[str],
) -> int:
    settings = load_settings(data_dir=data_dir, live=live)
    try:
        orchestrator = build_default_orchestrator(settings=settings, live=live)
    except LiveModeUnavailableError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    artifact = asyncio.run(orchestrator.deliberate(inquiry, risk_level=risk_level, emit_recommendation=recommend))
    store = build_artifact_store(settings)
    store.save_artifact(artifact)
    store.save_bundle(orchestrator.last_frozen_bundle())
    _print_artifact(artifact, as_json=as_json)
    return 0


def _replay(deliberation_id: str, *, as_json: bool, data_dir: Optional[str]) -> int:
    settings = load_settings(data_dir=data_dir)
    store = ArtifactStore(settings)
    replayed = ReplayEngine.replay(store.load_bundle(deliberation_id))
    store.save_artifact(replayed)
    _print_artifact(replayed, as_json=as_json)
    return 0


def _show(deliberation_id: str, *, as_json: bool, data_dir: Optional[str]) -> int:
    settings = load_settings(data_dir=data_dir)
    artifact = ArtifactStore(settings).load_artifact(deliberation_id)
    _print_artifact(artifact, as_json=as_json)
    return 0


def _ingest_evidence(args) -> int:
    settings = load_settings(data_dir=args.data_dir)
    from prismcognition.evidence.store import EvidenceStore

    store = EvidenceStore.load_json(settings.evidence_path) if settings.evidence_path.exists() else EvidenceStore(
        snapshot_id="runtime"
    )
    store.ingest(
        EvidenceRecord(
            record_id=args.record_id,
            snapshot_id=store.snapshot_id,
            domain_key=args.domain_key,
            polarity=Polarity(args.polarity),
            regime=GroundingRegime(args.regime),
            status=GroundingStatus(args.status),
            support_score=args.score,
            source_ref=args.source_ref,
        )
    )
    settings.evidence_path.parent.mkdir(parents=True, exist_ok=True)
    store.save_json(settings.evidence_path)
    print(f"Wrote {settings.evidence_path}")
    return 0


def _print_artifact(artifact, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(artifact.model_dump(mode="json"), indent=2, sort_keys=True))
        return
    print(render_deliberation(artifact))


if __name__ == "__main__":
    raise SystemExit(main())
