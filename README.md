# PrismCognition: evidence-aware AI deliberation for Python

**Keep disagreement visible. Trace the evidence. Replay the reasoning.**

**Author:** Amin Parva

PrismCognition is a Python library for structured AI deliberation. It represents
claims, assumptions, evidence gaps, and conflicting perspectives in a reviewable
artifact. Model adapters let you connect an OpenAI-compatible provider, while an
offline mode lets you explore the workflow without API credentials.

Use it when a reviewer needs to understand **why perspectives disagree, what
evidence is missing, and which assumptions could change a decision**.

[Run the quickstart](#quickstart) · [Explore the Python API](#python-example) ·
[Review release readiness](docs/testing.md) ·
[Read the architecture](docs/design/prismcognition-v2.1-architecture.md)

## Why use PrismCognition?

- **Inspect disagreement.** Structured clashes distinguish differences in claims,
  assumptions, definitions, and values. Unresolved disagreement can remain in
  the final artifact.
- **Follow the evidence.** Grounding profiles and provenance expose sources and
  missing support across different reasoning regimes.
- **Revisit a recorded analysis.** Frozen bundles let the replay engine rebuild
  downstream results without requesting new model output.
- **Connect your own models.** Adapter protocols separate orchestration from
  model and embedding implementations.
- **See what ran.** Routing and coverage identify methods that were executed,
  probed, skipped, or failed.
- **Keep humans in the decision.** Recommendations are optional; the structured
  deliberation artifact is the primary output.

These capabilities are useful for prototyping AI review tools, comparing
assumptions in research workflows, and building evidence-aware decision support.
Accuracy improvements, cost savings, and production scale have not been benchmarked.

## Project status

Version **2.1.0** · Python **3.11+** · Development and evaluation stage.

The latest local validation recorded **125 passing tests** and **94.06% statement
coverage** on Windows/Python 3.12.14. CI has a 93% coverage floor and is configured
for Windows/Linux and Python 3.11–3.14; the full matrix has not yet been verified.
See the [test guide](docs/testing.md) for scope and limitations.

The bundled service has no authentication or tenant isolation. File storage is
not transactional across artifacts and bundles, and repeated inquiries can
overwrite earlier runs. Review the [enterprise release gates](docs/testing.md#before-an-enterprise-release)
before a production deployment.

## Quickstart

From a local checkout of this project:

```sh
python -m venv .venv
```

Activate the environment on macOS/Linux:

```sh
source .venv/bin/activate
```

Or in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install from the checkout and run an offline inquiry:

```sh
python -m pip install .
python -m prismcognition deliberate "Should we expand the plant this quarter?" --risk-level LOW --data-dir .prismcognition-demo
```

This uses deterministic scaffolding by default. It demonstrates the data flow;
its method outputs are not independent expert judgments. No provider credentials
are needed. Installation from a public package registry is not assumed here.

For machine-readable output or an optional action note:

```sh
python -m prismcognition deliberate "Measure rainfall" --json --data-dir .prismcognition-demo
python -m prismcognition deliberate "Measure rainfall" --recommend --data-dir .prismcognition-demo
python -m prismcognition list --data-dir .prismcognition-demo
```

Use an ID returned by `list` in place of `DELIBERATION_ID`:

```sh
python -m prismcognition show DELIBERATION_ID --data-dir .prismcognition-demo
python -m prismcognition replay DELIBERATION_ID --json --data-dir .prismcognition-demo
```

Replay uses the stored bundle. It does not retrieve new evidence or ask a model
to repeat its answer.

## Python example

```python
import asyncio

from prismcognition import build_default_orchestrator
from prismcognition.replay.engine import ReplayEngine
from prismcognition.settings import load_settings


async def main():
    settings = load_settings(data_dir=".prismcognition-demo", live=False)
    engine = build_default_orchestrator(settings=settings)
    artifact = await engine.deliberate(
        "Should we expand the plant this quarter?",
        risk_level="LOW",
    )
    print(artifact.model_dump_json(indent=2))

    bundle = engine.last_frozen_bundle()
    replayed = ReplayEngine.replay(bundle)
    assert replayed.inquiry_text == artifact.inquiry_text


asyncio.run(main())
```

This example keeps results in memory. The CLI and HTTP API persist results;
Python callers can explicitly save them using `ArtifactStore`.

## What does a deliberation contain?

A `DeliberationArtifact` includes:

- `strongly_supported_claims`: claims selected under the engine's grounding rules.
- `active_disagreements` and `perspective_diversities`: conflicts and differences
  represented separately.
- `evidence_needed` and `assumptions_that_matter`: gaps and assumptions surfaced
  for further review.
- `irreducible_tensions`: disagreements the engine leaves unresolved.
- `ruin_analysis`: structured failure boundaries and analysis status.
- `route_plan`, `coverage`, and method accounting: what the engine considered.
- `provenance`: information linking derived output to its inputs.
- `optional_recommendation`: absent unless explicitly requested.

The engine routes an inquiry, evaluates methods, applies failure predicates,
assesses grounding, and derives the final artifact. See the
[architecture and invariants](docs/design/prismcognition-v2.1-architecture.md)
for the detailed contracts.

## Add evidence

This synthetic row demonstrates ingestion; replace it with reviewed evidence
that actually supports the domain and claim being assessed:

```sh
python -m prismcognition ingest-evidence --record-id demo-1 --domain-key Inquiry.thesis_holds --polarity 1 --regime EMPIRICAL --score 0.8 --source-ref demo://synthetic-example --data-dir .prismcognition-demo
```

Evidence ingestion does not verify source truth. The current assessment logic
uses polarity and does not incorporate the row's `status`; this is an outstanding
release gate. Do not use a synthetic score as a real confidence measurement.

## Local API and browser UI

```sh
python -m prismcognition serve --host 127.0.0.1 --port 8765 --data-dir .prismcognition-demo
```

Open [the local UI](http://127.0.0.1:8765) or
[interactive API documentation](http://127.0.0.1:8765/docs).
Routes support deliberation creation, retrieval, listing, replay, and evidence
ingestion. Keep this unauthenticated development service on a trusted local interface.

## Connect a model provider

Configure credentials through your environment or secret manager, then opt in
with `--live` or `load_settings(live=True)`:

- `PRISM_LLM_API_KEY`: provider credential; `OPENAI_API_KEY` is a fallback.
- `PRISM_LLM_BASE_URL`: the provider's compatible API base URL.
- `PRISM_LLM_MODEL`: the chat model identifier.
- `PRISM_EMBED_MODEL`: the embedding model identifier.
- `PRISM_HTTP_TIMEOUT`: transport timeout in seconds; defaults to 30.
- `PRISM_DATA_DIR`: default local storage location.

Select models and an endpoint supported by your provider. Live requests send
inquiry content to that provider and may incur charges. Missing credentials
currently disable live mode, and extraction failures can fall back to deterministic
methods. Account for this behavior in evaluations.

## Tests and development

```sh
python -m pip install ".[dev]"
python -m pytest -q --cov=prismcognition --cov-report=term-missing
python -m build
```

Tests cover reasoning invariants, provider failures, CLI workflows, input
validation, persistence, concurrency scenarios, and replay. See
[testing and release acceptance](docs/testing.md) before interpreting coverage
as evidence of production readiness.

## Frequently asked questions

### Is PrismCognition an AI agent framework?

It is an AI deliberation and orchestration library focused on structured reasoning
artifacts. The current implementation does not provide a general-purpose tool
execution agent or autonomous workflow platform.

### Does it work without an LLM?

Yes. Offline deterministic adapters support development and repeatable tests.
Use live adapters and domain evaluations to assess model-backed behavior.

### Does it replace RAG?

It has an evidence ledger and grounding interfaces. It does not include a full
document retrieval or vector-search pipeline. A retrieval system can supply
reviewed evidence through an integration you build.

### Is it ready for enterprise production?

Further engineering is required, including access controls, tenant isolation,
transactional history, observable fallback behavior, and load testing. See the
[release checklist](docs/testing.md#before-an-enterprise-release).

## License and feedback

No license has been selected in this checkout. Usage and redistribution terms
need to be established before a public release.

For a useful bug report, include the library/Python version, an anonymized inquiry,
offline or live mode, expected behavior, actual behavior, and a minimal reproducer.
Remove credentials and private evidence before sharing a report.

Maintainers can use the [inbound marketing templates](docs/marketing/inbound-templates.md)
to turn verified examples into tutorials, landing-page copy, and launch content.
