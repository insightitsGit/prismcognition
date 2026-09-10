# Testing and release acceptance

## Verified in this development pass

Coverage follow-up: **125 tests passed** on Windows/Python 3.12.14, with
**94.06% statement coverage**. The CLI reached 98% and the OpenAI-compatible
adapter reached 100%. The coverage gate is now 93%. New public CLI workflow tests
exposed and fixed an argument-parser conflict that broke explicit deliberation
commands. Provider tests cover HTTP request encoding, malformed completions,
missing embeddings, invalid HTTP response bodies, and transport failures without
network calls. The two dependency warnings below remain.

### Initial baseline

On Windows with Python 3.12.14: **104 tests passed**, including 59 newly added
parameterized cases and scenarios. Statement coverage was **89.89%**, above the
85% gate. The final wheel built successfully. Two dependency deprecation warnings
were emitted by Starlette's test client (httpx and BlockingPortal); there were no
test failures. Live providers, production load, and the CI matrix were not tested.

## Run locally

Use Python 3.11 or newer in an isolated virtual environment:

```sh
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install ".[dev]"
python -m pytest -q --cov=prismcognition --cov-report=term-missing
python -m build
```

All tests use local deterministic adapters or mocked transports, temporary storage,
and explicit offline settings. No live provider credentials are required.
The coverage floor is 93% statement coverage; coverage is not a correctness or
security certification. CI is configured for Windows and Linux, Python 3.11–3.14,
and includes a clean wheel installation outside the checkout. CI runs on
https://github.com/insightitsGit/prismcognition. The cross-platform matrix has
not been run locally.

## New regression tests

`tests/test_production_boundaries.py` checks invalid API/form inputs, evidence enum
and score validation, unsafe IDs on all four persistence operations, failed atomic
replacement, malformed embeddings, provider execution off the event loop,
cancellation of a provider wait, and missing resources.

`tests/test_enterprise_scenarios.py` checks:

1. Ingest evidence, deliberate at both risk levels, recreate the application,
   retrieve the persisted result, and replay while new orchestration is disabled.
   Routing, method accounting, recommendation policy, and coverage are retained.
2. Run eight independent inquiries concurrently, persist artifacts and bundles,
   and verify each replay belongs to the correct inquiry. This is a correctness
   scenario, not a throughput benchmark or a multi-process storage test.
3. Inject a cluster outage, retain other methods, expose the failed method, omit
   the exception's sensitive message, and preserve the failure through replay.

The existing suites cover grounding, conflict classification, pruning, routing,
provenance, missing evidence, and replay invariants.

## Before an enterprise release

The library is a useful development foundation, but the bundled service is not
ready for exposure to untrusted or multiple-tenant traffic. Passing these tests
does not close the following release gates:

- Add authenticated identities, authorization, tenant-scoped storage, and tests
  proving one tenant cannot read or modify another tenant's evidence or results.
  The current API exposes read/write routes without authentication.
- Replace file-based persistence with transactional storage or specify a strict
  single-writer deployment. Atomic replacement protects individual files, but
  artifact-plus-bundle writes are not a transaction. Concurrent evidence updates
  can lose data. Same inquiry and risk level produce the same ID and overwrite
  earlier runs, even when evidence or provider output changes. Use unique run IDs
  with a separate content hash before treating this as an immutable audit log.
- Version evidence snapshots and define how EvidenceRecord.status affects
  assessment. The present store assesses polarity and score and ignores the
  row's status; ingestion does not validate that evidence is trustworthy.
  Polarity-aligned EMPIRICAL rows can close matching factual clashes
  (`resolved_disagreements`) and promote `strongly_supported_claims`. They do
  not resolve normative, definitional, or assumption clashes; that is a design
  boundary, not an unfinished wire. Establish reviewed evidence fixtures before
  claiming reliable grounding.
- Live mode without credentials is an error (`LiveModeUnavailableError` / CLI
  exit 2). Hybrid extraction failures after live activation fall back to
  deterministic evaluators and must stay labeled as fallbacks, not provider
  agreement.
- Add whole-request deadlines, bounded admission/concurrency, provider budgets,
  payload-size enforcement before JSON parsing, and overload behavior. Threaded
  transport no longer blocks the event loop, but cancelling its await does not
  terminate the underlying HTTP request; socket timeout still governs that work.
  Chaos-room embedding translation is outside its generation timeout.
- Validate real provider contracts and hostile/malformed model output in staging;
  test rate limits, 5xx errors, stalled connections, recovery, and cancellation.
  Do not grant generated model output tool execution privileges.
- Establish latency/error objectives and load-test sustained traffic, same-key
  writes, worker restarts, disk exhaustion, and recovery on target infrastructure.
- Add redacted logs, metrics, tracing, restricted storage permissions, retention,
  encryption, backup/restore drills, and operational runbooks. Local file writers
  assume a trusted storage directory; path validation is not filesystem isolation.
- Document supported API/version compatibility, lock deployment dependencies,
  scan them for vulnerabilities, and review the built distribution. The library
  is MIT-licensed (`LICENSE`). Publishing uses `python tools/publish.py` with
  `PYPI_API_TOKEN` in the environment, or the GitHub `publish.yml` workflow.

## Benefits future users can evaluate

- **Visible disagreement:** structured clashes and irreducible tensions give a
  reviewer reasons to investigate instead of collapsing output into one answer.
- **Evidence-aware review:** provenance, source references, evidence gaps, and
  grounding regimes help distinguish measured support from missing information.
- **Replay without new model calls:** frozen intermediate artifacts allow users
  to re-evaluate recorded reasoning and thresholds with the replay engine.
- **Replaceable adapters:** protocol boundaries and an OpenAI-compatible adapter
  allow integration without tying the engine to a single model implementation.
- **Risk-aware method allocation:** probe/full routing can limit work while
  exposing coverage. Actual cost or quality improvements require benchmarks.
- **Human decision support:** optional recommendations are separate from the
  artifact, and the engine retains unresolved assumptions and tensions.

These are architectural capabilities, not demonstrated accuracy improvements,
compliance guarantees, or a substitute for domain evaluation. Deterministic
methods are scaffolding: they derive structure from the inquiry text so two
questions do not share one skeleton, but they are not independent expertise or
content-complete judgment.
