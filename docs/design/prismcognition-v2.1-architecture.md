# PrismCognition v2.1 — Final Architecture & Implementation Design

**Status:** Implemented system (freeze suite + pipeline suite)  
**Version:** 2.1.0-final-freeze  
**Schema version:** 2.1  
**Date:** 2026-09-09  
**Package:** `prismcognition`

This is the design record for the v2.1 final implementation specification. It states the invariants, the end-to-end pipeline, the contracts, and the modules that exist in this repository. Implementation that violates an invariant is incorrect even if a convenience test is green.

---

## 1. Purpose

`PrismCognition` is a model-agnostic, disagreement-preserving epistemic orchestrator and subtractive deliberation engine.

It does **not** exist to drive artificial consensus, force compromise, compress divergence into a truth scalar, treat missing evidence as refutation, treat geometric collapse as logical elimination, or treat low classifier relevance as epistemic exclusion.

It exists to preserve epistemic friction, isolate systemic ruin via subtractive boundaries, ground claims only inside their native regime, surface structured uncertainty, and emit a replayable `DeliberationArtifact` as the primary deliverable. Recommendations are optional and subordinate (I9).

---

## 2. Invariant Ledger

| ID | Name | Rule |
| --- | --- | --- |
| I1 | Logical primacy over vector spaces | Embeddings are auxiliary. Only `RuinBoundary.trigger_predicates` may prune a stance. `‖z_safe‖ < 10⁻⁶` sets `GEOMETRICALLY_COLLAPSED` and leaves the logical stance intact. |
| I2 | Strict multi-regime grounding | Missing grounding is never defaulted to `0.0` or `1.0`. Values are `Optional[float]` plus typed statuses. Tension stays decoupled: empirical, formal, causal, assumption, normative. |
| I3 | Diversity ⊥ conflict | Non-overlapping warrants and non-conflicting normative frameworks are `PerspectiveDiversity`. They feed coverage indices and never inflate adversarial tension. |
| I4 | Absence ≠ falsity | `None` is not `0.0`. Missing evidence preserves raw structural tension; calibrated tension stays `None`. |
| I5 | Coverage loss and router depth | Scores allocate `REQUIRED` vs `PROBE`. `SKIPPED` requires a typed structural incompatibility. Failed or missing nodes never masquerade as consensus or safety. |
| I6 | Semantic containment | Chaos Room output is quarantined. Raw catastrophic prose is parsed in isolation and never enters constructive prompts or downstream claim text. |
| I7 | Immutability of provenance | Emitted artifacts are frozen. Transforms emit new artifacts with explicit `parent_ids`. |
| I8 | Irreducible disagreement is valid | Value, axiom, and definition clashes are legal terminal states. The engine does not force convergence. |
| I9 | Optional downstream action | Recommendations are non-mandatory. The deliberation artifact is the product. |
| I10 | Universal provenance | Every claim, warrant, stance, clash, diversity, ruin artifact, grounding profile, tension profile, midpoint, pivot, disagreement, coverage report, and deliberation exposes a complete `ProvenanceRef`. |

---

## 3. Goals and Non-Goals

| ID | Goal |
| --- | --- |
| G1 | Preserve irreducible disagreement as a valid terminal state. |
| G2 | Prune only by deterministic ruin predicates. |
| G3 | Multi-regime grounding with explicit `None`. |
| G4 | Separate conflict from perspective diversity. |
| G5 | Allocate compute depth without epistemological exclusion. |
| G6 | Quarantine Chaos Room prose from constructive reasoning. |
| G7 | Derive artifacts with parent IDs; replay downstream deterministically. |
| G8 | Ground warrants individually and roll them into a `GroundingProfile`. |
| G9 | Expose coverage loss, failed nodes, and ungrounded scores to a human. |

| ID | Non-goal |
| --- | --- |
| NG1 | A single truth score or forced consensus vote. |
| NG2 | Empirical RAG on normative / interpretive / phenomenological warrants. |
| NG3 | Treating `GEOMETRICALLY_COLLAPSED` as logical deletion. |
| NG4 | Treating classifier irrelevance as `SKIPPED`. |
| NG5 | Mandatory strategic action selection. |
| NG6 | Multi-tenant SaaS, hosted UI, or cloud-provider lock-in. Live HTTP model vendors are adapter-optional. |

---

## 4. End-to-End Architecture

```
Inquiry
  │
  ▼
InquiryFeatures (typed ontology)
  │
  ▼
Depth-Allocating Router ── REQUIRED / PROBE / SKIPPED
  │
  ├──────────────┬──────────────┐
  ▼              ▼              ▼
Clusters 1–6    Clusters 1–6    Cluster 7 Chaos Room
REQUIRED        PROBE           (sandboxed JSON only)
full catalog    primary method  │
  │              │              ▼
  │              │         Containment envelope
  │              │         (digest + AST primitives;
  │              │          raw_statement stripped)
  ▼              ▼              │
EpistemicStance[]               RuinAnalysisResult
  │                             │
  └────────────┬────────────────┘
               ▼
        Subtractive Gate
        1. Predicate AST prune
        2. SVD annotate only
               │
               ▼
        EvidenceStore + Verifier
        WarrantGrounding → GroundingProfile
               │
               ▼
        Tension Engine
        clashes / diversity / profiles
               │
               ▼
        DeliberationArtifact
        + FrozenDeliberationBundle
               │
               ▼
        Deterministic Replay
               │
               ▼
        Renderer (None ≠ 0.0)
```

### 4.1 Component map

| Component | Module | Role |
| --- | --- | --- |
| Inquiry ontology | `engine/inquiry.py` | Typed features; `SKIPPED` is decided on features, not raw English in the router core |
| Router | `engine/router.py` | Depth allocation |
| Method catalog | `engine/catalog.py` | Two methods per constructive cluster; PROBE = primary only |
| Cluster emitters | `engine/clusters.py` | Deterministic stance emission per method |
| Structured extractor | `engine/extractor.py` | Optional LLM JSON → `EpistemicStance` |
| Chaos Room | `subtractive/containment.py` | Isolated generation + AST translation |
| Subtractive gate | `subtractive/gate.py` | Predicate prune + SVD annotation |
| Evidence ledger | `evidence/store.py` | Snapshot records; absence is `None` |
| Verifier / rollup | `evidence/verifier.py`, `evidence/rollup.py` | Per-warrant grounding → profile |
| Inference | `engine/inference.py` | `CompatiblePremises` ∧ contradictory conclusions |
| Tension | `engine/tension.py` | Multi-regime profiles |
| Derive | `engine/derive.py` | Shared disagreement + support promotion |
| Orchestrator | `engine/orchestrator.py` | Full pipeline + freeze |
| Replay | `replay/engine.py` | Downstream reconstruction, including re-derived evidence/irreducibles |
| Renderer | `render/artifact.py` | Human text that cannot paint `None` as zero |
| Factory / CLI | `factory.py`, `cli.py` | Default wiring; `python -m prismcognition` |

---

## 5. Formal Model

### 5.1 Tension profile

\[
\mathbf{\Delta}_{ij}=\langle D_{\text{pred}}, D_{\text{warr}}, D_{\text{assump}}, D_{\text{norm}}, \Delta_{\text{aggregate}}\rangle
\]

Calibration, per regime \(\mathcal{R}\):

\[
\Delta^{*}(\mathcal{R})=
\begin{cases}
D_{\mathcal{R}}\sqrt{g_i(\mathcal{R})g_j(\mathcal{R})} & g_i,g_j\in[0,1] \\
\text{None} & g_i\text{ or }g_j\text{ is None} \\
\text{None} & \mathcal{R}\in\{\text{NORMATIVE},\text{INTERPRETIVE},\text{PHENOMENOLOGICAL}\}
\end{cases}
\]

Default aggregate:

\[
\Delta_{\text{agg}}=0.40\,D_{\text{pred}}+0.30\,D_{\text{warr}}+0.20\,D_{\text{assump}}+0.10\,D_{\text{norm}}
\]

Weights live in `threshold_config` so replay can reproduce them. Causal calibration uses `CAUSAL` scores. Assumption calibration uses empirical scores, then causal if empirical is missing. Normative calibrated tension is always `None`.

Overall calibrated tension is the mean of available `{empirical, formal, causal, assumption}` or `None` if that set is empty.

### 5.2 Normative state

\[
\text{Conflict}_{1\to 2}=P_1\cap O_2,\quad \text{Conflict}_{2\to 1}=P_2\cap O_1
\]

Nonempty either way ⇒ `NORMATIVE_CONFLICT`. Different frameworks and empty conflicts ⇒ `NORMATIVE_FRAMEWORK_DIVERSITY`. Intersection is on `(domain_key, polarity)` so opposite-polarity obligation/prohibition pairs do not false-positive. `priority_rank` is recorded on the clash and never used to force a winner (I8).

### 5.3 Compatible premises

An `INFERENCE_RULE_CONFLICT` requires:

1. Shared formal `rule_statement`
2. Dereferenced conclusions with same `domain_key`, both non-neutral, opposite polarity
3. `CompatiblePremises`: every premise ID resolves in the claims universe, and no shared premise `domain_key` has opposing polarity

Unresolved premise IDs refuse the clash (no false positive). Empty shared keys are compatible.

### 5.4 SVD safe-harbor

\(B=U\Sigma V^{\mathsf T}\), rank \(K=\#\{\sigma_i>\tau\sigma_1\}\), \(z_{\text{safe}}=(I-Q_KQ_K^{\mathsf T})z\). Vanishing residual annotates collapse; it does not prune.

---

## 6. Data Contracts

All artifacts inherit `ImmutableBase` (`frozen=True`, `extra=forbid`). Sequences on artifacts are tuples.

Canonical types live in `prismcognition/schemas/core.py`:

- Identity: `ProvenanceRef`
- Graph: `StructuredClaim`, `Warrant`, `WarrantGrounding`, `Assumption` (`bool | float | str`), `NormativeCommitment`, `EpistemicStance`
- Grounding: `RegimeAssessment`, `GroundingProfile`
- Tension: `EpistemicClash` (includes `CAUSAL_CONFLICT`, `DEFINITIONAL_CONFLICT`), `PerspectiveDiversity`, `TensionProfile`, `DisagreementArtifact`, `AssumptionMidpoint`, `EpistemicPivot`
- Ruin: `RuinBoundary`, `RuinAnalysisResult`, `ChaosExtractionEnvelope`
- Routing: `InquiryFeatures`, `ClusterRoute`, `RoutePlan`, `ExecutionTier`, `AllocationScoreSource`
- Outputs: `DeliberationArtifact`, `FrozenDeliberationBundle`, `CoverageReport`, `EvidenceRecord`

`FrozenDeliberationBundle.frozen_grounding_profiles` is `Dict[str, GroundingProfile]` (dict payloads coerce). `frozen_warrant_groundings` is `Dict[str, Tuple[WarrantGrounding, ...]]`. Bundles also store `frozen_route_plan`.

`ClusterRoute.cluster_id` remains `int` (classifier space 1–7). `EpistemicStance.cluster_id` is `str`. The orchestrator converts at the boundary. Each route records `classifier_score` (`None` if the key was missing), `allocated_score`, and `score_source`. A missing classifier key is a compute default for depth only; it is never an empty-cluster or safety finding.

`DeliberationArtifact` carries `route_plan` and `coverage`. `assumptions_that_matter` is only the high-spread parameters that produced `ASSUMPTION_CONFLICT` midpoints, not every assumption on every stance.

`router_required` and `high_risk_floor` in `threshold_config` are written onto the injected `DepthAllocatingRouter` at orchestrator construction. Grounding and deliberation provenance copy `evidence_snapshot_id` from the bound evidence ledger.

---

## 7. Router and Inquiry Ontology

`extract_inquiry_features(text)` emits typed flags:

| Flag | Meaning |
| --- | --- |
| `formal_a_priori` | Closed proof / axiom-definition request |
| `physical_constant_query` | Closed physical-constant or orbital calculation |
| `empirical_measurement` | Observation / measurement request |
| `normative_judgment` | Duty / ought / ethics request |
| `causal_mechanism` | Cause / intervention / dynamics |
| `strategic_decision` | Decision / risk / should-we |

`SKIPPED` is decided on flags, not by the router scanning arbitrary prose:

- Cluster 2 (empirical) is incompatible when `formal_a_priori` and not `empirical_measurement`
- Cluster 6 (normative) is incompatible when `physical_constant_query` and not `normative_judgment`
- Cluster 7 is never skipped

Low classifier score ⇒ `PROBE`. High risk floors clusters 4 and 5 to `REQUIRED`. Missing classifier keys are `0.0` **compute** defaults (`AllocationScoreSource.MISSING_COMPUTE_DEFAULT`), never “the cluster is empty.” The distinction is stored on `ClusterRoute` and rendered when present.

Feature extraction uses a typed phrase-and-token ontology (proof/axiom language, physical constants, measurement verbs, deontic language, causal language, strategic verbs). The router still decides `SKIPPED` only from those flags.

---

## 8. Method Catalog (Clusters 1–6)

| Cluster | Methods | PROBE | REQUIRED |
| --- | --- | --- | --- |
| 1 Formal | `1.1` propositional closure, `1.2` definitional analysis | `1.1` | both |
| 2 Empirical | `2.1` observational claim, `2.2` measurement bound | `2.1` | both |
| 3 Causal | `3.1` mechanism, `3.2` intervention | `3.1` | both |
| 4 Adversarial | `4.1` opposing counsel, `4.2` incentive failure | `4.1` | both |
| 5 Pragmatic | `5.1` implementation cost, `5.2` institutional constraint | `5.1` | both |
| 6 Normative | `6.1` deontology, `6.2` utilitarianism | `6.1` | both |
| 7 Ruin | Chaos Room sandbox | n/a | always |

`ClusterGroup.evaluate_all` returns one stance per selected method. Deterministic emitters produce regime-correct warrants, assumptions, and commitments. An optional `StructuredClusterExtractor` accepts provider JSON and builds the same frozen stance type, including a method-scoped premise proposition, optional commitments, and a semantic vector so live stances remain indexable by `CompatiblePremises`.

---

## 9. Semantic Containment (I6)

`ChaosRoomSandbox`:

1. Calls `LLMAdapter.generate_json` only (never a raw string API).
2. Wraps the object in `ChaosExtractionEnvelope` (`session_id`, `raw_response_digest`, structured primitives).
3. Translates primitives to `RuinBoundary` AST.
4. Sets `failure_state.raw_statement = ""` and trigger `raw_statement = ""`. Catastrophic language is not copied onto claims.
5. Embeds only the trigger predicate expression, never failure prose.
6. Timeouts and parse failures become `TIMEOUT` / `FAILED` with non-prose diagnostics. Exception text is not forwarded.
7. Empty or all-invalid primitives ⇒ `PARTIAL`, never “no ruin found.”

Constructive cluster prompts never receive Chaos Room output.

---

## 10. Grounding and Evidence

1. Each warrant is verified inside its declared `GroundingRegime`.
2. `NORMATIVE`, `INTERPRETIVE`, `PHENOMENOLOGICAL`, `UNVERIFIABLE` ⇒ `NOT_GROUNDABLE`, `score=None`.
3. `EvidenceStore` is a snapshot ledger keyed by `domain_key` + regime. Matching polarity ⇒ `SUPPORTED`; opposing ⇒ `CONTRADICTED`; both ⇒ `CONTESTED`; no row ⇒ `INSUFFICIENT_EVIDENCE` / `None`.
4. Declared warrant confidence is never used as a grounding score.
5. Rollup: per-regime mean of non-`None` scores; empty regime ⇒ `applicable=False`, `score=None`.
6. The store persists to JSON (`save_json` / `load_json`) without inventing retrieval.

Strongly supported claims are **per-warrant**: a claim is promoted only if it is the conclusion of a `SUPPORTED` warrant with `support_score >= support_threshold`. If a stance has no warrant groundings, the fallback promotes the **conclusion only** when the stance-level empirical or formal score meets the threshold — never every proposition on the stance.

---

## 11. Subtractive Gate

1. Applied only when ruin status is `COMPLETED` or `PARTIAL`.
2. A stance is eliminated iff every trigger of some boundary is present with matching `domain_key` and `polarity`.
3. Survivors are projected off the ruin subspace. Collapse annotates and writes `parent_ids`.
4. Dimension mismatch or SVD failure leaves geometry `NOT_CALCULATED` and never prunes.
5. `FAILED` / `TIMEOUT` skip pruning and record that absence of boundaries is not safety.

---

## 12. Tension and Resolution

Pairwise evaluation emits clashes and diversities, then a `TensionProfile`.

Resolution order:

1. Raw aggregate `< structural_clash` ⇒ `NO_MATERIAL_DISAGREEMENT`
2. `DEFINITIONAL_CONFLICT` ⇒ `DEFINITIONALLY_IRREDUCIBLE`
3. `NORMATIVE_CONFLICT` ⇒ `NORMATIVELY_IRREDUCIBLE`
4. `INFERENCE_RULE_CONFLICT` ⇒ `AXIOMATICALLY_IRREDUCIBLE`
5. Assumption midpoint ⇒ `CONDITIONALLY_RESOLVABLE`
6. Unverifiable / not-groundable clash regime ⇒ `INSUFFICIENT_EPISTEMIC_ACCESS`
7. Missing regime scores ⇒ `CURRENTLY_UNRESOLVED`
8. Factual clash with asymmetric supported/contradicted counts ⇒ `RESOLVED_BY_EVIDENCE`
9. Else `CURRENTLY_UNRESOLVED`

`METHODOLOGICAL_COMPLEMENT` is emitted when cluster IDs differ and no clash was recorded.

Every high-spread numeric assumption emits its own `AssumptionMidpoint` and `EpistemicPivot`. Derive attaches the matching midpoint/pivot to the corresponding `ASSUMPTION_CONFLICT` disagreement instead of keeping only the first variable.

---

## 13. Orchestrator and Replay

`EpistemicOrchestrator.deliberate(inquiry, risk_level)`:

1. Route  
2. Chaos Room ∥ cluster catalog  
3. Gate  
4. Index claims (propositions, conclusions, normative claims)  
5. Verify + roll up  
6. Derive disagreements, diversity, evidence needed, irreducibles  
7. Promote strongly supported claims  
8. Freeze a `FrozenDeliberationBundle` (including `frozen_route_plan`)
9. Attach a first-class `CoverageReport` (cluster IDs and method IDs kept distinct)

Replay re-applies gate + tension and **re-derives** `evidence_needed`, `irreducible_tensions`, strongly supported claims, `assumptions_that_matter`, and coverage. It copies only inputs that are not downstream-derived: inquiry, method coverage lists, ruin result, optional recommendation, route plan, provenance. Canonical hash quantizes floats to 8 decimals.

---

## 14. Rendering Contract

`render_score(None)` is the token `UNGROUNDED`, never `"0"` or `"0.0"`.  
`render_score(0.0)` is `"0.0000"` and means scored-as-zero (refutation or empty support), not absence.

`render_deliberation` prints method coverage, ruin status, disagreements, diversity, evidence needed, and irreducibles. It never invents a recommendation when `optional_recommendation` is `None`.

---

## 15. Thresholds

| Key | Default |
| --- | --- |
| `subtractive_rank_tol` | `1e-6` |
| `structural_clash` | `0.35` |
| `assumption_spread` | `0.25` |
| `support_threshold` | `0.70` |
| `router_required` | `0.40` |
| `high_risk_floor` | `0.50` |
| `w_pred`, `w_warr`, `w_assump`, `w_norm` | `0.40`, `0.30`, `0.20`, `0.10` |

---

## 16. Verification

Required suite: `tests/test_prismcognition_freeze.py`, `tests/test_pipeline.py`, `tests/test_runtime.py`, `tests/test_audit_partials.py`, `tests/test_gaps.py`.

Covered:

- Degenerate SVD and geometric collapse without logical prune  
- Compatible premises clash; incompatible premises do not clash  
- `None` calibration with raw tension intact  
- Symmetric normative conflict  
- Router `PROBE` vs structural `SKIPPED`  
- Replay hash parity  
- Predicate prune  
- Ruin failure visibility  
- Containment timeout / failure / no prose leak  
- Evidence store assess + persist  
- Catalog PROBE vs REQUIRED method counts  
- Orchestrator end-to-end + re-derived replay fields  
- Renderer `None` vs `0.0`  
- Definitional / causal clashes, `RESOLVED_BY_EVIDENCE`, `INSUFFICIENT_EPISTEMIC_ACCESS`  
- Router thresholds applied from `threshold_config`  
- PROBE stances retain premise propositions  
- Pivot reused from tension evaluation (not rebuilt)  
- I10 provenance on tension, grounding, midpoint, pivot, disagreement, and coverage artifacts
- Multi-parameter midpoints retained and matched per clash
- Normative intersection on `(domain_key, polarity)`; priority ranks recorded, not decisive
- Missing classifier keys distinguished from scored-zero / empty-cluster
- Coverage and route plan stored on the artifact; coverage re-derived on replay
- `assumptions_that_matter` limited to high-spread clash parameters
- Evidence snapshot IDs on grounding and deliberation provenance
- Live extractor premises, commitments, and vectors indexable for inference clashes

---

## 17. File Layout

```
prismcognition/
  schemas/core.py
  identity.py
  settings.py
  factory.py
  cli.py
  __main__.py
  adapters/protocols.py
  adapters/deterministic.py
  adapters/features.py
  adapters/openai_compat.py
  evidence/store.py
  evidence/rollup.py
  evidence/verifier.py
  persist/store.py
  engine/inquiry.py
  engine/router.py
  engine/catalog.py
  engine/clusters.py
  engine/extractor.py
  engine/hybrid.py
  engine/recommend.py
  engine/inference.py
  engine/tension.py
  engine/derive.py
  engine/coverage.py
  engine/orchestrator.py
  subtractive/gate.py
  subtractive/containment.py
  replay/engine.py
  render/artifact.py
  api/app.py
tests/
  test_prismcognition_freeze.py
  test_pipeline.py
  test_runtime.py
  test_audit_partials.py
  test_gaps.py
docs/design/
  prismcognition-v2.1-architecture.md
```

Run:

```
python -m prismcognition "Should we expand the plant this quarter?"
python -m prismcognition deliberate "..." --recommend --data-dir .prismcognition
python -m prismcognition ingest-evidence --record-id e1 --domain-key Inquiry.thesis_holds --polarity 1 --regime EMPIRICAL --score 0.8 --source-ref note://1
python -m prismcognition serve --port 8765
```

Live providers (optional): set `PRISM_LLM_API_KEY` or `OPENAI_API_KEY` and pass `--live`. The OpenAI-compatible adapter posts to `/chat/completions` and `/embeddings` and accepts only JSON objects. Offline mode stays fully functional.

---

## 18. Runtime Completeness

Implemented locally (single-operator, not a multi-tenant SaaS):

- Feature classifier from typed inquiry flags  
- OpenAI-compatible LLM and embedder adapters (injectable transport; no vendor SDK)  
- Hybrid cluster extractors with deterministic fallback  
- File persistence for artifacts, bundles, and evidence  
- Opt-in subordinate recommendation (`--recommend`) that cannot claim consensus or safety  
- Local HTTP API + form UI (`python -m prismcognition serve`)  
- CLI: `deliberate`, `replay`, `show`, `list`, `ingest-evidence`, `serve`

Still not this engine's job:

- Multi-tenant auth and hosted SaaS  
- Open-web RAG or a general theorem prover  

Those are non-goals (NG6 / NG2), not unfinished modules.

Absence of a live key or evidence snapshot never invents grounding, consensus, or safety.
