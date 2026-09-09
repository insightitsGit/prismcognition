# PrismCognition inbound marketing templates

Reusable copy for attracting developers who need to inspect AI reasoning and
evidence. These are drafts for adaptation, not published campaigns. Replace
`{{placeholders}}` before publication. No repository URL, domain, license,
customer story, or contact address is assumed.

## Core message

**Tagline:** Keep disagreement visible. Trace the evidence. Replay the reasoning.

**One sentence:** PrismCognition is a Python library for evidence-aware AI
deliberation that preserves conflicting perspectives in reviewable, replayable
reasoning artifacts.

**Short description:** Build AI review workflows that expose claims, assumptions,
evidence gaps, and unresolved disagreement. PrismCognition provides structured
deliberation artifacts, model adapters, and offline replay for Python developers.
Start with a local example and evaluate the output against your own domain cases.

**Primary reader:** A Python developer building an AI analysis or review feature.

**Primary next step:** Run the offline quickstart.

**Secondary next step:** Inspect an artifact and evaluate it on a real, anonymized case.

## SEO publishing brief

Suggested topic phrases, not measured search-volume or ranking data:

- Main topic: evidence-aware AI deliberation in Python.
- Related topics: LLM orchestration, reasoning provenance, disagreement in AI
  systems, replayable AI reasoning, evidence grounding.
- Tutorial questions: “How do I preserve disagreement in LLM output?” and
  “How can I replay an AI reasoning artifact without another model call?”

Use one clear topic per page. Write descriptive titles and links, answer the
reader's question with a working example, and connect tutorials to the quickstart.
This follows [Google's SEO Starter Guide](https://developers.google.com/search/docs/fundamentals/seo-starter-guide).
Search performance is not guaranteed.

A GitHub README cannot set a site's HTML title, canonical URL, or description
metadata. The following fields belong in a future documentation site's CMS or
HTML head. Package keywords are discovery metadata, not a ranking guarantee.

- Page title: `PrismCognition | AI Deliberation for Python`
- Meta description: `Build evidence-aware AI review workflows in Python. Preserve disagreement, trace provenance, and replay reasoning with PrismCognition.`
- Suggested path: `/prismcognition/`
- Canonical: `{{actual_public_page_url}}`
- Social title: `Make AI disagreement reviewable with PrismCognition`
- Social description: `Explore structured claims, evidence gaps, and replayable reasoning in a Python quickstart.`
- Social image: `{{absolute_url_to_a_real_artifact_diagram}}`
- Image alt text: `PrismCognition flow from inquiry through grounding and disagreement analysis to a deliberation artifact`

After a site exists, verify discoverable links and indexing configuration. Use
[Google's developer guide](https://developers.google.com/search/docs/fundamentals/get-started-developers)
for the technical publishing checks. Do not add placeholder canonical URLs or
fabricated review/rating structured data to a live page.

## Landing-page copy template

### Hero

**Headline:** Make AI disagreement reviewable.

**Body:** PrismCognition turns an inquiry into a structured record of claims,
assumptions, evidence gaps, and conflicting perspectives. Build a Python workflow
that helps reviewers see what still needs investigation.

**Primary button:** Run the quickstart → `{{quickstart_url}}`

**Secondary link:** Inspect the architecture → `{{architecture_url}}`

### Problem

Your application returns an answer. Your reviewer still needs to know which
assumptions support it, where evidence is missing, and why perspectives disagree.

### Capabilities

Preserve unresolved disagreement. Inspect grounding and provenance. Rebuild
downstream output from a frozen bundle without requesting new model output.

### Proof

Show one real, anonymized inquiry and an excerpt from its generated artifact.
Label offline versus live mode, version, evidence inputs, and evaluation limits.
Link to a reproducer. Use the current verified test count only with its date and
platform; do not present coverage as a measure of decision accuracy.

### Readiness

PrismCognition is available for development and evaluation. Review the production
release gates before deployment. → `{{release_checklist_url}}`

### Final action

Run one inquiry. Inspect its evidence gaps. Decide whether the artifact helps
your review process. → `{{quickstart_url}}`

## Technical article template

**Working title:** How to {{specific_developer_task}} in Python

**Filled example:** How to replay an AI deliberation artifact in Python

**Description:** Learn how to save a PrismCognition deliberation bundle, replay
its downstream reasoning, and inspect the result without another model call.

**Reader's question:** {{one_question_the_article_answers}}

**Opening:** When {{concrete_problem}}, you need {{inspectable_result}}. This
tutorial demonstrates {{bounded_workflow}} using PrismCognition {{version}}.

1. Describe the input and expected artifact fields.
2. State the tested Python/library versions and offline/live mode.
3. Provide a complete runnable example with reviewed or clearly synthetic data.
4. Show actual output excerpts and explain what they mean.
5. Discuss one failure case or limitation that affects this workflow.
6. Link to the related API or architecture section.
7. End with one next step: run the quickstart or adapt the example.

**CTA:** Try this workflow with an anonymized inquiry from your project.
Start here: `{{quickstart_url}}`.

## Opt-in email sequence

Drafts only. Use for readers who subscribed; no messages are sent by these templates.

### Email 1: Start with an artifact

**Subject:** Your first PrismCognition deliberation

Hi {{first_name}},

Start with the offline quickstart to generate a structured deliberation artifact.
Inspect the claims, assumptions, and evidence gaps before connecting a provider.
The offline methods demonstrate the workflow; they are deterministic scaffolding.

Run the example: {{quickstart_url}}

{{sender_name}} · {{unsubscribe_link}}

### Email 2: Inspect the disagreement

**Subject:** What should stay unresolved in your AI workflow?

Hi {{first_name}},

Choose an inquiry where two perspectives rely on different assumptions. Inspect
`active_disagreements`, `assumptions_that_matter`, and `evidence_needed` in the
artifact. Which field would help a reviewer decide what to investigate next?

Follow the worked example: {{verified_tutorial_url}}

{{sender_name}} · {{unsubscribe_link}}

### Email 3: Evaluate it on your case

**Subject:** A practical PrismCognition evaluation

Hi {{first_name}},

Try a small set of anonymized cases with known disagreements. Record what the
artifact captures, what it misses, and where fallback occurs. Review the release
checklist before deciding how to integrate it.

Use the evaluation guide: {{release_checklist_url}}

{{sender_name}} · {{unsubscribe_link}}

## Developer launch post

I’m building PrismCognition, a Python library for evidence-aware AI deliberation.

It produces a structured artifact with claims, assumptions, evidence gaps, and
unresolved disagreements. Frozen bundles support downstream replay without new
model calls, and adapters separate the engine from provider implementations.

The project is at the development and evaluation stage. The README includes an
offline quickstart and the remaining production release gates.

Try it: {{readme_url}}

What information would make an AI-generated analysis easier for you to review?

## Case study template

**Title:** How {{team_with_permission}} evaluated {{specific_workflow}}

- Context: {{review_problem_and_constraints}}
- Baseline: {{existing_workflow_and_measurement_method}}
- Implementation: {{version_mode_evidence_sources_and_integration}}
- Evaluation: {{dataset_size_selection_and_review_rubric}}
- Results: {{observed_results_with_denominators_and_uncertainty}}
- Limitations: {{failure_cases_costs_and_manual_review_required}}
- Reproduce: {{code_or_anonymized_example_url}}

Publish only measured results with permission. Omit the case study until there is
an actual evaluation; do not invent customers, endorsements, or percentage gains.

## Campaign reuse checklist

- Replace all placeholders and test every destination link.
- Re-run code examples against the version named in the copy.
- Preserve the development-stage and deterministic-mode disclosures.
- Avoid “enterprise-ready,” “compliant,” “hallucination-free,” or accuracy/cost
  claims without supporting evaluation. A license must exist before calling the
  project open source.
- Give each article a distinct developer problem and working example.
- Use consistent links such as `?utm_source={{channel}}&utm_medium={{medium}}&utm_campaign={{campaign}}`
  on your owned website; do not put personal information in campaign parameters.
- Measure actual search visits, quickstart clicks, and voluntary evaluation
  feedback when analytics are configured. Metrics and tracking are not implemented
  by this document. Review outcomes before repeating a campaign.
