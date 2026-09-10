from __future__ import annotations

import re

from prismcognition.schemas.core import InquiryFeatures

_FORMAL_PHRASES = (
    "purely a priori",
    "formal axiom definition",
    "formal axiom",
    "prove that",
    "by definition only",
    "closed-form proof",
    "formal proof",
    "necessary truth",
)
_PHYSICAL_PHRASES = (
    "speed of light",
    "calculate orbit",
    "planck constant",
    "gravitational constant",
    "orbital period",
    "fine-structure",
    "boltzmann constant",
    "avogadro",
)
_EMPIRICAL_PHRASES = (
    "measure",
    "observe",
    "dataset",
    "experiment",
    "empirical",
    "survey",
    "observation",
    "measurement",
    "field data",
)
_NORMATIVE_PHRASES = (
    "should we",
    "ought",
    "ethical",
    "moral",
    "justified",
    "rights",
    "duty",
    "permissible",
    "forbidden",
)
_CAUSAL_PHRASES = (
    "cause",
    "because",
    "mechanism",
    "intervene",
    "leads to",
    "feedback",
    "if we change",
    "causal",
)
_STRATEGIC_PHRASES = (
    "should we",
    "risk",
    "decide",
    "expand",
    "invest",
    "strategy",
    "go/no-go",
    "portfolio",
)

_FORMAL_TOKENS = {"prove", "proof", "theorem", "lemma", "tautology", "axiom", "qed"}
_FORMAL_ANCHORS = {"axiom", "definition", "priori", "deduce", "deduction"}
_PHYSICAL_TOKENS = {"planck", "kepler", "avogadro", "boltzmann", "orbit", "orbital"}
_EMPIRICAL_TOKENS = {"measure", "observe", "observation", "dataset", "experiment", "survey", "measurement"}
_NORMATIVE_TOKENS = {"ought", "ethical", "moral", "duty", "rights", "permissible", "forbidden"}
_CAUSAL_TOKENS = {"cause", "because", "mechanism", "intervene", "intervention", "causal"}
_STRATEGIC_TOKENS = {
    "decide", "decision", "risk", "expand", "invest", "strategy", "portfolio",
    "bankruptcy", "bankrupt",
}
_CATASTROPHIC_TOKENS = {
    "bankruptcy", "bankrupt", "insolvency", "liquidation", "collapse", "evacuate", "catastrophic",
}
_AESTHETIC_TOKENS = {"paint", "color", "colour", "decor", "decorate", "decorating"}


def extract_inquiry_features(text: str) -> InquiryFeatures:
    lowered = text.lower()
    tokens = tuple(part for part in re.findall(r"[a-z0-9']+", lowered) if part)
    token_set = set(tokens)
    return InquiryFeatures(
        formal_a_priori=_formal(lowered, token_set),
        physical_constant_query=_physical(lowered, token_set),
        empirical_measurement=_phrase_or_token(lowered, token_set, _EMPIRICAL_PHRASES, _EMPIRICAL_TOKENS),
        normative_judgment=_phrase_or_token(lowered, token_set, _NORMATIVE_PHRASES, _NORMATIVE_TOKENS),
        causal_mechanism=_phrase_or_token(lowered, token_set, _CAUSAL_PHRASES, _CAUSAL_TOKENS),
        strategic_decision=_phrase_or_token(lowered, token_set, _STRATEGIC_PHRASES, _STRATEGIC_TOKENS),
        catastrophic_stakes=bool(token_set & _CATASTROPHIC_TOKENS),
        routine_or_aesthetic=bool(token_set & _AESTHETIC_TOKENS),
        token_set=tokens,
    )


def structurally_incompatible(cluster_id: int, features: InquiryFeatures) -> bool:
    if cluster_id == 2:
        return features.formal_a_priori and not features.empirical_measurement
    if cluster_id == 6:
        return features.physical_constant_query and not features.normative_judgment
    return False


def _formal(text: str, tokens: set[str]) -> bool:
    if _contains(text, _FORMAL_PHRASES):
        return True
    return bool(tokens & _FORMAL_TOKENS) and bool(tokens & _FORMAL_ANCHORS or "priori" in text)


def _physical(text: str, tokens: set[str]) -> bool:
    if _contains(text, _PHYSICAL_PHRASES):
        return True
    if tokens & {"planck", "avogadro", "boltzmann", "kepler"}:
        return True
    return bool(tokens & {"orbit", "orbital"}) and bool(tokens & {"calculate", "period", "kepler"})


def _phrase_or_token(text: str, tokens: set[str], phrases: tuple[str, ...], token_needles: set[str]) -> bool:
    return _contains(text, phrases) or bool(tokens & token_needles)


def _contains(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)
