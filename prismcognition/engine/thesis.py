from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from prismcognition.engine.inquiry import extract_inquiry_features

Severity = Literal["routine", "material", "catastrophic"]

_STOP = {
    "a", "an", "the", "to", "for", "of", "and", "or", "in", "on", "at", "by",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "should", "we", "our", "us", "you", "i", "it", "is", "are", "be",
    "do", "does", "did", "can", "could", "would", "will", "just",
    "immediately", "now", "then", "quarter", "please",
}

_CATASTROPHIC = (
    "bankruptcy", "bankrupt", "insolvency", "insolvent", "liquidation",
    "collapse", "evacuate", "evacuation", "catastrophe", "catastrophic",
)
_AESTHETIC = ("paint", "color", "colour", "decor", "decorate", "decorating")


@dataclass(frozen=True)
class ThesisFrame:
    subject: str
    predicate: str
    target: str
    domain_key: str
    term: str
    assumption_name: str
    severity: Severity


def extract_thesis(text: str) -> ThesisFrame:
    """Map inquiry text onto a typed thesis. Scaffolding, not expert judgment."""
    features = extract_inquiry_features(text)
    lowered = text.lower()
    tokens = [item for item in features.token_set if item not in _STOP]

    if any(item in lowered for item in _CATASTROPHIC):
        return ThesisFrame(
            subject="Company",
            predicate="file_bankruptcy",
            target="immediately" if "immediate" in lowered else "True",
            domain_key="Company.file_bankruptcy",
            term="bankruptcy",
            assumption_name="solvency_buffer",
            severity="catastrophic",
        )
    if any(item in lowered for item in _AESTHETIC):
        return ThesisFrame(
            subject="Office",
            predicate="paint_color",
            target="chosen",
            domain_key="Office.paint_color",
            term="office_color",
            assumption_name="aesthetic_weight",
            severity="routine",
        )
    if "rainfall" in lowered:
        return ThesisFrame(
            subject="Rainfall",
            predicate="observed",
            target="measured",
            domain_key="Rainfall.observed",
            term="rainfall",
            assumption_name="measurement_error",
            severity="routine",
        )
    if "expand" in features.token_set and "plant" in features.token_set:
        return ThesisFrame(
            subject="Plant",
            predicate="expand",
            target="this_quarter",
            domain_key="Plant.expand",
            term="expansion",
            assumption_name="implementation_cost",
            severity="material",
        )
    if features.empirical_measurement:
        head = next((item for item in tokens if item not in {"measure", "observe", "experiment"}), "measurement")
        subject = head.title()
        return ThesisFrame(
            subject=subject,
            predicate="observed",
            target="True",
            domain_key=f"{subject}.observed",
            term=head,
            assumption_name="measurement_error",
            severity="routine",
        )

    content = tokens[:4] or ["thesis"]
    subject = content[0].title()
    predicate = "_".join(content[1:] or ["holds"])
    return ThesisFrame(
        subject=subject,
        predicate=predicate,
        target="True",
        domain_key=f"{subject}.{predicate}",
        term=content[0],
        assumption_name="implementation_cost",
        severity="material" if features.strategic_decision else "routine",
    )
