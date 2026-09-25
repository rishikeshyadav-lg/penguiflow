"""Classify a question once, from several reads, so its rubric never changes between replays.

Classifying each replay live let one question land on different categories from run to run (its
confidence sat near the cutoff), so its rubric, the agent's instructions and the verdict all flipped.
A question is instead classified a few times before any replay: the category and each profile field
are decided by majority, and the replay reuses that frozen decision.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ClassificationRead:
    """One independent classification of a question: its category, confidence and profile fields."""

    category: str | None
    confidence: float | None = None
    profile: Mapping[str, Any] = field(default_factory=dict)


def majority(values: Sequence[Any]) -> Any | None:
    """Return the value more than half of the reads agree on, or None when there is none."""

    counted = Counter(value for value in values if value is not None)
    if not counted:
        return None
    value, count = counted.most_common(1)[0]
    return value if count * 2 > len(values) else None


def freeze_classification(
    reads: Sequence[ClassificationRead],
    *,
    fields: Sequence[str],
    list_fields: Sequence[str] = (),
    min_confidence: float,
) -> dict[str, Any]:
    """Decide one question's category and profile from several independent reads.

    The category is the majority choice, kept only when the mean confidence of the reads that chose
    it clears `min_confidence` (the bar a live classification must clear); `raw_category` keeps the
    majority choice either way. Each field in `fields` is its own majority. Each field in
    `list_fields` keeps the items more than half of the reads listed.
    """

    raw_category = majority([read.category for read in reads])
    agreeing = [read.confidence for read in reads if read.category == raw_category and read.confidence is not None]
    confidence = sum(agreeing) / len(agreeing) if agreeing else None
    category = raw_category if confidence is not None and confidence >= min_confidence else None
    profile: dict[str, Any] = {name: majority([read.profile.get(name) for read in reads]) for name in fields}
    for name in list_fields:
        votes = Counter(item for read in reads for item in read.profile.get(name) or ())
        profile[name] = sorted(item for item, count in votes.items() if count * 2 > len(reads))
    return {
        "category": category,
        "raw_category": raw_category,
        "confidence": round(confidence, 3) if confidence is not None else None,
        "profile": profile,
        "reads": [{"category": read.category, "confidence": read.confidence} for read in reads],
    }


def mining_skip_reason(classification: Mapping[str, Any] | None, *, mineable_categories: Collection[str]) -> str | None:
    """Return why a question is not replayed for mining, or None when it should be.

    A question needs a frozen classification whose category the judge can check; anything else can
    never produce a verified run, so replaying it only costs money.
    """

    if not isinstance(classification, Mapping):
        return "not_classified"
    category = classification.get("category")
    if category is None:
        return "no_confident_category"
    if category not in mineable_categories:
        return f"not_mineable:{category}"
    return None


__all__ = ["ClassificationRead", "freeze_classification", "majority", "mining_skip_reason"]
