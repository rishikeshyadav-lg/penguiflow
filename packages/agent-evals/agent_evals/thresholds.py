"""The numbers a promotion gate compares a candidate against, and the name a decision records them by.

The owner's rule the gate encodes: accuracy decides. A change earns approval by getting measurably
more answers right; a speed or cost saving can also earn it, but only one large enough to be felt.
Cost and latency are otherwise a veto only when the change is huge, judged on the unlucky end of the
range. Speed or cost falling short of a *benefit* is never a reason to refuse.

The measured values (the accuracy bar, the noise floors, the candidate's floor, the margin) have no
defaults: they come from a calibration of the agent being judged (`calibration.propose_thresholds`),
and a default would quietly carry another agent's numbers into this one. The structural rules do
have defaults.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PromotionThresholds:
    """The promotion gate's thresholds.

    Measured from a baseline calibration (required):
    - `minimum_accuracy_improvement`: the gain a change must show before it is more than the agent's
      own run-to-run swing (2.8 standard errors of that swing).
    - `latency_improvement_noise_floor_ms`, `cost_improvement_noise_floor_usd`: the same swing for
      latency and cost, so a saving smaller than it is not counted as a benefit.
    - `minimum_candidate_correct_rate`: the lowest correct rate a candidate may show.
    - `non_inferiority_margin`: how far an identical candidate's correct rate moves by chance.

    The owner's rules (defaults): a saving of 35% is a felt benefit; an answer may cost up to +100% or
    take up to +50% longer before that alone is a veto.

    Rules of the range (defaults): the accuracy gain's range must sit above `accuracy_gain_interval_floor`
    (0: "the gain is real") and its average must reach `minimum_accuracy_improvement` ("big enough").
    None makes the range itself clear the bar, which counts the noise twice. Fewer than
    `minimum_distinct_prompts` prompts cannot be judged at all.
    """

    minimum_accuracy_improvement: float
    latency_improvement_noise_floor_ms: float
    cost_improvement_noise_floor_usd: float
    minimum_candidate_correct_rate: float
    non_inferiority_margin: float
    minimum_complete_pairs_per_frozen_case: int = 3
    minimum_latency_improvement_fraction: float = 0.35
    minimum_cost_improvement_fraction: float = 0.35
    maximum_cost_regression: float = 1.00
    maximum_latency_regression: float = 0.50
    confidence_level: float = 0.95
    bootstrap_resamples: int = 10_000
    accuracy_gain_interval_floor: float | None = 0.0
    # True holds the candidate's own correct rate to the floor on its range's low end; False holds the
    # average to it and leaves the low end to the non-regression cap.
    candidate_floor_on_interval: bool = False
    minimum_distinct_prompts: int = 8

    def __post_init__(self) -> None:
        if self.minimum_complete_pairs_per_frozen_case < 1:
            raise ValueError("minimum_complete_pairs_per_frozen_case must be at least 1")
        for name, value in {
            "minimum_accuracy_improvement": self.minimum_accuracy_improvement,
            "minimum_latency_improvement_fraction": self.minimum_latency_improvement_fraction,
            "minimum_cost_improvement_fraction": self.minimum_cost_improvement_fraction,
            "latency_improvement_noise_floor_ms": self.latency_improvement_noise_floor_ms,
            "cost_improvement_noise_floor_usd": self.cost_improvement_noise_floor_usd,
            "minimum_candidate_correct_rate": self.minimum_candidate_correct_rate,
            "maximum_cost_regression": self.maximum_cost_regression,
            "maximum_latency_regression": self.maximum_latency_regression,
            "non_inferiority_margin": self.non_inferiority_margin,
        }.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.minimum_candidate_correct_rate > 1:
            raise ValueError("minimum_candidate_correct_rate must not exceed 1")
        # A fraction above 1 would ask a candidate to improve a metric by more
        # than the whole baseline, which no run can do for latency or cost.
        for name, fraction in {
            "minimum_latency_improvement_fraction": self.minimum_latency_improvement_fraction,
            "minimum_cost_improvement_fraction": self.minimum_cost_improvement_fraction,
        }.items():
            if not 0 <= fraction <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.non_inferiority_margin > 1:
            raise ValueError("non_inferiority_margin must not exceed 1")
        if self.accuracy_gain_interval_floor is not None and not math.isfinite(self.accuracy_gain_interval_floor):
            raise ValueError("accuracy_gain_interval_floor must be finite")
        if self.minimum_distinct_prompts < 0:
            raise ValueError("minimum_distinct_prompts must not be negative")

    @property
    def accuracy_gain_range_floor(self) -> float:
        """What the low end of the accuracy gain's range must clear."""

        if self.accuracy_gain_interval_floor is None:
            return self.minimum_accuracy_improvement
        return self.accuracy_gain_interval_floor


def values_digest(thresholds: PromotionThresholds, *, omit_when: Mapping[str, Any] | None = None) -> str:
    """A 12-character digest of the values.

    A field named in `omit_when` is left out of the digest when it equals the value given there. That
    lets a rule added later keep every name recorded before it existed exactly as it was.
    """

    values = asdict(thresholds)
    for name, legacy in (omit_when or {}).items():
        if values[name] == legacy:
            del values[name]
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[:12]


def thresholds_version(
    thresholds: PromotionThresholds,
    *,
    named: Mapping[str, PromotionThresholds] | None = None,
    omit_when: Mapping[str, Any] | None = None,
) -> str:
    """The name a gate decision under these thresholds is recorded with.

    A decision under one set never matches a decision under another, because the values are part of
    the name: `<name>.<6 digest characters>` for a set in `named`, `thresholds.custom-<digest>` otherwise.
    """

    digest = values_digest(thresholds, omit_when=omit_when)
    for name, known in (named or {}).items():
        if thresholds == known:
            return f"{name}.{digest[:6]}"
    return f"thresholds.custom-{digest}"


__all__ = ["PromotionThresholds", "thresholds_version", "values_digest"]
