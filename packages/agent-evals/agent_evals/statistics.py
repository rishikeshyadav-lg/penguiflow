"""The statistics every evaluation needs: paired bootstrap intervals, run-to-run noise, detectability.

The unit of resampling is always the case (a prompt), with all its repeats kept together, because
repeats of one prompt are not independent evidence. Two bootstraps live here and they are different
on purpose:

- `bootstrap_paired_intervals` is the promotion gate's: many metrics and statistics at once over
  paired runs, with a seed derived from the evidence so one evidence set always yields one decision.
- `paired_bootstrap` is the plain one: the mean per-prompt difference in a rate, for calibration.

Both were moved here unchanged from the learning control plane and the campaign, so their numbers
are the numbers those produced.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from .evaluation import MetricSpecification, PairedCaseResult, _non_empty

# Two-sided 95% critical value and the one-sided value for 80% power. Requiring an interval's lower
# bound above a bar means the estimate must exceed the bar by 1.96 standard errors; reaching that 80%
# of the time costs another 0.84. The confidence level they encode must match the gate's own.
TWO_SIDED_95_CRITICAL_VALUE = 1.959963985
ONE_SIDED_80_POWER_VALUE = 0.8416212336
DETECTION_POWER = 0.80
DETECTION_MULTIPLIER = TWO_SIDED_95_CRITICAL_VALUE + ONE_SIDED_80_POWER_VALUE

ConfidenceIntervalStatistic = Literal[
    "mean_improvement",
    "candidate_mean",
    "relative_mean_improvement",
    "relative_mean_regression",
]


@dataclass(frozen=True, slots=True)
class ConfidenceIntervalRequirement:
    """One conservative confidence-bound condition for a promotion metric."""

    metric_name: str
    statistic: ConfidenceIntervalStatistic
    minimum_lower_bound: float | None = None
    maximum_upper_bound: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metric_name", _non_empty(self.metric_name, "confidence interval metric"))
        if self.statistic not in {
            "mean_improvement",
            "candidate_mean",
            "relative_mean_improvement",
            "relative_mean_regression",
        }:
            raise ValueError("confidence interval statistic is not supported")
        has_lower_bound = self.minimum_lower_bound is not None
        has_upper_bound = self.maximum_upper_bound is not None
        if has_lower_bound == has_upper_bound:
            raise ValueError("confidence interval requirement needs exactly one lower or upper bound")
        if self.minimum_lower_bound is not None and not math.isfinite(self.minimum_lower_bound):
            raise ValueError("confidence interval lower bound must be finite")
        if self.maximum_upper_bound is not None and not math.isfinite(self.maximum_upper_bound):
            raise ValueError("confidence interval upper bound must be finite")


@dataclass(frozen=True, slots=True)
class MetricConfidenceInterval:
    """A bootstrap confidence interval retained with the gate decision."""

    metric_name: str
    statistic: ConfidenceIntervalStatistic
    confidence_level: float
    estimate: float
    lower_bound: float
    upper_bound: float
    required_lower_bound: float | None = None
    required_upper_bound: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metric_name", _non_empty(self.metric_name, "confidence interval metric"))
        if self.statistic not in {
            "mean_improvement",
            "candidate_mean",
            "relative_mean_improvement",
            "relative_mean_regression",
        }:
            raise ValueError("confidence interval statistic is not supported")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be greater than 0 and less than 1")
        has_lower_requirement = self.required_lower_bound is not None
        has_upper_requirement = self.required_upper_bound is not None
        if has_lower_requirement == has_upper_requirement:
            raise ValueError("confidence interval needs exactly one lower or upper requirement")
        values = [self.estimate, self.lower_bound, self.upper_bound]
        if self.required_lower_bound is not None:
            values.append(self.required_lower_bound)
        if self.required_upper_bound is not None:
            values.append(self.required_upper_bound)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("confidence interval values must be finite")
        if self.lower_bound > self.upper_bound:
            raise ValueError("confidence interval lower bound must not exceed its upper bound")


def bootstrap_paired_intervals(
    *,
    pairs_by_cluster: Mapping[str, Sequence[PairedCaseResult]],
    cluster_ids: Sequence[str],
    requirements: Sequence[ConfidenceIntervalRequirement],
    metric_specification: Callable[[str], MetricSpecification],
    confidence_level: float,
    resamples: int,
) -> tuple[MetricConfidenceInterval, ...]:
    """Bootstrap case-balanced paired metrics across questions and repetitions."""

    random_source = random.Random(
        _bootstrap_seed(pairs_by_cluster, cluster_ids, requirements, confidence_level, resamples)
    )
    estimates: dict[tuple[str, str], list[float]] = {
        (requirement.metric_name, requirement.statistic): [] for requirement in requirements
    }
    observed_pairs = [pair for cluster_id in cluster_ids for pair in pairs_by_cluster[cluster_id]]
    observed_statistics = {
        (requirement.metric_name, requirement.statistic): confidence_statistic(
            observed_pairs,
            requirement,
            metric_specification(requirement.metric_name),
        )
        for requirement in requirements
    }

    for _ in range(resamples):
        sampled_cluster_ids = [random_source.choice(cluster_ids) for _ in cluster_ids]
        sampled_pairs = [pair for cluster_id in sampled_cluster_ids for pair in pairs_by_cluster[cluster_id]]
        for requirement in requirements:
            key = (requirement.metric_name, requirement.statistic)
            estimates[key].append(
                confidence_statistic(
                    sampled_pairs,
                    requirement,
                    metric_specification(requirement.metric_name),
                )
            )

    tail_probability = (1 - confidence_level) / 2
    intervals = []
    for requirement in requirements:
        key = (requirement.metric_name, requirement.statistic)
        samples = estimates[key]
        intervals.append(
            MetricConfidenceInterval(
                metric_name=requirement.metric_name,
                statistic=requirement.statistic,
                confidence_level=confidence_level,
                estimate=observed_statistics[key],
                lower_bound=_percentile(samples, tail_probability),
                upper_bound=_percentile(samples, 1 - tail_probability),
                required_lower_bound=requirement.minimum_lower_bound,
                required_upper_bound=requirement.maximum_upper_bound,
            )
        )
    return tuple(intervals)


def confidence_statistic(
    pairs: Sequence[PairedCaseResult],
    requirement: ConfidenceIntervalRequirement,
    specification: MetricSpecification,
) -> float:
    """Calculate one direction-normalized statistic from paired metric values."""

    baseline_mean = sum(pair.baseline.metrics[requirement.metric_name] for pair in pairs) / len(pairs)
    candidate_mean = sum(pair.candidate.metrics[requirement.metric_name] for pair in pairs) / len(pairs)
    if requirement.statistic == "candidate_mean":
        return candidate_mean

    improvement = candidate_mean - baseline_mean
    if specification.direction == "lower_is_better":
        improvement = baseline_mean - candidate_mean
    if requirement.statistic == "mean_improvement":
        return improvement
    if baseline_mean == 0:
        # A resample of a few cases can draw a baseline that scored nothing (a correct rate of 0).
        # Nothing can be lost from nothing, and any gain is the whole of it: the relative change is
        # +1 for a gain, -1 for a loss, 0 for no change. Raising here aborted the whole gate.
        relative_improvement = math.copysign(1.0, improvement) if improvement else 0.0
    else:
        relative_improvement = improvement / abs(baseline_mean)
    if requirement.statistic == "relative_mean_improvement":
        return relative_improvement
    return -relative_improvement


def _bootstrap_seed(
    pairs_by_cluster: Mapping[str, Sequence[PairedCaseResult]],
    cluster_ids: Sequence[str],
    requirements: Sequence[ConfidenceIntervalRequirement],
    confidence_level: float,
    resamples: int,
) -> str:
    """Build a stable random seed so one evidence set always yields one decision."""

    seed_material = [str(confidence_level), str(resamples)]
    seed_material.extend(cluster_ids)
    seed_material.extend(f"{item.metric_name}:{item.statistic}" for item in requirements)
    for cluster_id in cluster_ids:
        for pair in pairs_by_cluster[cluster_id]:
            seed_material.append(pair.case_id)
            seed_material.append(str(sorted(pair.baseline.metrics.items())))
            seed_material.append(str(sorted(pair.candidate.metrics.items())))
    return "|".join(seed_material)


def _percentile(values: Sequence[float], probability: float) -> float:
    """Return one linearly interpolated percentile from finite bootstrap estimates."""

    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    lower_weight = upper_index - position
    upper_weight = position - lower_index
    return ordered[lower_index] * lower_weight + ordered[upper_index] * upper_weight


@dataclass(frozen=True)
class PairedPrompt:
    """One prompt's success rates in both variants, each over that prompt's repeats."""

    prompt_id: str
    baseline_rate: float
    candidate_rate: float

    @property
    def difference(self) -> float:
        return self.candidate_rate - self.baseline_rate


@dataclass(frozen=True)
class BootstrapInterval:
    """A percentile interval for a mean per-prompt difference, with no bar attached."""

    prompt_count: int
    mean_difference: float
    lower: float
    upper: float
    resamples: int
    confidence: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "prompt_count": self.prompt_count,
            "mean_difference": round(self.mean_difference, 4),
            "lower": round(self.lower, 4),
            "upper": round(self.upper, 4),
            "resamples": self.resamples,
            "confidence": self.confidence,
        }


def paired_bootstrap(
    prompts: Sequence[PairedPrompt], *, resamples: int = 10_000, confidence: float = 0.95, seed: int = 0
) -> BootstrapInterval | None:
    """The percentile interval for the mean candidate-minus-baseline difference; None without prompts."""

    if not prompts:
        return None
    differences = [prompt.difference for prompt in prompts]
    count = len(differences)
    generator = random.Random(seed)
    means = sorted(sum(differences[generator.randrange(count)] for _ in range(count)) / count for _ in range(resamples))
    tail = (1.0 - confidence) / 2
    lower = means[int(tail * (resamples - 1))]
    upper = means[int((1.0 - tail) * (resamples - 1))]
    return BootstrapInterval(
        prompt_count=count,
        mean_difference=sum(differences) / count,
        lower=lower,
        upper=upper,
        resamples=resamples,
        confidence=confidence,
    )


def standard_error_from_interval(interval: BootstrapInterval) -> float:
    """The standard error a 95% interval implies: its half-width over 1.96."""

    return (interval.upper - interval.lower) / 2 / TWO_SIDED_95_CRITICAL_VALUE


def prompts_needed_to_clear(bar: float, *, true_effect: float, target_sd: float) -> int | None:
    """How many held-out prompts a candidate with `true_effect` needs to clear `bar` at 80% power.

    The gate passes when the improvement's 95% lower bound reaches the bar; with a per-prompt spread
    of `target_sd` that takes (2.8 x sd / (effect - bar))^2 prompts. None when the effect is not
    above the bar at all.
    """

    if true_effect <= bar:
        return None
    return math.ceil((DETECTION_MULTIPLIER * target_sd / (true_effect - bar)) ** 2)


def accuracy_improvement_detectability(
    case_mean_accuracies: Sequence[float],
    *,
    required_accuracy_improvement: float,
) -> dict[str, Any]:
    """Report whether this suite can detect the accuracy improvement the gate demands.

    The gate proves a benefit by resampling the cases with replacement (all repeats of a drawn case
    together) and requiring the interval's low end to clear `required_accuracy_improvement`. So the
    sampling unit is the case, and the standard error of the mean is the between-case standard
    deviation over the square root of the case count. Repeats reduce only the within-case term.

    The candidate modelled is the best that can exist against this baseline: one that answers every
    case correctly. Its paired difference is `1 - case_mean_accuracy`, an affine transform of the
    baseline's own, so its standard deviation is the baseline's; that is what makes this computable
    before any candidate has run. Any real candidate has a smaller effect and usually a larger relative
    spread, so these numbers are an optimistic bound on what the suite can see.

    - `minimum_detectable_effect`: the smallest true improvement that clears the requirement 80% of
      the time, 2.80 standard errors above it.
    - `detectable`: whether that effect fits inside the headroom the baseline leaves. When False, no
      candidate can prove the accuracy benefit and a verdict would be theatre.
    - `supported_improvement_threshold`: the largest requirement this sample supports at 80% power.
    - `required_case_count`: how many cases would support the requirement at 80% power; None when the
      maximal effect does not exceed it, since more cases cannot close a gap that is not there.
    """

    case_count = len(case_mean_accuracies)
    if case_count < 2:
        raise ValueError("detectability needs at least two measured cases")

    baseline_accuracy = statistics.fmean(case_mean_accuracies)
    maximum_possible_improvement = max(0.0, 1.0 - baseline_accuracy)
    case_standard_deviation = statistics.pstdev(case_mean_accuracies)
    standard_error = case_standard_deviation / math.sqrt(case_count)

    minimum_detectable_effect = required_accuracy_improvement + DETECTION_MULTIPLIER * standard_error
    margin = maximum_possible_improvement - required_accuracy_improvement
    required_case_count = None
    if margin > 0 and case_standard_deviation > 0:
        required_case_count = math.ceil((DETECTION_MULTIPLIER * case_standard_deviation / margin) ** 2)

    return {
        "case_count": case_count,
        "detection_power": DETECTION_POWER,
        "baseline_accuracy": baseline_accuracy,
        "maximum_possible_improvement": maximum_possible_improvement,
        "case_standard_deviation": case_standard_deviation,
        "standard_error": standard_error,
        "required_accuracy_improvement": required_accuracy_improvement,
        "minimum_detectable_effect": minimum_detectable_effect,
        "detectable": minimum_detectable_effect <= maximum_possible_improvement,
        "supported_improvement_threshold": maximum_possible_improvement - DETECTION_MULTIPLIER * standard_error,
        "required_case_count": required_case_count,
    }


__all__ = [
    "DETECTION_MULTIPLIER",
    "DETECTION_POWER",
    "ONE_SIDED_80_POWER_VALUE",
    "TWO_SIDED_95_CRITICAL_VALUE",
    "BootstrapInterval",
    "ConfidenceIntervalRequirement",
    "ConfidenceIntervalStatistic",
    "MetricConfidenceInterval",
    "PairedPrompt",
    "accuracy_improvement_detectability",
    "bootstrap_paired_intervals",
    "confidence_statistic",
    "paired_bootstrap",
    "prompts_needed_to_clear",
    "standard_error_from_interval",
]
