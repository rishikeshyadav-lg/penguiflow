"""Derive promotion thresholds from a baseline-only run over real prompts, with the reasoning.

The gate's thresholds must come from the population a candidate is judged on. This takes what a
baseline calibration run recorded, per prompt and per repeat, and computes what that population
supports: how accurate the baseline is and how noisy, the smallest accuracy gain the sample can
detect, the floor a candidate must clear, and how much latency and cost vary between repeats of the
same prompt. The owner's rules (what counts as a felt speed-up, a tolerable slow-down, a tolerable
extra cost) are inputs, not derived. Every number comes with one sentence saying where it came from.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..running.execution import RepeatedRun
from .statistics import (
    DETECTION_MULTIPLIER,
    PairedPrompt,
    accuracy_improvement_detectability,
    paired_bootstrap,
    prompts_needed_to_clear,
    standard_error_from_interval,
)
from .thresholds import PromotionThresholds


@dataclass
class PromptBaseline:
    """One prompt's baseline repeats: correct (1/0), latency and cost per repeat, in repeat order."""

    trace_id: str
    correct: list[float] = field(default_factory=list)
    latency_ms: list[float] = field(default_factory=list)
    cost_usd: list[float] = field(default_factory=list)

    @property
    def correct_rate(self) -> float:
        return statistics.fmean(self.correct) if self.correct else 0.0


@dataclass(frozen=True)
class OwnerRules:
    """What the owner decided efficiency means; the calibration only fills in the noise around it."""

    latency_benefit_fraction: float = 0.35
    cost_benefit_fraction: float = 0.35
    maximum_latency_regression: float = 0.50
    maximum_cost_regression: float = 1.00
    minimum_complete_pairs_per_frozen_case: int = 3
    accuracy_requirement_candidates: tuple[float, ...] = (0.05, 0.10, 0.15)
    minimum_non_inferiority_margin: float = 0.03


def baselines_from_run(
    run: RepeatedRun, variant_id: str, *, correct_metric: str = "correct"
) -> dict[str, PromptBaseline]:
    """Group one variant's rows by case; a failed run counts as incorrect and carries no timing."""

    prompts: dict[str, PromptBaseline] = {}
    for row in sorted(run.rows_for(variant_id), key=lambda item: (item.case_id, item.repeat)):
        prompt = prompts.setdefault(row.case_id, PromptBaseline(row.case_id))
        failed = row.result.error is not None
        prompt.correct.append(0.0 if failed else float(row.result.metrics[correct_metric]))
        if not failed and row.result.latency_ms is not None:
            prompt.latency_ms.append(float(row.result.latency_ms))
            prompt.cost_usd.append(float(row.result.cost_usd or 0.0))
    return prompts


def accuracy_calibration(prompts: Mapping[str, PromptBaseline], requirements: Sequence[float]) -> dict[str, Any]:
    """The baseline's correct rate, its spread across prompts, and what each candidate requirement needs."""

    rates = [prompt.correct_rate for prompt in prompts.values()]
    return {
        "prompt_count": len(rates),
        "baseline_correct_rate": round(statistics.fmean(rates), 4),
        "prompt_standard_deviation": round(statistics.pstdev(rates), 4),
        "standard_error": round(statistics.pstdev(rates) / math.sqrt(len(rates)), 4),
        "detectability": {
            str(required): accuracy_improvement_detectability(rates, required_accuracy_improvement=required)
            for required in requirements
        },
    }


def candidate_floor(prompts: Mapping[str, PromptBaseline], *, margin: float, resamples: int = 10_000) -> dict[str, Any]:
    """The lowest correct rate a candidate may show: the baseline's bootstrap lower bound minus the margin."""

    interval = paired_bootstrap(
        [PairedPrompt(prompt.trace_id, 0.0, prompt.correct_rate) for prompt in prompts.values()], resamples=resamples
    )
    assert interval is not None
    floor = max(0.0, math.floor((interval.lower - margin) * 20) / 20)
    return {"baseline_lower_bound": round(interval.lower, 4), "margin": margin, "floor": floor}


def null_repeat_noise(prompts: Mapping[str, PromptBaseline], metric: str, *, resamples: int = 10_000) -> dict[str, Any]:
    """How much a metric moves between two repeats of the same prompt when nothing changed.

    Pairs every repeat with the next one, prompt by prompt, and bootstraps the mean difference the way
    the gate does; the interval's half-width is the noise a candidate must rise above.
    """

    pairs = [
        PairedPrompt(f"{prompt.trace_id}:{index}", values[index], values[index + 1])
        for prompt in prompts.values()
        for values in [getattr(prompt, metric)]
        for index in range(len(values) - 1)
    ]
    baseline_mean = statistics.fmean(value for prompt in prompts.values() for value in getattr(prompt, metric))
    interval = paired_bootstrap(pairs, resamples=resamples)
    assert interval is not None
    half_width = (interval.upper - interval.lower) / 2
    standard_error = standard_error_from_interval(interval)
    return {
        "baseline_mean": round(baseline_mean, 4),
        "null_interval": [round(interval.lower, 4), round(interval.upper, 4)],
        "half_width": round(half_width, 4),
        "standard_error": round(standard_error, 4),
        "noise_floor": round(DETECTION_MULTIPLIER * standard_error, 4),
        "null_relative_regression_upper": round(max(0.0, interval.upper) / baseline_mean, 4) if baseline_mean else 0.0,
    }


def null_accuracy_margin(prompts: Mapping[str, PromptBaseline], *, resamples: int = 10_000) -> dict[str, Any]:
    """How far an identical candidate's correct rate can fall below the baseline by chance alone."""

    noise = null_repeat_noise(prompts, "correct", resamples=resamples)
    return {"null_interval": noise["null_interval"], "margin": round(max(0.0, -noise["null_interval"][0]), 4)}


def propose_thresholds(
    prompts: Mapping[str, PromptBaseline], rules: OwnerRules | None = None, *, resamples: int = 10_000
) -> tuple[PromotionThresholds, dict[str, Any], list[str]]:
    """Thresholds for this population, the measurements behind them, and one sentence per field."""

    rules = rules or OwnerRules()
    accuracy = accuracy_calibration(prompts, rules.accuracy_requirement_candidates)
    # The accuracy bar is the agent's own jitter, like the efficiency floors: the gain a skill must
    # show before it is more than the correct rate's swing between repeats of the same prompts.
    accuracy_noise = null_repeat_noise(prompts, "correct", resamples=resamples)
    minimum_accuracy_improvement = round(accuracy_noise["noise_floor"], 3)
    margin_facts = null_accuracy_margin(prompts, resamples=resamples)
    margin = max(rules.minimum_non_inferiority_margin, margin_facts["margin"])
    floor_facts = candidate_floor(prompts, margin=margin, resamples=resamples)
    latency = null_repeat_noise(prompts, "latency_ms", resamples=resamples)
    cost = null_repeat_noise(prompts, "cost_usd", resamples=resamples)
    thresholds = PromotionThresholds(
        minimum_complete_pairs_per_frozen_case=rules.minimum_complete_pairs_per_frozen_case,
        minimum_accuracy_improvement=minimum_accuracy_improvement,
        minimum_latency_improvement_fraction=rules.latency_benefit_fraction,
        minimum_cost_improvement_fraction=rules.cost_benefit_fraction,
        latency_improvement_noise_floor_ms=math.ceil(latency["noise_floor"] / 100) * 100,
        cost_improvement_noise_floor_usd=math.ceil(cost["noise_floor"] * 1000) / 1000,
        minimum_candidate_correct_rate=floor_facts["floor"],
        maximum_cost_regression=rules.maximum_cost_regression,
        maximum_latency_regression=rules.maximum_latency_regression,
        non_inferiority_margin=round(margin, 4),
    )
    n = accuracy["prompt_count"]
    headroom = round(1 - accuracy["baseline_correct_rate"], 3)
    prompts_for = {
        effect: prompts_needed_to_clear(minimum_accuracy_improvement, true_effect=effect, target_sd=0.35)
        for effect in (0.35, 0.23)
    }
    reasoning = [
        f"minimum_complete_pairs_per_frozen_case = {rules.minimum_complete_pairs_per_frozen_case}: repeats only "
        "shrink within-prompt noise; the prompt-to-prompt spread dominates, and three is the least that "
        "separates consistent from intermittent behaviour.",
        f"minimum_accuracy_improvement = {minimum_accuracy_improvement}: between repeats of the same prompts an "
        f"unchanged agent's correct rate moved {accuracy_noise['null_interval']} "
        f"(SE {accuracy_noise['standard_error']}); "
        f"the bar is 2.8 x that (1.96 for 95% confidence + 0.84 for 80% power), the gain a skill must show to be more "
        f"than the agent's own swing. The random baseline ({accuracy['baseline_correct_rate']} over {n} prompts, "
        f"headroom {headroom}) cannot exhibit such a gain; the bar binds on a candidate's held-out prompts, which are "
        f"chosen for headroom: a true +0.35 effect clears it with about {prompts_for[0.35]} target prompts, a true "
        f"+0.23 effect with about {prompts_for[0.23]}.",
        f"non_inferiority_margin = {thresholds.non_inferiority_margin}: an identical candidate's correct rate moved "
        f"{margin_facts['null_interval']} between repeats; the margin is the larger of that drop and the "
        f"{rules.minimum_non_inferiority_margin} the owner's rules set as a minimum.",
        f"minimum_candidate_correct_rate = {thresholds.minimum_candidate_correct_rate}: the baseline's bootstrap lower "
        f"bound {floor_facts['baseline_lower_bound']} minus the margin, rounded down to 0.05.",
        f"minimum_latency_improvement_fraction = {rules.latency_benefit_fraction}: the owner's rule; a wait users feel "
        f"is about a third off ({latency['baseline_mean'] / 1000:.1f} s -> "
        f"{latency['baseline_mean'] * (1 - rules.latency_benefit_fraction) / 1000:.1f} s).",
        f"latency_improvement_noise_floor_ms = {thresholds.latency_improvement_noise_floor_ms}: between repeats of the "
        f"same prompt latency moved {latency['null_interval']} ms (SE {latency['standard_error']} ms); "
        "the floor is that noise at 80% power, rounded up to 100 ms.",
        f"maximum_latency_regression = {rules.maximum_latency_regression}: the owner's rule (up to half again "
        f"as slow is tolerable); the null upper bound on relative regression is "
        f"{latency['null_relative_regression_upper']}, so the cap sits well clear of noise.",
        f"minimum_cost_improvement_fraction = {rules.cost_benefit_fraction}: mirrors the latency rule on a "
        f"${cost['baseline_mean']:.3f} mean call.",
        f"cost_improvement_noise_floor_usd = {thresholds.cost_improvement_noise_floor_usd}: between repeats cost moved "
        f"{cost['null_interval']} $ (SE {cost['standard_error']}); the floor is that noise at 80% power.",
        f"maximum_cost_regression = {rules.maximum_cost_regression}: the owner's rule (a "
        f"${cost['baseline_mean']:.2f} call may cost up to "
        f"${cost['baseline_mean'] * (1 + rules.maximum_cost_regression):.2f}); null upper bound "
        f"{cost['null_relative_regression_upper']}.",
        "confidence_level = 0.95 and bootstrap_resamples = 10000: the defaults; two-sided intervals, resample noise "
        "flat beyond 10k.",
        f"accuracy_gain_interval_floor = {thresholds.accuracy_gain_interval_floor}: the gain's range must stay "
        "above zero, so it is real; its size is judged on the average against minimum_accuracy_improvement "
        "(not both on the low end, which would count the power margin twice).",
        f"candidate_floor_on_interval = {thresholds.candidate_floor_on_interval}: the candidate's own correct rate "
        "is checked on its average, not its range's low end; the range already gates the gain.",
        f"minimum_distinct_prompts = {thresholds.minimum_distinct_prompts}: below that many prompts a paired range "
        "is too wide to judge, so the gate declines to rule rather than reject.",
    ]
    facts = {
        "accuracy": accuracy,
        "candidate_floor": floor_facts,
        "non_inferiority": margin_facts,
        "latency": latency,
        "cost": cost,
        "resolved_bars": {
            "latency_benefit_ms": round(
                max(
                    thresholds.latency_improvement_noise_floor_ms,
                    rules.latency_benefit_fraction * latency["baseline_mean"],
                ),
                1,
            ),
            "cost_benefit_usd": round(
                max(thresholds.cost_improvement_noise_floor_usd, rules.cost_benefit_fraction * cost["baseline_mean"]), 4
            ),
            "latency_cap_ms": round(rules.maximum_latency_regression * latency["baseline_mean"], 1),
            "cost_cap_usd": round(rules.maximum_cost_regression * cost["baseline_mean"], 4),
        },
    }
    return thresholds, facts, reasoning


__all__ = [
    "OwnerRules",
    "PromptBaseline",
    "accuracy_calibration",
    "baselines_from_run",
    "candidate_floor",
    "null_accuracy_margin",
    "null_repeat_noise",
    "propose_thresholds",
]
