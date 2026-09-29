"""The statistics moved out of the gate and the campaign give the numbers they always gave.

Expected values marked "recorded from the old code" were produced by the code before it moved, on the
same inputs, and pasted here as literals.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

import agent_evals
from agent_evals import (
    BootstrapInterval,
    ConfidenceIntervalRequirement,
    EvaluationCase,
    EvaluationVariant,
    MetricConfidenceInterval,
    MetricSpecification,
    PairedCaseResult,
    PairedPrompt,
    PredictionResult,
    PromotionThresholds,
    PromptBaseline,
    RunSettings,
    VariantCaseResult,
    accuracy_improvement_detectability,
    baselines_from_run,
    bootstrap_paired_intervals,
    null_repeat_noise,
    paired_bootstrap,
    prompts_needed_to_clear,
    propose_thresholds,
    run_repeated,
    standard_error_from_interval,
    thresholds_version,
    values_digest,
)

FIXTURE = Path(__file__).parents[1] / "fixtures" / "agent_evals" / "mined_v3_case_metrics.json"

# Recorded from the old code (the gate's private bootstrap) on the metrics-only fixture above.
GATE_GOLDEN = {
    ("correct_answer_rate", "mean_improvement"): (0.22666666666666668, 0.10666666666666669, 0.36),
    ("correct_answer_rate", "candidate_mean"): (0.9866666666666667, 0.96, 1.0),
    ("estimated_llm_cost_usd", "relative_mean_regression"): (
        0.10135617802407121,
        0.02175778007693312,
        0.19454068967695853,
    ),
    ("agent_execution_latency_ms", "relative_mean_regression"): (
        0.15181298182014447,
        0.07025090034913654,
        0.2389384636793438,
    ),
    ("correct_answer_rate", "relative_mean_improvement"): (
        0.29824561403508776,
        0.12121212121212123,
        0.5777777777777778,
    ),
}
LOWER_IS_BETTER = {"estimated_llm_cost_usd", "agent_execution_latency_ms"}


def _specification(name: str) -> MetricSpecification:
    return MetricSpecification(name, "lower_is_better" if name in LOWER_IS_BETTER else "higher_is_better")


def _fixture_pairs() -> dict[str, tuple[PairedCaseResult, ...]]:
    rows = json.loads(FIXTURE.read_text())
    by_key = {(row["case"], row["arm"], row["repeat"]): row for row in rows}

    def metrics(row: dict) -> dict[str, float]:
        return {
            "correct_answer_rate": 1.0 if row["correct"] else 0.0,
            "estimated_llm_cost_usd": row["cost_usd"],
            "agent_execution_latency_ms": float(row["latency_ms"]),
        }

    pairs: dict[str, tuple[PairedCaseResult, ...]] = {}
    for case in sorted({row["case"] for row in rows}):
        repeats = sorted({row["repeat"] for row in rows if row["case"] == case})
        pairs[case] = tuple(
            PairedCaseResult(
                f"{case}-r{repeat}",
                VariantCaseResult("baseline", metrics=metrics(by_key[(case, "baseline", repeat)])),
                VariantCaseResult("cand", metrics=metrics(by_key[(case, "candidate", repeat)])),
            )
            for repeat in repeats
        )
    return pairs


def _gate_requirements() -> list[ConfidenceIntervalRequirement]:
    return [
        ConfidenceIntervalRequirement("correct_answer_rate", "mean_improvement", minimum_lower_bound=0.0),
        ConfidenceIntervalRequirement("correct_answer_rate", "candidate_mean", minimum_lower_bound=0.0),
        ConfidenceIntervalRequirement("estimated_llm_cost_usd", "relative_mean_regression", maximum_upper_bound=1.0),
        ConfidenceIntervalRequirement(
            "agent_execution_latency_ms", "relative_mean_regression", maximum_upper_bound=0.5
        ),
        ConfidenceIntervalRequirement("correct_answer_rate", "relative_mean_improvement", minimum_lower_bound=0.0),
    ]


def _intervals(pairs: dict[str, tuple[PairedCaseResult, ...]], case_ids: list[str]) -> list[MetricConfidenceInterval]:
    return list(
        bootstrap_paired_intervals(
            pairs_by_cluster=pairs,
            cluster_ids=case_ids,
            requirements=_gate_requirements(),
            metric_specification=_specification,
            confidence_level=0.95,
            resamples=10_000,
        )
    )


def test_the_gate_bootstrap_reproduces_the_numbers_the_old_code_gave() -> None:
    pairs = _fixture_pairs()

    intervals = _intervals(pairs, sorted(pairs))

    assert len(intervals) == len(GATE_GOLDEN)
    for interval in intervals:
        estimate, lower, upper = GATE_GOLDEN[(interval.metric_name, interval.statistic)]
        assert (interval.estimate, interval.lower_bound, interval.upper_bound) == (estimate, lower, upper)


def test_the_gate_numbers_read_as_the_decision_record_states_them() -> None:
    pairs = _fixture_pairs()
    by_name = {(i.metric_name, i.statistic): i for i in _intervals(pairs, sorted(pairs))}

    accuracy = by_name[("correct_answer_rate", "mean_improvement")]
    candidate = by_name[("correct_answer_rate", "candidate_mean")]
    assert [round(value, 3) for value in (accuracy.estimate, accuracy.lower_bound, accuracy.upper_bound)] == [
        0.227,
        0.107,
        0.36,
    ]
    assert [round(value, 3) for value in (candidate.estimate, candidate.lower_bound, candidate.upper_bound)] == [
        0.987,
        0.96,
        1.0,
    ]


def test_the_same_evidence_always_gives_the_same_intervals() -> None:
    pairs = _fixture_pairs()

    assert _intervals(pairs, sorted(pairs)) == _intervals(pairs, sorted(pairs))


def test_the_point_estimate_does_not_depend_on_the_order_the_cases_are_listed_in() -> None:
    pairs = _fixture_pairs()
    forward = _intervals(pairs, sorted(pairs))
    backward = _intervals(pairs, sorted(pairs, reverse=True))

    assert [i.estimate for i in forward] == pytest.approx([i.estimate for i in backward], abs=1e-12)


def test_an_interval_from_evidence_with_no_spread_has_no_width() -> None:
    same = {
        f"case-{n}": (
            PairedCaseResult(
                f"case-{n}",
                VariantCaseResult("baseline", metrics={"correct_answer_rate": 0.5}),
                VariantCaseResult("cand", metrics={"correct_answer_rate": 0.75}),
            ),
        )
        for n in range(6)
    }

    (interval,) = bootstrap_paired_intervals(
        pairs_by_cluster=same,
        cluster_ids=sorted(same),
        requirements=[ConfidenceIntervalRequirement("correct_answer_rate", "mean_improvement", minimum_lower_bound=0)],
        metric_specification=_specification,
        confidence_level=0.95,
        resamples=1_000,
    )

    assert interval.estimate == interval.lower_bound == interval.upper_bound == 0.25


def test_a_resample_whose_baseline_scored_nothing_still_has_a_defined_relative_change() -> None:
    zero_baseline = {
        f"case-{n}": (
            PairedCaseResult(
                f"case-{n}",
                VariantCaseResult("baseline", metrics={"correct_answer_rate": 0.0}),
                VariantCaseResult("cand", metrics={"correct_answer_rate": 1.0 if n % 2 else 0.0}),
            ),
        )
        for n in range(4)
    }

    (interval,) = bootstrap_paired_intervals(
        pairs_by_cluster=zero_baseline,
        cluster_ids=sorted(zero_baseline),
        requirements=[
            ConfidenceIntervalRequirement("correct_answer_rate", "relative_mean_improvement", minimum_lower_bound=0)
        ],
        metric_specification=_specification,
        confidence_level=0.95,
        resamples=500,
    )

    assert interval.estimate == 1.0


def test_the_gate_and_the_package_share_one_interval_type() -> None:
    from learning_control_plane.control_plane import (
        ConfidenceIntervalRequirement as GateRequirement,
    )
    from learning_control_plane.control_plane import (
        MetricConfidenceInterval as GateInterval,
    )
    from learning_control_plane.control_plane.control_plane import _confidence_statistic

    assert GateRequirement is ConfidenceIntervalRequirement
    assert GateInterval is MetricConfidenceInterval
    assert _confidence_statistic is agent_evals.confidence_statistic


def test_a_requirement_needs_exactly_one_bound() -> None:
    with pytest.raises(ValueError, match="exactly one lower or upper bound"):
        ConfidenceIntervalRequirement("m", "mean_improvement")
    with pytest.raises(ValueError, match="exactly one lower or upper bound"):
        ConfidenceIntervalRequirement("m", "mean_improvement", minimum_lower_bound=0, maximum_upper_bound=1)
    with pytest.raises(ValueError, match="lower bound must be finite"):
        ConfidenceIntervalRequirement("m", "mean_improvement", minimum_lower_bound=float("nan"))


def test_the_plain_bootstrap_gives_an_interval_around_the_mean_difference() -> None:
    prompts = [PairedPrompt(f"p{n}", 0.4, 0.4 + 0.1 * (n % 4)) for n in range(12)]

    interval = paired_bootstrap(prompts, resamples=2_000)

    assert interval is not None
    assert interval.prompt_count == 12
    assert interval.mean_difference == pytest.approx(0.15)
    assert interval.lower <= interval.mean_difference <= interval.upper


def test_the_plain_bootstrap_has_nothing_to_say_about_no_prompts() -> None:
    assert paired_bootstrap([]) is None


def test_the_plain_bootstrap_repeats_for_a_seed_and_differs_for_another() -> None:
    prompts = [PairedPrompt(f"p{n}", 0.0, (n * 37 % 10) / 10) for n in range(15)]

    assert paired_bootstrap(prompts, seed=1) == paired_bootstrap(prompts, seed=1)
    assert paired_bootstrap(prompts, seed=1) != paired_bootstrap(prompts, seed=2)


def test_a_standard_error_is_the_half_width_over_1_96() -> None:
    interval = BootstrapInterval(
        prompt_count=10, mean_difference=0.0, lower=-0.196, upper=0.196, resamples=1, confidence=0.95
    )

    assert standard_error_from_interval(interval) == pytest.approx(0.196 / 1.959963985)


def test_prompts_needed_to_clear_a_bar_gives_what_the_old_code_gave() -> None:
    # The campaign's notes say "about 17" and "about 77"; the code, before and after the move, gives these.
    assert prompts_needed_to_clear(0.115, true_effect=0.35, target_sd=0.35) == 18
    assert prompts_needed_to_clear(0.115, true_effect=0.23, target_sd=0.35) == 73


def test_no_number_of_prompts_helps_an_effect_that_is_not_above_the_bar() -> None:
    assert prompts_needed_to_clear(0.115, true_effect=0.115, target_sd=0.35) is None
    assert prompts_needed_to_clear(0.115, true_effect=0.05, target_sd=0.35) is None


def test_detectability_says_when_the_baseline_leaves_too_little_headroom() -> None:
    # Recorded from the old code on these rates.
    rates = [1.0, 1.0, 1.0, 0.6666666666666666] * 3

    report = accuracy_improvement_detectability(rates, required_accuracy_improvement=0.1)

    assert report["detectable"] is False
    assert report["case_count"] == 12
    assert report["maximum_possible_improvement"] == pytest.approx(1 - sum(rates) / 12)
    assert report["standard_error"] == pytest.approx(0.041666666666666664)
    assert report["minimum_detectable_effect"] == 0.21673271744166667
    assert report["required_case_count"] is None


def test_detectability_has_no_case_count_to_offer_when_there_is_no_headroom() -> None:
    report = accuracy_improvement_detectability([1.0, 1.0, 1.0], required_accuracy_improvement=0.1)

    assert report["required_case_count"] is None
    assert report["detectable"] is False


def test_detectability_needs_at_least_two_cases() -> None:
    with pytest.raises(ValueError, match="at least two measured cases"):
        accuracy_improvement_detectability([0.5], required_accuracy_improvement=0.1)


def _synthetic_baseline() -> dict[str, PromptBaseline]:
    """The population the old code was run on; see GOLDEN_CALIBRATION."""

    prompts = {}
    for i in range(12):
        prompt = PromptBaseline(f"p{i:02d}")
        prompt.correct = [1.0, 1.0, 1.0] if i % 4 else ([1.0, 0.0, 1.0] if i % 8 == 0 else [0.0, 0.0, 1.0])
        prompt.latency_ms = [20000.0 + 1500.0 * i + 700.0 * r * (1 if i % 2 else -1) for r in range(3)]
        prompt.cost_usd = [0.30 + 0.02 * i + 0.01 * r * (1 if i % 3 else -1) for r in range(3)]
        prompts[prompt.trace_id] = prompt
    return prompts


def test_calibration_proposes_the_thresholds_the_old_code_proposed() -> None:
    thresholds, facts, reasoning = propose_thresholds(_synthetic_baseline())

    assert thresholds == PromotionThresholds(
        minimum_accuracy_improvement=0.238,
        latency_improvement_noise_floor_ms=500,
        cost_improvement_noise_floor_usd=0.006,
        minimum_candidate_correct_rate=0.6,
        non_inferiority_margin=0.125,
    )
    assert facts["accuracy"]["baseline_correct_rate"] == 0.8889
    assert facts["accuracy"]["standard_error"] == 0.06
    assert facts["candidate_floor"] == {"baseline_lower_bound": 0.75, "margin": 0.125, "floor": 0.6}
    assert facts["non_inferiority"] == {"margin": 0.125, "null_interval": [-0.125, 0.2083]}
    assert facts["resolved_bars"] == {
        "cost_benefit_usd": 0.1447,
        "cost_cap_usd": 0.4133,
        "latency_benefit_ms": 9887.5,
        "latency_cap_ms": 14125.0,
    }
    assert len(reasoning) == 14


def test_calibration_of_latency_reports_the_noise_the_old_code_measured() -> None:
    noise = null_repeat_noise(_synthetic_baseline(), "latency_ms")

    assert noise == {
        "baseline_mean": 28250.0,
        "half_width": 291.6667,
        "noise_floor": 416.9102,
        "null_interval": [-291.6667, 291.6667],
        "null_relative_regression_upper": 0.0103,
        "standard_error": 148.8123,
    }


def test_the_threshold_version_names_match_the_ones_recorded_by_the_campaign() -> None:
    """Recorded from the campaign's `thresholds_version` for the same values."""

    business_v3 = PromotionThresholds(
        minimum_accuracy_improvement=0.115,
        latency_improvement_noise_floor_ms=5_300.0,
        cost_improvement_noise_floor_usd=0.054,
        minimum_candidate_correct_rate=0.70,
        non_inferiority_margin=0.0806,
    )
    business_v2 = dataclasses.replace(
        business_v3, accuracy_gain_interval_floor=None, candidate_floor_on_interval=True, minimum_distinct_prompts=0
    )
    rules_added_in_v3 = {
        "accuracy_gain_interval_floor": None,
        "candidate_floor_on_interval": True,
        "minimum_distinct_prompts": 0,
    }
    named = {"thresholds.business-v3": business_v3, "thresholds.business-v2": business_v2}

    assert thresholds_version(business_v3, named=named, omit_when=rules_added_in_v3) == "thresholds.business-v3.a67949"
    assert thresholds_version(business_v2, named=named, omit_when=rules_added_in_v3) == "thresholds.business-v2.3744db"
    tweaked = dataclasses.replace(business_v3, minimum_accuracy_improvement=0.2)
    assert thresholds_version(tweaked, named=named, omit_when=rules_added_in_v3) == "thresholds.custom-bc15dfc63ebd"


def test_a_threshold_set_is_named_by_its_values() -> None:
    first = PromotionThresholds(0.1, 100, 0.01, 0.7, 0.05)
    same = PromotionThresholds(0.1, 100, 0.01, 0.7, 0.05)
    other = PromotionThresholds(0.2, 100, 0.01, 0.7, 0.05)

    assert thresholds_version(first) == thresholds_version(same)
    assert thresholds_version(first) != thresholds_version(other)
    assert thresholds_version(first) == f"thresholds.custom-{values_digest(first)}"


def test_the_measured_thresholds_have_no_defaults() -> None:
    with pytest.raises(TypeError):
        PromotionThresholds()  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"minimum_accuracy_improvement": -0.1}, "minimum_accuracy_improvement must be finite and non-negative"),
        ({"non_inferiority_margin": float("nan")}, "non_inferiority_margin must be finite and non-negative"),
        ({"minimum_candidate_correct_rate": 1.5}, "minimum_candidate_correct_rate must not exceed 1"),
        ({"minimum_latency_improvement_fraction": 1.5}, "minimum_latency_improvement_fraction must be between 0 and 1"),
        ({"minimum_complete_pairs_per_frozen_case": 0}, "minimum_complete_pairs_per_frozen_case must be at least 1"),
        ({"minimum_distinct_prompts": -1}, "minimum_distinct_prompts must not be negative"),
        ({"accuracy_gain_interval_floor": float("inf")}, "accuracy_gain_interval_floor must be finite"),
    ],
)
def test_thresholds_that_make_no_sense_are_refused(change: dict, message: str) -> None:
    values = {
        "minimum_accuracy_improvement": 0.1,
        "latency_improvement_noise_floor_ms": 100,
        "cost_improvement_noise_floor_usd": 0.01,
        "minimum_candidate_correct_rate": 0.7,
        "non_inferiority_margin": 0.05,
    }

    with pytest.raises(ValueError, match=message):
        PromotionThresholds(**{**values, **change})


def test_the_accuracy_gain_range_must_clear_the_bar_only_when_no_separate_floor_is_set() -> None:
    values = (0.115, 5_300.0, 0.054, 0.7, 0.0806)

    assert PromotionThresholds(*values).accuracy_gain_range_floor == 0.0
    assert PromotionThresholds(*values, accuracy_gain_interval_floor=None).accuracy_gain_range_floor == 0.115


async def test_a_baseline_is_read_from_a_run_with_failed_repeats_counted_as_incorrect_and_untimed() -> None:
    cases = [EvaluationCase("c1", {}, expected="a"), EvaluationCase("c2", {}, expected="b")]
    variant = EvaluationVariant("base")
    calls: dict[str, int] = {}

    def runner(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
        calls[case.case_id] = calls.get(case.case_id, 0) + 1
        if case.case_id == "c2" and calls["c2"] == 2:
            raise RuntimeError("tool down")
        return PredictionResult(answer=str(case.expected), latency_ms=1000.0 * calls[case.case_id], cost_usd=0.5)

    def scorer(case: EvaluationCase, output: PredictionResult) -> dict[str, float]:
        return {"correct": 1.0 if output.answer == case.expected else 0.0}

    run = await run_repeated(cases, [variant], runner, scorer, RunSettings(repeats=3))
    baselines = baselines_from_run(run, "base")

    assert baselines["c1"].correct == [1.0, 1.0, 1.0]
    assert baselines["c1"].latency_ms == [1000.0, 2000.0, 3000.0]
    assert baselines["c2"].correct == [1.0, 0.0, 1.0]
    assert baselines["c2"].latency_ms == [1000.0, 3000.0]
    assert baselines["c2"].correct_rate == pytest.approx(2 / 3)
