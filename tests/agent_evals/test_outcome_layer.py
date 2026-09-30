"""Outcome scorers, state checks, weighted partial credit, and repeatability (pass@k, pass^k)."""

from __future__ import annotations

import itertools
import math

import pytest

from agent_evals import (
    Contains,
    EvaluationCase,
    EvaluationVariant,
    ExactMatch,
    NumericMatch,
    PredictionResult,
    RegexMatch,
    RunSettings,
    Tolerance,
    WeightedRubric,
    consistency_label,
    expected_pass_hat_k,
    pass_at_k,
    pass_hat_k,
    repeatability,
    run_repeated,
    score_rubric,
)

# Recorded from the campaign's DecisionTolerance and _consistency on these inputs.
TOLERANCE_PAIRS = [
    (0.0665, 0.066), (0.0666, 0.066), (100.0, 102.0), (100.0, 103.0), (0.4, 0.0), (0.6, 0.0), (1000.0, 1010.0),
    (1000.0, 1021.0), (0.0, 0.0), (5.0, -5.0), (0.05, 0.0), (math.nan, 1.0), (math.inf, 1.0), (2.5, 2.5),
]  # fmt: skip
T, F = True, False
TOLERANCE_GOLDEN = {
    "absolute_0.0005": [T, F, F, F, F, F, F, F, T, F, F, F, F, T],
    "relative_0.02_floor_0.5": [T, T, T, F, T, F, T, F, T, F, T, F, F, T],
    "relative_0.05": [T, T, T, T, F, F, T, T, T, F, F, F, F, T],
}
TOLERANCES = {
    "absolute_0.0005": Tolerance("absolute", 0.0005),
    "relative_0.02_floor_0.5": Tolerance("relative", 0.02, floor=0.5),
    "relative_0.05": Tolerance("relative", 0.05),
}


def _case(expected: object, case_id: str = "c1") -> EvaluationCase:
    return EvaluationCase(case_id, {}, expected=expected)


def test_exact_match_ignores_case_and_spacing_by_default() -> None:
    assert ExactMatch()(_case("Paris"), "  paris ") == {"exact_match": 1.0}
    assert ExactMatch()(_case("Paris"), "Lyon") == {"exact_match": 0.0}


def test_exact_match_can_be_strict() -> None:
    assert ExactMatch(normalize=False)(_case("Paris"), "paris") == {"exact_match": 0.0}
    assert ExactMatch(normalize=False)(_case(42), 42) == {"exact_match": 1.0}


def test_scorers_read_the_answer_of_a_prediction_result_a_string_or_a_number() -> None:
    assert ExactMatch()(_case("yes"), PredictionResult(answer="Yes")) == {"exact_match": 1.0}
    assert ExactMatch()(_case("yes"), PredictionResult()) == {"exact_match": 0.0}
    assert ExactMatch()(_case("7"), 7) == {"exact_match": 1.0}


def test_contains_needs_every_expected_item_when_given_a_list() -> None:
    answer = "Spend rose 12% while clicks fell."

    assert Contains()(_case("12%"), answer) == {"contains": 1.0}
    assert Contains()(_case(["12%", "clicks"]), answer) == {"contains": 1.0}
    assert Contains()(_case(["12%", "impressions"]), answer) == {"contains": 0.0}


def test_regex_match_uses_its_pattern_or_the_cases_expected_value() -> None:
    assert RegexMatch(r"\d{4}-\d{2}-\d{2}")(_case(None), "on 2026-09-29") == {"regex_match": 1.0}
    assert RegexMatch()(_case(r"^done$"), "done") == {"regex_match": 1.0}
    assert RegexMatch()(_case(r"^done$"), "not done") == {"regex_match": 0.0}


@pytest.mark.parametrize("name", list(TOLERANCES))
def test_a_tolerance_gives_what_the_campaign_decision_tolerance_gave(name: str) -> None:
    matched = [TOLERANCES[name].matches(claimed, expected) for claimed, expected in TOLERANCE_PAIRS]

    assert matched == TOLERANCE_GOLDEN[name]


def test_a_numeric_answer_within_tolerance_scores_one_and_outside_scores_zero() -> None:
    scorer = NumericMatch(Tolerance("relative", 0.02, floor=0.5))

    assert scorer(_case(100.0), "101.5") == {"numeric_match": 1.0}
    assert scorer(_case(100.0), "103") == {"numeric_match": 0.0}
    assert scorer(_case(1234.5), PredictionResult(answer="1,234.5")) == {"numeric_match": 1.0}


def test_an_answer_that_is_not_a_number_scores_zero_and_a_bool_is_not_a_number() -> None:
    scorer = NumericMatch(Tolerance("absolute", 1.0))

    assert scorer(_case(1.0), "about one") == {"numeric_match": 0.0}
    assert scorer(_case(1.0), True) == {"numeric_match": 0.0}


def test_a_case_without_a_numeric_expected_value_is_an_error_not_a_zero() -> None:
    with pytest.raises(ValueError, match="case c1 has no numeric expected value"):
        NumericMatch(Tolerance("absolute", 1.0))(_case("n/a"), "3")


def test_a_tolerance_that_makes_no_sense_is_refused() -> None:
    with pytest.raises(ValueError, match="must be 'absolute' or 'relative'"):
        Tolerance("percent", 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="value must be finite and non-negative"):
        Tolerance("absolute", -1.0)
    with pytest.raises(ValueError, match="floor must be finite and non-negative"):
        Tolerance("relative", 0.1, floor=math.nan)


async def test_success_is_judged_on_the_state_of_the_world_and_not_on_the_agents_claim() -> None:
    database: set[str] = {"order-1"}  # order-2 was never created

    def order_exists(case: EvaluationCase, output: PredictionResult) -> bool:
        return case.inputs["order"] in database

    cases = [EvaluationCase("c1", {"order": "order-1"}), EvaluationCase("c2", {"order": "order-2"})]

    def claims_done(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
        return PredictionResult(answer="Done, the order is booked.")

    run = await run_repeated(cases, [EvaluationVariant("v")], claims_done, order_exists)

    assert [row.result.metrics for row in run.rows] == [{"order_exists": 1.0}, {"order_exists": 0.0}]
    assert all(row.answer == "Done, the order is booked." for row in run.rows)


async def test_a_state_check_may_be_a_coroutine_function() -> None:
    async def file_written(case: EvaluationCase, output: PredictionResult) -> bool:
        return output.answer == "wrote it"

    run = await run_repeated(
        [_case(None)], [EvaluationVariant("v")], lambda case, variant: PredictionResult(answer="wrote it"), file_written
    )

    assert run.rows[0].result.metrics == {"file_written": 1.0}


WEIGHTS = {
    "factual_numerical_correctness": 0.40,
    "scope_correctness": 0.20,
    "evidence_grounding": 0.15,
    "completeness": 0.15,
    "interpretation_correctness": 0.10,
}
STATUS_SCORES = {"passed": 1.0, "partial": 0.5, "failed": 0.0, "not_applicable": None}


def test_the_weighted_rubric_agrees_with_the_learning_control_planes_on_every_status_combination() -> None:
    from learning_control_plane.evaluation.verification import VerificationCheck, score_final_answer

    rubric = WeightedRubric(WEIGHTS, minimum_score=0.85, required_full_score=("factual_numerical_correctness",))
    for statuses in itertools.product(STATUS_SCORES, repeat=len(WEIGHTS)):
        for failure_codes in ((), ("policy_violation",)):
            checks = {name: VerificationCheck(name, status) for name, status in zip(WEIGHTS, statuses, strict=True)}
            expected = score_final_answer(checks, hard_failure_codes=failure_codes)
            actual = score_rubric(
                rubric,
                {name: STATUS_SCORES[status] for name, status in zip(WEIGHTS, statuses, strict=True)},
                failure_codes=failure_codes,
            )

            assert actual.score == expected.score, (statuses, failure_codes)
            assert actual.passed == expected.passed, (statuses, failure_codes)
            assert dict(actual.effective_weights) == expected.effective_weights


def test_a_criterion_that_does_not_apply_leaves_the_denominator() -> None:
    rubric = WeightedRubric({"a": 0.5, "b": 0.25, "c": 0.25}, minimum_score=0.8)

    result = score_rubric(rubric, {"a": 1.0, "b": None, "c": 0.5})

    assert result.applicable == ("a", "c")
    assert result.score == pytest.approx((0.5 * 1.0 + 0.25 * 0.5) / 0.75)
    assert dict(result.effective_weights) == {"a": pytest.approx(2 / 3), "c": pytest.approx(1 / 3)}


def test_a_required_criterion_that_does_not_apply_cannot_satisfy_the_requirement() -> None:
    rubric = WeightedRubric({"a": 0.5, "b": 0.5}, minimum_score=0.0, required_full_score=("a",))

    assert score_rubric(rubric, {"a": None, "b": 1.0}).passed is False
    assert score_rubric(rubric, {"a": 1.0, "b": 0.0}).passed is True


def test_a_failure_code_fails_an_answer_however_well_it_scored() -> None:
    rubric = WeightedRubric({"a": 1.0}, minimum_score=0.5)

    result = score_rubric(rubric, {"a": 1.0}, failure_codes=["invented_evidence"])

    assert result.score == 1.0
    assert result.passed is False
    assert result.failure_codes == ("invented_evidence",)


def test_a_rubric_with_no_applicable_criterion_scores_zero() -> None:
    rubric = WeightedRubric({"a": 1.0})

    assert score_rubric(rubric, {"a": None}).score == 0.0


def test_scores_that_do_not_match_the_rubrics_criteria_are_refused() -> None:
    rubric = WeightedRubric({"a": 1.0, "b": 1.0})

    with pytest.raises(ValueError, match=r"missing=\['b'\], extra=\['z'\]"):
        score_rubric(rubric, {"a": 1.0, "z": 1.0})


def test_a_rubric_that_makes_no_sense_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one criterion"):
        WeightedRubric({})
    with pytest.raises(ValueError, match="weight must be positive and finite: a"):
        WeightedRubric({"a": 0.0})
    with pytest.raises(ValueError, match="minimum_score must be between 0 and 1"):
        WeightedRubric({"a": 1.0}, minimum_score=1.5)
    with pytest.raises(ValueError, match="required_full_score must name rubric criteria"):
        WeightedRubric({"a": 1.0}, required_full_score=("b",))


def test_three_straight_wins_at_75_percent_per_attempt_is_about_42_percent() -> None:
    assert expected_pass_hat_k(0.75, 3) == 0.421875


def test_pass_hat_k_falls_as_k_grows_while_pass_at_k_rises() -> None:
    hat = [pass_hat_k(10, 8, k) for k in (1, 2, 3, 5)]
    at = [pass_at_k(10, 8, k) for k in (1, 2, 3, 5)]

    assert hat == sorted(hat, reverse=True)
    assert at == sorted(at)
    assert hat[0] == pytest.approx(0.8)
    assert at[0] == pytest.approx(0.8)


def test_the_estimators_give_the_exact_counts_they_stand_for() -> None:
    assert pass_hat_k(4, 3, 2) == 0.5  # C(3,2) / C(4,2)
    assert pass_at_k(4, 3, 2) == 1.0  # only one failure, so two attempts cannot both miss
    assert pass_at_k(4, 1, 2) == pytest.approx(0.5)  # 1 - C(3,2)/C(4,2)


def test_every_run_succeeding_or_none_gives_the_extremes() -> None:
    assert (pass_at_k(5, 5, 3), pass_hat_k(5, 5, 3)) == (1.0, 1.0)
    assert (pass_at_k(5, 0, 3), pass_hat_k(5, 0, 3)) == (0.0, 0.0)


def test_counts_that_cannot_happen_are_refused() -> None:
    with pytest.raises(ValueError, match="must not exceed the number of runs"):
        pass_hat_k(3, 2, 4)
    with pytest.raises(ValueError, match="successes <= runs"):
        pass_at_k(3, 4, 1)
    with pytest.raises(ValueError, match="at least 1"):
        pass_at_k(3, 1, 0)
    with pytest.raises(ValueError, match="rate must be between 0 and 1"):
        expected_pass_hat_k(1.5, 2)


CONSISTENCY_CASES = {
    "all_correct": ([1.0, 1.0, 1.0], 0, 0, "consistently_correct"),
    "all_wrong": ([0.0, 0.0, 0.0], 0, 0, "consistently_incorrect"),
    "mixed": ([1.0, 0.0, 1.0], 0, 0, "intermittent"),
    "partial": ([0.6, 0.7], 0, 0, "intermittent"),
    "one_failed": ([1.0, 1.0], 1, 0, "intermittent"),
    "one_missing": ([1.0, 1.0, 1.0], 0, 1, "intermittent"),
    "single_correct": ([1.0], 0, 0, "consistently_correct"),
    "single_wrong": ([0.0], 0, 0, "consistently_incorrect"),
    "wrong_with_fail": ([0.0, 0.0], 1, 0, "intermittent"),
}


@pytest.mark.parametrize("name", list(CONSISTENCY_CASES))
def test_a_cases_consistency_label_matches_the_campaigns(name: str) -> None:
    scores, failed, missing, label = CONSISTENCY_CASES[name]

    assert consistency_label(scores, failed=failed, missing=missing) == label


def test_no_scores_at_all_is_intermittent() -> None:
    assert consistency_label([]) == "intermittent"


async def _scripted_run(outcomes: dict[str, list[float | None]]):
    """One case per key; each repeat takes the next outcome, and None makes that repeat fail."""

    queues = {case_id: iter(values) for case_id, values in outcomes.items()}
    cases = [_case("x", case_id) for case_id in outcomes]

    def runner(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
        value = next(queues[case.case_id])
        if value is None:
            raise RuntimeError("tool down")
        return PredictionResult(answer=str(value))

    def scorer(case: EvaluationCase, output: PredictionResult) -> dict[str, float]:
        return {"correct": float(output.answer)}  # type: ignore[arg-type]

    repeats = len(next(iter(outcomes.values())))
    return await run_repeated(cases, [EvaluationVariant("v")], runner, scorer, RunSettings(repeats=repeats))


async def test_repeatability_reports_each_case_and_the_average() -> None:
    run = await _scripted_run(
        {"steady": [1.0, 1.0, 1.0, 1.0], "flaky": [1.0, 0.0, 1.0, 1.0], "broken": [0.0, 0.0, 0.0, 0.0]}
    )

    report = repeatability(run, "v", "correct", k=2)

    assert report.cases["steady"].pass_hat_k == 1.0
    assert report.cases["flaky"].pass_hat_k == 0.5  # C(3,2) / C(4,2)
    assert report.cases["broken"].pass_at_k == 0.0
    assert report.pass_hat_k == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert report.pass_at_k == pytest.approx((1.0 + 1.0 + 0.0) / 3)
    assert {case_id: case.consistency for case_id, case in report.cases.items()} == {
        "steady": "consistently_correct",
        "flaky": "intermittent",
        "broken": "consistently_incorrect",
    }
    assert report.consistency_counts() == {"consistently_correct": 1, "intermittent": 1, "consistently_incorrect": 1}


async def test_a_run_that_failed_is_never_a_success_and_makes_its_case_intermittent() -> None:
    run = await _scripted_run({"c1": [1.0, 1.0, None, 1.0]})

    report = repeatability(run, "v", "correct", k=2)

    case = report.cases["c1"]
    assert (case.runs, case.successes) == (4, 3)
    assert case.consistency == "intermittent"


async def test_repeatability_at_k_of_one_is_the_plain_success_rate() -> None:
    run = await _scripted_run({"a": [1.0, 0.0, 1.0, 1.0], "b": [0.0, 0.0, 1.0, 0.0]})

    report = repeatability(run, "v", "correct", k=1)

    assert report.pass_at_k == pytest.approx(report.pass_hat_k)
    assert report.pass_hat_k == pytest.approx((0.75 + 0.25) / 2)


async def test_a_success_threshold_below_one_counts_partial_credit_as_success() -> None:
    run = await _scripted_run({"c1": [0.9, 0.6, 0.9]})

    strict = repeatability(run, "v", "correct", k=1)
    lenient = repeatability(run, "v", "correct", k=1, success_threshold=0.5)

    assert strict.cases["c1"].successes == 0
    assert lenient.cases["c1"].successes == 3


async def test_repeatability_needs_at_least_k_runs_of_every_case() -> None:
    run = await _scripted_run({"c1": [1.0, 1.0]})

    with pytest.raises(ValueError, match="case c1 has 2 runs, fewer than k=3"):
        repeatability(run, "v", "correct", k=3)


async def test_repeatability_of_a_metric_the_run_did_not_produce_is_an_error() -> None:
    run = await _scripted_run({"c1": [1.0, 1.0]})

    with pytest.raises(ValueError, match="did not produce metric 'other'"):
        repeatability(run, "v", "other", k=1)
    with pytest.raises(ValueError, match="no rows for variant 'ghost'"):
        repeatability(run, "ghost", "correct", k=1)
