"""Percentiles, step counts and bands, efficiency, loop detection, and cost regression."""

from __future__ import annotations

import pytest

from agent_evals import (
    EvaluationCase,
    EvaluationVariant,
    ExecutionEfficiency,
    GenericStep,
    GenericTrajectory,
    LoopGuard,
    NoLoop,
    PredictionResult,
    RunSettings,
    StepBand,
    StepCount,
    WithinStepBand,
    action_fingerprint,
    clustered_percentile,
    find_loop,
    operational_summary,
    percentile,
    regression_status,
    run_repeated,
    step_band_from_baseline,
)


def _steps(*calls: str | tuple[str, dict]) -> list[GenericStep]:
    return [GenericStep(call) if isinstance(call, str) else GenericStep(call[0], call[1]) for call in calls]


def _output(*calls: str | tuple[str, dict]) -> PredictionResult:
    return PredictionResult(answer="a", trajectory=GenericTrajectory(query="q", steps=_steps(*calls), final_answer="a"))


def _case(expected: object = None, category: str | None = None, case_id: str = "c1") -> EvaluationCase:
    inputs = {"category": category} if category else {}
    return EvaluationCase(case_id, inputs, expected=expected)


def test_a_percentile_is_linearly_interpolated() -> None:
    values = list(range(1, 101))

    assert percentile(values, 0.5) == pytest.approx(50.5)
    assert percentile(values, 0.95) == pytest.approx(95.05)
    assert percentile(values, 0.0) == 1
    assert percentile(values, 1.0) == 100


def test_the_interval_of_a_percentile_contains_the_estimate_and_widens_with_less_data() -> None:
    many = {f"c{n}": [float(n)] for n in range(60)}
    few = {f"c{n}": [float(n)] for n in range(0, 60, 6)}

    wide, narrow = clustered_percentile(few, 0.95, resamples=500), clustered_percentile(many, 0.95, resamples=500)

    assert narrow.lower <= narrow.estimate <= narrow.upper
    assert (wide.upper - wide.lower) > (narrow.upper - narrow.lower)
    assert narrow.sample_size == 60


def test_the_same_seed_gives_the_same_percentile_interval() -> None:
    groups = {f"c{n}": [float(n), float(n) + 1] for n in range(20)}

    assert clustered_percentile(groups, 0.5, seed=4) == clustered_percentile(groups, 0.5, seed=4)


def test_a_percentile_of_nothing_is_an_error() -> None:
    with pytest.raises(ValueError, match="no values"):
        clustered_percentile({"c1": []}, 0.5)
    with pytest.raises(ValueError, match="between 0 and 1"):
        clustered_percentile({"c1": [1.0]}, 1.5)


async def _run(
    latencies: dict[str, float | None], *, cost: float | None = 0.01, steps: int = 2, fail: set[str] = frozenset()
):
    """One case per key; the runner reports that latency, a cost, and `steps` calls."""

    cases = [EvaluationCase(case_id, {"category": "lookup"}) for case_id in latencies]

    def runner(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
        if case.case_id in fail:
            raise RuntimeError("tool down")
        trajectory = GenericTrajectory(query="q", steps=_steps(*(["t"] * steps)), final_answer="a")
        return PredictionResult(answer="a", trajectory=trajectory, latency_ms=latencies[case.case_id], cost_usd=cost)

    def scorer(case: EvaluationCase, output: PredictionResult) -> dict[str, float]:
        return {"ok": 1.0}

    return await run_repeated(cases, [EvaluationVariant("v")], runner, scorer, RunSettings())


async def test_a_summary_gives_tail_latency_and_the_cost_per_task() -> None:
    run = await _run({f"c{n}": float(100 * (n + 1)) for n in range(10)}, cost=0.05, steps=3)

    summary = operational_summary(run, "v", resamples=300)

    assert summary.latency_ms is not None
    assert summary.latency_ms["p50"].estimate == pytest.approx(550.0)
    assert summary.latency_ms["p95"].estimate == pytest.approx(955.0)
    assert summary.latency_ms["p95"].lower <= 955.0 <= summary.latency_ms["p95"].upper
    assert summary.mean_cost_per_task_usd == pytest.approx(0.05)
    assert summary.mean_step_count == 3
    assert (summary.runs, summary.failed_runs) == (10, 0)


async def test_failed_runs_are_counted_and_leave_the_percentiles() -> None:
    run = await _run({"c1": 100.0, "c2": 200.0, "c3": 9_999.0}, fail={"c3"})

    summary = operational_summary(run, "v", resamples=100)

    assert summary.failed_runs == 1
    assert summary.latency_ms is not None
    assert summary.latency_ms["p99"].estimate < 300


async def test_a_runner_that_reports_no_cost_leaves_cost_unmeasured_and_latency_is_timed_by_the_harness() -> None:
    run = await _run({"c1": None, "c2": None}, cost=None)

    summary = operational_summary(run, "v")

    assert summary.cost_usd is None
    assert summary.mean_cost_per_task_usd is None
    assert summary.latency_ms is not None  # measured wall-clock by the executor, since the runner gave none
    assert 0 <= summary.latency_ms["p95"].estimate < 1_000


async def test_a_summary_of_a_variant_with_no_rows_is_an_error() -> None:
    run = await _run({"c1": 1.0})

    with pytest.raises(ValueError, match="no rows for variant 'ghost'"):
        operational_summary(run, "ghost")


def test_a_cost_regression_is_ok_then_an_alert_then_a_block() -> None:
    assert regression_status(1.00, 1.05, alert=0.10, block=0.15) == "ok"
    assert regression_status(1.00, 1.12, alert=0.10, block=0.15) == "alert"
    assert regression_status(1.00, 1.20, alert=0.10, block=0.15) == "block"
    assert regression_status(1.00, 0.50, alert=0.10, block=0.15) == "ok"


def test_the_edge_of_a_threshold_is_not_yet_over_it() -> None:
    assert regression_status(1.0, 1.10, alert=0.10, block=0.15) == "ok"
    assert regression_status(1.0, 1.15, alert=0.10, block=0.15) == "alert"


def test_a_zero_baseline_treats_any_increase_as_plus_100_percent() -> None:
    assert regression_status(0.0, 0.01, alert=0.10, block=0.15) == "block"
    assert regression_status(0.0, 0.0, alert=0.10, block=0.15) == "ok"


def test_an_alert_above_the_block_level_is_refused() -> None:
    with pytest.raises(ValueError, match="0 <= alert <= block"):
        regression_status(1.0, 1.0, alert=0.2, block=0.1)


def test_the_step_scorer_counts_steps_and_failed_steps() -> None:
    steps = [
        GenericStep("a"),
        GenericStep("b", error="boom"),
        GenericStep("c", failure={"code": "x"}),
        GenericStep("d"),
    ]
    output = PredictionResult(answer="a", trajectory=GenericTrajectory(query="q", steps=steps))

    assert StepCount()(_case(), output) == {"step_count": 4.0, "failed_step_count": 2.0}


def test_efficiency_is_optimal_steps_over_steps_taken() -> None:
    scorer = ExecutionEfficiency()

    assert scorer(_case({"optimal_steps": 3}), _output("a", "b", "c")) == {"execution_efficiency": 1.0}
    assert scorer(_case({"optimal_steps": 3}), _output(*["a"] * 12)) == {"execution_efficiency": 0.25}


def test_beating_the_optimum_scores_one_not_more() -> None:
    assert ExecutionEfficiency()(_case({"optimal_steps": 5}), _output("a", "b")) == {"execution_efficiency": 1.0}


def test_efficiency_of_a_run_with_no_steps_depends_on_whether_any_were_needed() -> None:
    no_steps = PredictionResult(answer="a", trajectory=GenericTrajectory(query="q"))

    assert ExecutionEfficiency()(_case({"optimal_steps": 0}), no_steps) == {"execution_efficiency": 1.0}
    assert ExecutionEfficiency()(_case({"optimal_steps": 2}), no_steps) == {"execution_efficiency": 0.0}


def test_efficiency_needs_an_optimum_from_the_case_or_a_rule() -> None:
    with pytest.raises(ValueError, match="no expected"):
        ExecutionEfficiency()(_case(None), _output("a"))
    assert ExecutionEfficiency(optimal_steps=lambda case: 2)(_case(None), _output("a", "b")) == {
        "execution_efficiency": 1.0
    }
    with pytest.raises(ValueError, match="negative optimal"):
        ExecutionEfficiency(optimal_steps=lambda case: -1)(_case(None), _output("a"))


def test_a_step_band_says_whether_a_count_is_under_within_or_over() -> None:
    band = StepBand(7, 11)

    assert [band.position(count) for count in (3, 7, 9, 11, 20)] == ["under", "within", "within", "within", "over"]


def test_a_band_that_makes_no_sense_is_refused() -> None:
    with pytest.raises(ValueError, match="0 <= low <= high"):
        StepBand(5, 2)


def test_a_three_step_run_is_as_much_an_alarm_as_a_twenty_step_one_in_a_seven_to_eleven_class() -> None:
    scorer = WithinStepBand({"*": StepBand(7, 11)})

    assert scorer(_case(), _output(*["a"] * 9)) == {"within_step_band": 1.0}
    assert scorer(_case(), _output(*["a"] * 3)) == {"within_step_band": 0.0}
    assert scorer(_case(), _output(*["a"] * 20)) == {"within_step_band": 0.0}


def test_a_kind_of_task_without_its_own_band_uses_the_pooled_band() -> None:
    scorer = WithinStepBand({"*": StepBand(1, 3), "ranking": StepBand(8, 12)})

    assert scorer(_case(category="ranking"), _output(*["a"] * 10)) == {"within_step_band": 1.0}
    assert scorer(_case(category="ranking"), _output("a")) == {"within_step_band": 0.0}
    assert scorer(_case(category="trend"), _output("a", "b")) == {"within_step_band": 1.0}


async def _baseline_run(step_counts: dict[str, list[int]]):
    """Per case, the step count of each repeat; the category is the case id's letter prefix."""

    cases = [EvaluationCase(case_id, {"category": case_id[0]}) for case_id in step_counts]
    queues = {case_id: iter(counts) for case_id, counts in step_counts.items()}

    def runner(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
        return _output(*["t"] * next(queues[case.case_id]))

    def scorer(case: EvaluationCase, output: PredictionResult) -> dict[str, float]:
        return {"ok": 1.0}

    repeats = len(next(iter(step_counts.values())))
    return await run_repeated(cases, [EvaluationVariant("base")], runner, scorer, RunSettings(repeats=repeats))


async def test_the_expected_band_comes_from_the_baselines_own_step_counts() -> None:
    run = await _baseline_run({"a1": [7, 8, 9], "a2": [8, 9, 10], "a3": [9, 10, 11], "b1": [2, 2, 3]})

    bands = step_band_from_baseline(run, "base", minimum_runs=5)

    assert bands["a"] == StepBand(7, 11)
    assert "b" not in bands  # three runs is too little to say
    assert bands["*"].low <= 2 and bands["*"].high >= 10


async def test_a_baseline_with_no_successful_run_has_no_band() -> None:
    def runner(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
        raise RuntimeError("down")

    run = await run_repeated([_case()], [EvaluationVariant("base")], runner, lambda case, output: {"ok": 1.0})

    with pytest.raises(ValueError, match="no successful rows"):
        step_band_from_baseline(run, "base")


def test_the_same_action_three_times_running_is_a_loop() -> None:
    loop = find_loop(_steps("read", ("lookup", {"id": 1}), ("lookup", {"id": 1}), ("lookup", {"id": 1})))

    assert loop is not None
    assert (loop.start_index, loop.window, loop.repeats) == (1, 1, 3)


def test_the_same_tool_with_different_arguments_is_progress_not_a_loop() -> None:
    assert find_loop(_steps(("lookup", {"id": 1}), ("lookup", {"id": 2}), ("lookup", {"id": 3}))) is None


def test_two_calls_are_not_yet_a_loop() -> None:
    assert find_loop(_steps("a", "a")) is None


def test_a_short_cycle_repeated_three_times_is_a_loop() -> None:
    loop = find_loop(_steps("ask", "answer", "ask", "answer", "ask", "answer"))

    assert loop is not None
    assert (loop.window, loop.repeats) == (2, 3)


def test_repeats_that_are_not_back_to_back_are_not_a_loop() -> None:
    assert find_loop(_steps("a", "b", "a", "c", "a", "d")) is None


def test_a_cycle_longer_than_the_window_is_not_looked_for() -> None:
    cycle = ["a", "b", "c", "d"] * 3

    assert find_loop(_steps(*cycle), max_window=3) is None
    assert find_loop(_steps(*cycle), max_window=4) is not None


def test_loop_settings_that_make_no_sense_are_refused() -> None:
    with pytest.raises(ValueError, match="max_window must be at least 1"):
        find_loop([], max_window=0)
    with pytest.raises(ValueError, match="repeats at least 2"):
        find_loop([], repeats=1)


def test_an_action_is_fingerprinted_by_tool_and_arguments_ignoring_tool_name_case() -> None:
    assert action_fingerprint(GenericStep("Lookup", {"a": 1, "b": 2})) == action_fingerprint(
        GenericStep(" lookup ", {"b": 2, "a": 1})
    )
    assert action_fingerprint(GenericStep("lookup", {"a": 1})) != action_fingerprint(GenericStep("lookup", {"a": 2}))


def test_a_runner_guard_reports_the_loop_as_soon_as_the_third_repeat_arrives() -> None:
    guard = LoopGuard()

    assert guard.record("lookup", {"id": 1}) is None
    assert guard.record("lookup", {"id": 1}) is None
    loop = guard.record("lookup", {"id": 1})
    assert loop is not None and loop.window == 1


def test_a_runner_guard_finds_a_cycle_too_and_stays_quiet_on_progress() -> None:
    cycle_guard, progress_guard = LoopGuard(), LoopGuard()

    cycle = [cycle_guard.record(tool, {}) for tool in ("a", "b", "a", "b", "a", "b")]
    progress = [progress_guard.record("lookup", {"id": n}) for n in range(6)]

    assert cycle[:5] == [None] * 5 and cycle[5] is not None
    assert progress == [None] * 6


def test_the_guard_and_the_offline_finder_agree() -> None:
    calls = [("a", {}), ("b", {}), ("b", {}), ("b", {}), ("c", {})]
    guard = LoopGuard()

    detections = [guard.record(tool, args) for tool, args in calls]
    offline = find_loop([GenericStep(tool, args) for tool, args in calls])

    first_live = next(loop for loop in detections if loop is not None)
    assert offline is not None
    assert (first_live.start_index, first_live.window) == (offline.start_index, offline.window) == (1, 1)


def test_the_loop_scorer_gives_zero_to_a_run_that_repeated_itself() -> None:
    assert NoLoop()(_case(), _output("a", "b", "c")) == {"no_loop": 1.0}
    assert NoLoop()(_case(), _output("a", "a", "a")) == {"no_loop": 0.0}
