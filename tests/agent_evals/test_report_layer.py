"""The eight-metric scorecard, threshold profiles, reports, and the MLflow log of a report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_evals import (
    EXAMPLE_PROFILES,
    ArgumentCorrectness,
    DatasetManifest,
    EvaluationCase,
    EvaluationDataset,
    EvaluationVariant,
    ExecutionEfficiency,
    GenericStep,
    GenericTrajectory,
    PolicyCheck,
    PolicyVeto,
    PredictionResult,
    PromotionThresholds,
    RunRecord,
    RunSettings,
    ScorecardMetrics,
    ThresholdProfile,
    ToolArguments,
    ToolSelection,
    build_report,
    build_scorecard,
    derived_profile,
    evaluate_profile,
    log_report_to_mlflow,
    operational_summary,
    report_json,
    report_markdown,
    report_metrics,
    run_repeated,
)

SNAPSHOT = Path(__file__).parents[1] / "fixtures" / "agent_evals" / "report_snapshot.md"
EXPECTED = {"tools": ["lookup", "report"], "optimal_steps": 2}
CASE_IDS = [f"c{n}" for n in range(1, 7)]
CASES = [
    EvaluationCase(case_id, {"category": "lookup", "query": "SECRET question text"}, expected=EXPECTED)
    for case_id in CASE_IDS
]
LATENCY = {case_id: 1_000.0 + 100 * index for index, case_id in enumerate(CASE_IDS)}
TOOLS = {
    "c1": ["lookup", "report"], "c2": ["lookup", "report"], "c3": ["lookup", "report"], "c4": ["lookup", "report"],
    "c5": ["lookup"],  # skipped the report
    "c6": ["lookup", "report", "delete"],  # a forbidden call
}  # fmt: skip


def _trajectory(tools: list[str]) -> GenericTrajectory:
    steps = [GenericStep(tool, {"acid": "1"}) for tool in tools]
    return GenericTrajectory(query="q", steps=steps, final_answer="ok")


def _full_runner(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
    return PredictionResult(
        answer="ok", trajectory=_trajectory(TOOLS[case.case_id]), latency_ms=LATENCY[case.case_id], cost_usd=0.05
    )


def _answered(case: EvaluationCase, output: PredictionResult) -> dict[str, float]:
    return {"success": 1.0 if output.answer == "ok" else 0.0}


FULL_SCORERS = [
    PolicyVeto(_answered, PolicyCheck(forbidden_tools=["delete"]), "success"),
    ToolSelection(),
    ArgumentCorrectness({"lookup": ToolArguments(required=("acid",), types={"acid": str})}),
    ExecutionEfficiency(),
]


def _manifest(
    cases: list[EvaluationCase] = CASES, suite: str = "capability"
) -> tuple[EvaluationDataset, DatasetManifest]:
    dataset = EvaluationDataset("questions", "v1", cases)
    return dataset, DatasetManifest.from_dataset(dataset, suite=suite)  # type: ignore[arg-type]


async def _full_run(variant_id: str = "cand"):
    dataset, manifest = _manifest()
    run = await run_repeated(dataset.cases, [EvaluationVariant(variant_id)], _full_runner, FULL_SCORERS)
    return run, manifest


def _record(run, manifest, variant_id: str = "cand") -> RunRecord:
    return RunRecord.from_run(
        run, manifest, variant_id, run_id="run-1", bundle={"model": "m-1"}, metric_versions={"success": "1"},
        verdict_grade=True, created_at="2026-09-29T12:00:00+00:00",
    )  # fmt: skip


async def test_the_scorecard_has_all_eight_entries_in_a_fixed_order() -> None:
    run, _ = await _full_run()

    scorecard = build_scorecard(run, "cand", resamples=300)

    assert [entry.key for entry in scorecard.entries] == [
        "success_rate", "tool_selection", "argument_correctness", "plan_adherence",
        "execution_efficiency", "cost_per_task", "p95_latency", "policy_violation",
    ]  # fmt: skip


async def test_the_scorecard_numbers_are_the_ones_worked_out_by_hand() -> None:
    run, _ = await _full_run()

    scorecard = build_scorecard(run, "cand", resamples=300)

    assert scorecard.entry("success_rate").value == pytest.approx(5 / 6)  # c6 is vetoed
    assert scorecard.entry("tool_selection").value == pytest.approx((4 * 1.0 + 0.5 + 2 / 3) / 6)
    assert scorecard.entry("cost_per_task").value == pytest.approx(0.05)
    assert scorecard.entry("p95_latency").value == pytest.approx(1_475.0)  # linear interpolation, six values
    policy = scorecard.entry("policy_violation")
    assert (policy.value, policy.flag, policy.reason) == (pytest.approx(1 / 6), True, "cases: c6")


async def test_a_measured_entry_carries_a_range_that_contains_its_value() -> None:
    run, _ = await _full_run()

    for entry in build_scorecard(run, "cand", resamples=300).entries:
        if entry.measured and entry.lower is not None:
            assert entry.lower <= entry.value <= entry.upper, entry.key


async def test_plan_adherence_is_listed_as_not_measured_with_the_reason_and_never_left_out() -> None:
    run, manifest = await _full_run()

    report = build_report(run, manifest, "cand", record=_record(run, manifest), resamples=300)

    entry = report.scorecard.entry("plan_adherence")
    assert (entry.measured, entry.value) == (False, None)
    assert entry.reason == "not configured: plan adherence needs a judge"
    assert "| Plan adherence | trajectory | not measured |" in report_markdown(report)


async def test_a_metric_no_scorer_produced_is_not_measured_and_says_which() -> None:
    dataset, manifest = _manifest()
    run = await run_repeated(dataset.cases, [EvaluationVariant("cand")], _full_runner, [_answered])

    scorecard = build_scorecard(run, "cand", resamples=300)

    assert scorecard.entry("tool_selection").reason == "no scorer produced 'tool_selection_accuracy'"
    assert scorecard.entry("policy_violation").reason == "no policy check is configured"
    assert scorecard.entry("success_rate").measured is True


async def test_a_runner_that_reports_no_cost_leaves_cost_not_measured() -> None:
    dataset, manifest = _manifest()

    def no_cost(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
        return PredictionResult(answer="ok", latency_ms=100.0)

    run = await run_repeated(dataset.cases, [EvaluationVariant("cand")], no_cost, [_answered])

    assert build_scorecard(run, "cand", resamples=100).entry("cost_per_task").reason == "the runner reported no cost"


async def test_the_audit_line_puts_success_and_tool_selection_side_by_side() -> None:
    run, _ = await _full_run()

    scorecard = build_scorecard(run, "cand", resamples=300)

    assert len(scorecard.audit) == 1
    assert "success 0.83 against tool_selection_accuracy 0.86" in scorecard.audit[0]


async def _three_agents():
    """Three agents that supply different things, to show the report has one structure for all."""

    dataset, manifest = _manifest()
    full = await run_repeated(dataset.cases, [EvaluationVariant("a")], _full_runner, FULL_SCORERS)

    def minimal_runner(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
        return PredictionResult(answer="ok", latency_ms=500.0)

    minimal = await run_repeated(dataset.cases, [EvaluationVariant("b")], minimal_runner, [_answered])

    def rogue_runner(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
        return PredictionResult(answer="ok", trajectory=_trajectory(["delete"]), latency_ms=900.0, cost_usd=0.4)

    rogue = await run_repeated(
        dataset.cases, [EvaluationVariant("c")], rogue_runner,
        [PolicyVeto(_answered, PolicyCheck(forbidden_tools=["delete"]), "success")],
    )  # fmt: skip
    minimal_metrics = ScorecardMetrics(
        tool_selection=None, argument_correctness=None, execution_efficiency=None, policy=None
    )
    return [
        build_report(full, manifest, "a", record=_record(full, manifest, "a"), resamples=200),
        build_report(
            minimal, manifest, "b", record=_record(minimal, manifest, "b"), metrics=minimal_metrics, resamples=200
        ),
        build_report(rogue, manifest, "c", record=_record(rogue, manifest, "c"), resamples=200),
    ]


def _shape(value):
    if isinstance(value, dict):
        return {key: _shape(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_shape(value[0])] if value else []
    return type(value).__name__ if value is not None else "null"


async def test_three_different_agents_get_the_same_report_structure() -> None:
    reports = await _three_agents()

    payloads = [report_json(report) for report in reports]

    for payload in payloads:
        assert set(payload) == set(payloads[0])
        assert set(payload["scorecard"]) == {"entries", "audit", "failures"}
        assert [entry["key"] for entry in payload["scorecard"]["entries"]] == [
            entry["key"] for entry in payloads[0]["scorecard"]["entries"]
        ]
        assert [set(entry) for entry in payload["scorecard"]["entries"]] == [
            set(entry) for entry in payloads[0]["scorecard"]["entries"]
        ]
        assert set(payload["run"]) == set(payloads[0]["run"])
        assert set(payload["suite"]) == set(payloads[0]["suite"])


async def test_what_each_agent_did_not_supply_shows_as_not_measured_rather_than_disappearing() -> None:
    _, minimal, rogue = await _three_agents()

    assert [e.key for e in minimal.scorecard.entries if not e.measured] == [
        "tool_selection", "argument_correctness", "plan_adherence", "execution_efficiency", "cost_per_task",
        "policy_violation",
    ]  # fmt: skip
    assert rogue.scorecard.entry("policy_violation").flag is True
    assert rogue.scorecard.entry("success_rate").value == 0.0  # every run violated policy, so none is a success


async def test_a_report_is_json_serialisable_and_carries_its_schema_version() -> None:
    run, manifest = await _full_run()
    report = build_report(run, manifest, "cand", record=_record(run, manifest), resamples=200)

    payload = json.loads(json.dumps(report_json(report)))

    assert payload["schema_version"] == "agent-evals.report.v1"
    assert payload["run"]["bundle"] == {"model": "m-1"}
    assert payload["run"]["verdict_grade"] is True


async def test_a_report_holds_ids_and_numbers_and_never_case_text_or_error_messages() -> None:
    dataset, manifest = _manifest()

    def runner(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
        if case.case_id == "c2":
            raise RuntimeError("could not answer: SECRET question text")
        return _full_runner(case, variant)

    run = await run_repeated(dataset.cases, [EvaluationVariant("cand")], runner, [_answered])
    report = build_report(run, manifest, "cand", record=_record(run, manifest), resamples=200)

    text = json.dumps(report_json(report)) + report_markdown(report)

    assert "SECRET" not in text
    assert [(f.case_id, f.error) for f in report.scorecard.failures] == [("c2", "RuntimeError")]


async def test_error_messages_appear_only_when_asked_for() -> None:
    dataset, manifest = _manifest()

    def runner(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
        raise RuntimeError("tool down")

    run = await run_repeated(dataset.cases[:1], [EvaluationVariant("cand")], runner, [_answered])
    scorecard = build_scorecard(run, "cand", resamples=100, include_error_messages=True)

    assert scorecard.failures[0].error == "RuntimeError: tool down"


async def test_failed_runs_are_listed_and_counted_as_excluded_never_silently_dropped() -> None:
    dataset, manifest = _manifest()

    def runner(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
        if case.case_id in ("c1", "c2"):
            raise RuntimeError("down")
        return _full_runner(case, variant)

    run = await run_repeated(dataset.cases, [EvaluationVariant("cand")], runner, FULL_SCORERS)

    scorecard = build_scorecard(run, "cand", resamples=100)

    assert [f.case_id for f in scorecard.failures] == ["c1", "c2"]
    assert scorecard.entry("tool_selection").runs_excluded == 2
    assert scorecard.entry("success_rate").value == pytest.approx(3 / 6)  # a failed run is not a success


async def test_a_regression_suite_report_says_it_passed_or_failed_and_names_the_failing_cases() -> None:
    dataset, manifest = _manifest(suite="regression")
    run = await run_repeated(dataset.cases, [EvaluationVariant("cand")], _full_runner, FULL_SCORERS)

    report = build_report(run, manifest, "cand", record=_record(run, manifest), resamples=100)

    assert "Result: FAILED (83.3% of 6 cases passed; failing: c6)" in report_markdown(report)


async def test_a_capability_suite_report_gives_a_range_and_no_pass_or_fail() -> None:
    run, manifest = await _full_run()

    markdown = report_markdown(build_report(run, manifest, "cand", record=_record(run, manifest), resamples=100))

    assert "no pass or fail for a capability suite" in markdown


async def test_the_markdown_report_matches_the_pinned_snapshot() -> None:
    run, manifest = await _full_run()
    report = build_report(run, manifest, "cand", record=_record(run, manifest), resamples=300, seed=7)

    assert report_markdown(report) == SNAPSHOT.read_text()


async def test_a_run_record_holds_the_settings_and_versions_the_run_used() -> None:
    dataset, manifest = _manifest()
    run = await run_repeated(
        dataset.cases[:2], [EvaluationVariant("cand")], _full_runner, [_answered], RunSettings(repeats=2, concurrency=3)
    )

    record = _record(run, manifest)

    assert record.settings["repeats"] == 2 and record.settings["concurrency"] == 3
    assert (record.dataset_digest, record.suite) == (manifest.digest, "capability")


def test_the_article_profiles_are_labelled_as_examples_and_not_standards() -> None:
    for profile in EXAMPLE_PROFILES.values():
        assert profile.origin == "example"
        assert "proposal, not a measured standard" in profile.note
    assert (
        EXAMPLE_PROFILES["ci_gate"].minimum_success_rate,
        EXAMPLE_PROFILES["production_slo"].minimum_success_rate,
    ) == (0.85, 0.90)


async def test_a_run_is_held_to_a_profile_check_by_check() -> None:
    run, _ = await _full_run()
    summary = operational_summary(run, "cand", resamples=100)

    verdict = evaluate_profile(EXAMPLE_PROFILES["ci_gate"], success_rate=5 / 6, summary=summary, baseline=summary)

    by_name = {check.name: check.status for check in verdict.checks}
    assert by_name == {"success_rate": "fail", "p95_latency": "pass", "cost": "pass"}  # 0.833 is under the 0.85 floor
    assert verdict.passed is False
    assert verdict.origin == "example"


async def test_cost_is_not_measured_against_a_profile_without_a_baseline() -> None:
    run, _ = await _full_run()
    summary = operational_summary(run, "cand", resamples=100)

    verdict = evaluate_profile(EXAMPLE_PROFILES["ci_gate"], success_rate=0.95, summary=summary)

    assert {check.name: check.status for check in verdict.checks}["cost"] == "not_measured"
    assert verdict.passed is True


async def test_a_cost_rise_between_the_alert_and_block_levels_alerts_and_still_passes() -> None:
    run, _ = await _full_run()
    base = operational_summary(run, "cand", resamples=100)
    dataset, _ = _manifest()

    def pricier(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
        return PredictionResult(answer="ok", trajectory=_trajectory(["lookup"]), latency_ms=100.0, cost_usd=0.056)

    dearer = await run_repeated(dataset.cases, [EvaluationVariant("v")], pricier, [_answered])
    verdict = evaluate_profile(
        EXAMPLE_PROFILES["ci_gate"],
        success_rate=1.0,
        summary=operational_summary(dearer, "v", resamples=100),
        baseline=base,
    )

    assert {check.name: check.status for check in verdict.checks}["cost"] == "alert"
    assert verdict.passed is True


async def test_a_profile_with_no_bar_that_could_be_checked_has_no_verdict() -> None:
    run, _ = await _full_run()
    summary = operational_summary(run, "cand", resamples=100)

    verdict = evaluate_profile(ThresholdProfile("empty", "example", "n/a"), success_rate=1.0, summary=summary)

    assert verdict.passed is None


async def test_a_derived_profile_takes_its_bars_from_the_calibration() -> None:
    run, _ = await _full_run()
    baseline = operational_summary(run, "cand", resamples=100)
    thresholds = PromotionThresholds(0.09, 12_400.0, 0.076, 0.65, 0.0484)

    profile = derived_profile("ci_gate", thresholds, baseline=baseline)

    assert profile.origin == "derived"
    assert profile.minimum_success_rate == 0.65
    assert profile.cost_block == 1.0
    assert profile.maximum_p95_latency_ms == pytest.approx(baseline.latency_ms["p95"].estimate * 1.5)  # type: ignore[index]


def test_a_derived_profile_without_a_baseline_has_no_latency_bound() -> None:
    profile = derived_profile("ci_gate", PromotionThresholds(0.09, 12_400.0, 0.076, 0.65, 0.0484))

    assert profile.maximum_p95_latency_ms is None


def test_a_profile_alert_level_needs_a_block_level_and_must_not_exceed_it() -> None:
    with pytest.raises(ValueError, match="cost_alert needs cost_block"):
        ThresholdProfile("p", "example", "n", cost_alert=0.1)
    with pytest.raises(ValueError, match="must not exceed cost_block"):
        ThresholdProfile("p", "example", "n", cost_alert=0.3, cost_block=0.2)


async def test_profile_verdicts_appear_in_the_report_with_their_origin() -> None:
    run, manifest = await _full_run()
    summary = operational_summary(run, "cand", resamples=100)
    verdict = evaluate_profile(EXAMPLE_PROFILES["production_slo"], success_rate=0.95, summary=summary, baseline=summary)

    markdown = report_markdown(
        build_report(run, manifest, "cand", record=_record(run, manifest), profiles=[verdict], resamples=100)
    )

    assert "## Profile: production_slo (example): passed" in markdown


async def test_the_scorecard_of_an_unknown_variant_is_an_error() -> None:
    run, _ = await _full_run()

    with pytest.raises(ValueError, match="no rows for variant 'ghost'"):
        build_scorecard(run, "ghost")


async def test_a_report_logged_to_mlflow_reads_back_with_the_same_aggregates(tmp_path: Path) -> None:
    mlflow = pytest.importorskip("mlflow")
    from mlflow import MlflowClient

    run, manifest = await _full_run()
    report = build_report(run, manifest, "cand", record=_record(run, manifest), resamples=200)
    previous_uri = mlflow.get_tracking_uri()
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    try:
        run_id = log_report_to_mlflow(
            report, tracking_uri=uri, experiment_name="agent-evals-test", artifact_location=str(tmp_path / "artifacts")
        )
        stored = MlflowClient(tracking_uri=uri).get_run(run_id)
        artifact = MlflowClient(tracking_uri=uri).download_artifacts(run_id, "report.json")
    finally:
        mlflow.set_tracking_uri(previous_uri)

    expected = report_metrics(report)
    assert expected["scorecard.success_rate"] == pytest.approx(report.scorecard.entry("success_rate").value)
    assert set(stored.data.metrics) == set(expected)
    for name, value in expected.items():
        assert stored.data.metrics[name] == pytest.approx(value), name
    assert json.loads(Path(artifact).read_text()) == json.loads(json.dumps(report_json(report)))
    assert stored.data.params["dataset_digest"] == manifest.digest


async def test_only_measured_entries_become_mlflow_metrics() -> None:
    run, manifest = await _full_run()
    report = build_report(run, manifest, "cand", record=_record(run, manifest), resamples=100)

    metrics = report_metrics(report)

    assert "scorecard.plan_adherence" not in metrics
    assert "scorecard.success_rate.lower" in metrics
    assert "scorecard.policy_violation.lower" not in metrics  # a share of runs has no range
