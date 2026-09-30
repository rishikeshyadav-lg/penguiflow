"""An agent that misbehaves in every way at once is handled without dropping a single run."""

from __future__ import annotations

import asyncio

from agent_evals import (
    DatasetManifest,
    EvaluationCase,
    EvaluationDataset,
    EvaluationVariant,
    GenericStep,
    GenericTrajectory,
    NoLoop,
    PolicyCheck,
    PolicyVeto,
    PredictionResult,
    RunRecord,
    RunSettings,
    ToolSelection,
    build_report,
    policy_flag,
    report_json,
    report_markdown,
    run_repeated,
)

CASES = [
    EvaluationCase(case_id, {}, expected={"tools": ["lookup"]})
    for case_id in ("fine", "hangs", "malformed", "raises", "loops", "forbidden")
]


def _trajectory(*calls: tuple[str, dict]) -> GenericTrajectory:
    return GenericTrajectory(query="q", steps=[GenericStep(tool, args) for tool, args in calls], final_answer="ok")


async def hostile(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult | str:
    """Each case misbehaves in its own way."""

    if case.case_id == "hangs":
        await asyncio.Event().wait()  # never returns: a timeout
    if case.case_id == "malformed":
        return "this is not a PredictionResult"
    if case.case_id == "raises":
        raise RuntimeError("the tool blew up")
    if case.case_id == "loops":
        return PredictionResult(answer="ok", trajectory=_trajectory(*[("lookup", {"id": 1})] * 4))
    if case.case_id == "forbidden":
        return PredictionResult(
            answer="ok", trajectory=_trajectory(("lookup", {"id": 1}), ("delete_table", {"name": "t"}))
        )
    return PredictionResult(answer="ok", trajectory=_trajectory(("lookup", {"id": 1})))


def _answered(case: EvaluationCase, output: PredictionResult) -> dict[str, float]:
    return {"success": 1.0 if output.answer == "ok" else 0.0}


SCORERS = [
    PolicyVeto(_answered, PolicyCheck(forbidden_tools=["delete_table"]), "success"),
    ToolSelection(),
    NoLoop(),
]


async def _run():
    dataset = EvaluationDataset("hostile", "v1", CASES)
    manifest = DatasetManifest.from_dataset(dataset, suite="regression")
    run = await run_repeated(
        dataset.cases, [EvaluationVariant("hostile")], hostile, SCORERS, RunSettings(timeout_s=0.05, concurrency=4)
    )
    return run, manifest


async def test_every_expected_run_has_a_recorded_result_and_none_is_dropped() -> None:
    run, _ = await _run()

    assert [row.case_id for row in run.rows] == [case.case_id for case in CASES]
    assert len(run.rows) == 6


async def test_each_kind_of_failure_is_recorded_as_an_explicit_error_of_its_own_kind() -> None:
    run, _ = await _run()

    errors = {row.case_id: row.result.error for row in run.rows if row.result.error is not None}

    assert errors["hangs"].startswith("TimeoutError")
    assert errors["malformed"].startswith("AttributeError")  # a string has no `.answer` for the scorer
    assert errors["raises"] == "RuntimeError: the tool blew up"
    assert set(errors) == {"hangs", "malformed", "raises"}


async def test_the_forbidden_call_raises_the_policy_flag_and_is_not_counted_as_a_success() -> None:
    run, _ = await _run()

    flag = policy_flag(run, "hostile")
    row = next(row for row in run.rows if row.case_id == "forbidden")

    assert (flag.raised, flag.case_ids) == (True, ("forbidden",))
    assert row.result.metrics["success"] == 0.0 and row.result.metrics["success_unvetoed"] == 1.0


async def test_the_loop_is_caught_and_a_clean_run_is_not() -> None:
    run, _ = await _run()

    loops = {row.case_id: row.result.metrics["no_loop"] for row in run.rows if row.result.error is None}

    assert loops["loops"] == 0.0
    assert loops["fine"] == 1.0


async def test_the_report_lists_the_failures_and_shows_the_flag_and_still_has_all_eight_entries() -> None:
    run, manifest = await _run()
    record = RunRecord.from_run(run, manifest, "hostile", run_id="hostile-1", created_at="2026-09-29T00:00:00Z")

    report = build_report(run, manifest, "hostile", record=record, resamples=100)

    assert len(report.scorecard.entries) == 8
    assert sorted(f.case_id for f in report.scorecard.failures) == ["hangs", "malformed", "raises"]
    assert report.scorecard.entry("policy_violation").flag is True
    markdown = report_markdown(report)
    assert "FLAG RAISED" in markdown and "- hangs repeat 0: TimeoutError" in markdown
    assert "Result: FAILED" in markdown
    assert report_json(report)["scorecard"]["failures"]


async def test_a_hostile_run_can_be_resumed_without_running_anything_twice(tmp_path) -> None:
    from agent_evals import JsonlRowSink

    dataset = EvaluationDataset("hostile", "v1", CASES)
    sink = JsonlRowSink(tmp_path / "runs.jsonl")
    settings = RunSettings(timeout_s=0.05)
    await run_repeated(dataset.cases, [EvaluationVariant("hostile")], hostile, SCORERS, settings, sink)
    calls = 0

    def counting(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
        nonlocal calls
        calls += 1
        return PredictionResult(answer="ok")

    run = await run_repeated(dataset.cases, [EvaluationVariant("hostile")], counting, SCORERS, settings, sink)

    assert calls == 0
    assert len(run.rows) == 6
