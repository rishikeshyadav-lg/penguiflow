"""One evaluation definition, run on every framework's agent and on a plain function, gives comparable reports."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from _paths import CONFORMANCE_CASES

import pytest

from agent_evals import (
    ArgumentCorrectness,
    DatasetManifest,
    EvaluationCase,
    EvaluationDataset,
    EvaluationVariant,
    ExactMatch,
    GenericStep,
    GenericTrajectory,
    PolicyCheck,
    PolicyVeto,
    PredictionResult,
    RunRecord,
    ToolArguments,
    ToolSelection,
    build_report,
    report_json,
    run_repeated,
)

CASE = EvaluationCase(
    "clicks",
    {"question": "how many clicks did the campaign get"},
    expected={"answer": "400 clicks.", "tools": ["lookup"]},
)
FRAMEWORKS = ["penguiflow", "mock", "langchain", "langchain-messages"]


def _conformance_cases():
    path = CONFORMANCE_CASES
    if path is None:
        pytest.skip("this needs the monorepo; `python -m agent_evals.selfcheck` is the standalone proof")

    spec = importlib.util.spec_from_file_location("conformance_cases_proof", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CASES


def _scorers():
    def answered(case: EvaluationCase, output: PredictionResult) -> dict[str, float]:
        wanted = EvaluationCase(case.case_id, {}, case.expected["answer"])
        return ExactMatch(name="success")(wanted, output)

    return [
        PolicyVeto(answered, PolicyCheck(forbidden_tools=["delete"]), "success"),
        ToolSelection(),
        ArgumentCorrectness({"lookup": ToolArguments(required=("acid",), types={"acid": str})}),
    ]


async def _report(runner, name: str):
    dataset = EvaluationDataset("proof", "v1", [CASE])
    manifest = DatasetManifest.from_dataset(dataset, suite="regression")
    run = await run_repeated(dataset.cases, [EvaluationVariant(name)], runner, _scorers())
    record = RunRecord.from_run(run, manifest, name, run_id=f"proof-{name}", created_at="2026-09-29T00:00:00Z")
    return build_report(run, manifest, name, record=record, resamples=100)


def _framework_runner(framework: str, tmp_path: Path):
    adapter, native_run = _conformance_cases()[framework](tmp_path)

    def runner(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
        trajectory = adapter.to_generic_trajectory(native_run)
        return PredictionResult(answer=trajectory.final_answer, trajectory=trajectory)

    return runner


def _plain_function_runner():
    def agent(question: str) -> tuple[str, list[tuple[str, dict]]]:
        return "400 clicks.", [("lookup", {"acid": "112774"})]

    def runner(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
        answer, calls = agent(case.inputs["question"])
        return PredictionResult(
            answer=answer,
            trajectory=GenericTrajectory(case.inputs["question"], [GenericStep(t, a) for t, a in calls], answer),
        )

    return runner


async def _all_reports(tmp_path: Path):
    reports = {name: await _report(_framework_runner(name, tmp_path), name) for name in FRAMEWORKS}
    reports["plain-function"] = await _report(_plain_function_runner(), "plain-function")
    return reports


async def test_five_kinds_of_agent_get_reports_with_one_structure(tmp_path: Path) -> None:
    payloads = [report_json(report) for report in (await _all_reports(tmp_path)).values()]

    reference = payloads[0]
    for payload in payloads:
        assert set(payload) == set(reference)
        assert [set(e) for e in payload["scorecard"]["entries"]] == [set(e) for e in reference["scorecard"]["entries"]]
        assert [e["key"] for e in payload["scorecard"]["entries"]] == [
            e["key"] for e in reference["scorecard"]["entries"]
        ]


@pytest.mark.parametrize("key", ["success_rate", "tool_selection", "argument_correctness", "policy_violation"])
async def test_the_same_path_scores_the_same_whichever_framework_produced_it(key: str, tmp_path: Path) -> None:
    reports = await _all_reports(tmp_path)

    values = {name: report.scorecard.entry(key).value for name, report in reports.items()}

    assert len(set(values.values())) == 1, values
    assert all(value is not None for value in values.values())


async def test_every_agent_passes_the_regression_suite_on_the_same_evidence(tmp_path: Path) -> None:
    reports = await _all_reports(tmp_path)

    assert {name: report.suite.passed for name, report in reports.items()} == dict.fromkeys(reports, True)
    assert all(report.scorecard.entry("plan_adherence").measured is False for report in reports.values())
