from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from golden_fixture_judge import judge_by_answer

from learning_control_plane.judging import AgentRun, AgentStep
from learning_control_plane.judging.golden import (
    AcceptedMiss,
    GoldenLabel,
    GoldenReport,
    main,
    run_from_record,
    run_golden_set,
)

RUNS = {
    "right": AgentRun("How many units?", (AgentStep("query_stock"),), "North sold 1,200 units."),
    "wrong": AgentRun("How many units?", (AgentStep("query_stock"),), "North sold 900 units."),
    "ask": AgentRun("How many units?", (AgentStep("query_stock"),), "North or South: which one do you mean?"),
}


async def _units(run: AgentRun) -> str:
    return "1,200"


def _report(labels: list[GoldenLabel], accepted: tuple[AcceptedMiss, ...] = ()) -> GoldenReport:
    return asyncio.run(
        run_golden_set(
            labels, lambda label: RUNS.get(label.case_key), judge_by_answer, reference=_units, accepted_misses=accepted
        )
    )


def test_a_judge_that_agrees_with_every_label_passes() -> None:
    report = _report(
        [
            GoldenLabel("right", "verified"),
            GoldenLabel("wrong", "failed"),
            GoldenLabel("ask", "handled_correctly"),
        ]
    )

    assert report.passed
    assert report.agreement == 3


def test_a_disagreement_that_is_not_accepted_is_a_new_miss() -> None:
    report = _report([GoldenLabel("wrong", "verified")])

    assert not report.passed
    assert [result.case_key for result in report.new_misses] == ["wrong"]
    assert "NEW MISS wrong" in report.summary()


def test_an_accepted_miss_with_its_known_outcome_still_passes() -> None:
    report = _report([GoldenLabel("wrong", "verified")], (AcceptedMiss("wrong", "failed", "judge reads it wrong"),))

    assert report.passed
    assert report.agreement == 0
    assert "ACCEPTED wrong" in report.summary()


def test_an_accepted_miss_whose_outcome_changed_is_a_new_miss() -> None:
    report = _report([GoldenLabel("ask", "verified")], (AcceptedMiss("ask", "failed"),))

    assert not report.passed


def test_an_accepted_miss_that_now_agrees_is_reported_as_fixed() -> None:
    report = _report([GoldenLabel("right", "verified")], (AcceptedMiss("right", "failed"),))

    assert report.passed
    assert report.fixed_misses == ("right",)
    assert "FIXED right" in report.summary()


def test_a_label_whose_run_is_missing_never_agrees() -> None:
    report = _report([GoldenLabel("gone", "verified")])

    assert report.results[0].judged_outcome == "missing_run"
    assert not report.passed


def test_a_failing_reference_is_recorded_and_the_run_judged_without_it() -> None:
    async def broken(run: AgentRun) -> Any:
        raise ConnectionError("warehouse down")

    report = asyncio.run(
        run_golden_set([GoldenLabel("right", "failed")], lambda label: RUNS["right"], judge_by_answer, reference=broken)
    )

    assert report.passed
    assert report.results[0].reference_error == "ConnectionError: warehouse down"


def test_an_empty_golden_set_does_not_pass() -> None:
    assert not _report([]).passed


def test_a_label_with_an_unknown_outcome_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported expected outcome"):
        GoldenLabel("right", "great")  # type: ignore[arg-type]


def test_a_stored_run_record_is_rebuilt_with_its_steps_and_rendered_output() -> None:
    run = run_from_record(
        {
            "question": "q",
            "steps": [{"tool": "query_stock", "args": {"store": "N"}, "result": {"units": 5}}],
            "final_answer": "5 units.",
            "rendered": [{"kind": "table", "content": {"rows": []}}],
        }
    )

    assert run.steps[0].args == {"store": "N"}
    assert run.rendered[0].kind == "table"


def test_the_command_exits_non_zero_on_a_new_miss(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    labels = tmp_path / "labels.json"
    runs = tmp_path / "runs.json"
    misses = tmp_path / "misses.json"
    output = tmp_path / "results.json"
    labels.write_text(json.dumps([{"case_key": "a", "expected_outcome": "verified"}]))
    runs.write_text(json.dumps({"a": {"question": "q", "steps": [], "final_answer": "Sold 900 units."}}))
    misses.write_text(json.dumps([]))

    exit_code = main(
        [
            "--labels",
            str(labels),
            "--runs",
            str(runs),
            "--judge",
            "golden_fixture_judge:build_judge",
            "--reference",
            "golden_fixture_judge:build_reference",
            "--accepted-misses",
            str(misses),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 1
    assert "golden set: FAILED" in capsys.readouterr().out
    assert json.loads(output.read_text())[0]["judged_outcome"] == "failed"


def test_the_command_passes_when_every_label_agrees(tmp_path: Path) -> None:
    labels = tmp_path / "labels.json"
    runs = tmp_path / "runs.json"
    labels.write_text(json.dumps([{"case_key": "a", "expected_outcome": "verified"}]))
    runs.write_text(json.dumps({"a": {"question": "q", "steps": [], "final_answer": "Sold 1,200 units."}}))

    exit_code = main(
        [
            "--labels",
            str(labels),
            "--runs",
            str(runs),
            "--judge",
            "golden_fixture_judge:build_judge",
            "--reference",
            "golden_fixture_judge:build_reference",
        ]
    )

    assert exit_code == 0


def test_the_command_rejects_a_judge_that_is_not_module_and_factory(tmp_path: Path) -> None:
    labels = tmp_path / "labels.json"
    runs = tmp_path / "runs.json"
    labels.write_text("[]")
    runs.write_text("{}")

    with pytest.raises(ValueError, match="module:factory"):
        main(["--labels", str(labels), "--runs", str(runs), "--judge", "no_factory"])
