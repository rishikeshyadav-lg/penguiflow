"""Golden trajectories, trajectory diffs, and shadow comparison against recorded production runs."""

from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent_evals import (
    EvaluationCase,
    EvaluationVariant,
    GenericStep,
    GenericTrajectory,
    GoldenApproval,
    GoldenRefreshError,
    GoldenTrajectory,
    PolicyCheck,
    PredictionResult,
    RunSettings,
    diff_runs,
    freeze_golden,
    load_golden,
    refresh_golden,
    save_golden,
    shadow_compare,
)


def _trajectory(*calls: str | tuple[str, dict], answer: str = "400 clicks.") -> GenericTrajectory:
    steps = [GenericStep(c) if isinstance(c, str) else GenericStep(c[0], c[1]) for c in calls]
    return GenericTrajectory(query="how many clicks", steps=steps, final_answer=answer)


def _run(*calls: str | tuple[str, dict], answer: str = "400 clicks.", **fields: object) -> PredictionResult:
    return PredictionResult(answer=answer, trajectory=_trajectory(*calls, answer=answer), **fields)  # type: ignore[arg-type]


def _golden(**changes: object) -> GoldenTrajectory:
    steps = [
        GenericStep("lookup", {"acid": "112774"}, observation={"clicks": 400}),
        GenericStep("report", {"format": "short"}, observation="ok"),
    ]
    values: dict[str, object] = {
        "case_id": "c1",
        "query": "how many clicks",
        "steps": steps,
        "final_answer": "400 clicks.",
        "environment_ref": "fixtures@v3",
    }
    values.update(changes)
    return GoldenTrajectory(**values)  # type: ignore[arg-type]


def test_a_golden_trajectory_has_a_stable_digest() -> None:
    assert _golden().digest == _golden().digest
    assert _golden().digest.startswith("sha256:")


@pytest.mark.parametrize(
    "change",
    [
        {"final_answer": "401 clicks."},
        {"environment_ref": "fixtures@v4"},
        {"query": "how many impressions"},
        {"steps": [GenericStep("lookup", {"acid": "112774"}, observation={"clicks": 400})]},
        {"steps": [GenericStep("report", {"format": "short"}, observation="ok"),
                   GenericStep("lookup", {"acid": "112774"}, observation={"clicks": 400})]},
        {"steps": [GenericStep("lookup", {"acid": "999"}, observation={"clicks": 400}),
                   GenericStep("report", {"format": "short"}, observation="ok")]},
        {"steps": [GenericStep("lookup", {"acid": "112774"}, observation={"clicks": 401}),
                   GenericStep("report", {"format": "short"}, observation="ok")]},
        {"steps": [GenericStep("lookup", {"acid": "112774"}, observation={"clicks": 400}, error="boom"),
                   GenericStep("report", {"format": "short"}, observation="ok")]},
    ],
)  # fmt: skip
def test_any_change_to_the_frozen_run_changes_the_digest(change: dict) -> None:
    assert _golden(**change).digest != _golden().digest


def test_the_approval_history_is_not_part_of_what_the_digest_covers() -> None:
    approval = GoldenApproval("a", "why", "sha256:x", "sha256:y", "2026-09-29")

    assert _golden(approvals=[approval]).digest == _golden().digest


def test_a_golden_trajectory_is_frozen_from_a_run() -> None:
    golden = freeze_golden("c1", _trajectory("lookup", "report"), environment_ref="fixtures@v3")

    assert [step.tool for step in golden.steps] == ["lookup", "report"]
    assert golden.final_answer == "400 clicks."


def test_a_golden_trajectory_survives_a_round_trip_through_a_file(tmp_path: Path) -> None:
    golden = _golden(steps=[GenericStep("lookup", {"when": datetime(2026, 9, 29, tzinfo=UTC)}, observation={"n": 1})])

    save_golden(golden, tmp_path / "g.json")
    loaded = load_golden(tmp_path / "g.json")

    assert loaded.digest == golden.digest
    assert loaded.steps[0].observation == {"n": 1}


def test_a_golden_file_edited_after_it_was_frozen_is_refused(tmp_path: Path) -> None:
    save_golden(_golden(), tmp_path / "g.json")
    payload = json.loads((tmp_path / "g.json").read_text())
    payload["final_answer"] = "a different answer"
    (tmp_path / "g.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="was edited after it was frozen"):
        load_golden(tmp_path / "g.json")


def _approval(current: GoldenTrajectory, new: GoldenTrajectory, **changes: str) -> GoldenApproval:
    values = {
        "approver": "reviewer",
        "reason": "the report tool now returns a table",
        "previous_digest": current.digest,
        "new_digest": new.digest,
        "approved_at": "2026-09-29T12:00:00Z",
    }
    values.update(changes)
    return GoldenApproval(**values)


def test_a_refresh_with_an_approval_for_exactly_this_change_is_accepted_and_keeps_the_history() -> None:
    current, new = _golden(), _golden(final_answer="401 clicks.")

    refreshed = refresh_golden(current, new, _approval(current, new))

    assert refreshed.digest == new.digest
    assert [a.reason for a in refreshed.approvals] == ["the report tool now returns a table"]


def test_refreshes_chain_and_the_whole_history_is_kept() -> None:
    first, second, third = _golden(), _golden(final_answer="401"), _golden(final_answer="402")

    once = refresh_golden(first, second, _approval(first, second, reason="first change"))
    twice = refresh_golden(once, third, _approval(once, third, reason="second change"))

    assert [a.reason for a in twice.approvals] == ["first change", "second change"]


def test_a_refresh_without_an_approval_record_is_refused() -> None:
    with pytest.raises(GoldenRefreshError, match="without an approval record"):
        refresh_golden(_golden(), _golden(final_answer="401"), None)


def test_an_approval_for_another_replacement_cannot_be_reused() -> None:
    current, new, other = _golden(), _golden(final_answer="401"), _golden(final_answer="402")

    with pytest.raises(GoldenRefreshError, match="not for the new run"):
        refresh_golden(current, other, _approval(current, new))
    with pytest.raises(GoldenRefreshError, match="not for the golden trajectory being replaced"):
        refresh_golden(other, new, _approval(current, new))


def test_a_refresh_cannot_swap_in_a_run_of_another_case() -> None:
    current, new = _golden(), _golden(case_id="c2")

    with pytest.raises(GoldenRefreshError, match="is for case c2, not c1"):
        refresh_golden(current, new, _approval(current, new))


def test_an_approval_needs_an_approver_and_a_reason() -> None:
    with pytest.raises(ValueError, match="approver must be non-empty"):
        GoldenApproval("", "why", "a", "b", "t")
    with pytest.raises(ValueError, match="reason must be non-empty"):
        GoldenApproval("who", " ", "a", "b", "t")


def test_two_runs_with_the_same_answer_and_different_tools_are_not_identical() -> None:
    reference = _run("lookup", "report")
    candidate = _run("lookup", "search_web", "report")

    diff = diff_runs(reference, candidate)

    assert diff.same_answer is True
    assert diff.identical is False
    assert diff.process_identical is False
    assert diff.tools_added == ("search_web",)
    assert diff.tools_removed == ()
    assert diff.first_divergence == 1
    assert diff.differences == ["the tool sequence differs from step 1 (removed [], added ['search_web'])"]


def test_the_same_answer_by_the_same_process_is_identical() -> None:
    diff = diff_runs(_run(("lookup", {"a": 1})), _run(("lookup", {"a": 1})))

    assert diff.identical is True
    assert diff.differences == []
    assert diff.first_divergence is None


def test_a_replaced_tool_shows_as_one_removed_and_one_added() -> None:
    diff = diff_runs(_run("lookup", "report"), _run("lookup", "summary"))

    assert (diff.tools_removed, diff.tools_added, diff.first_divergence) == (("report",), ("summary",), 1)


def test_changed_arguments_are_named_and_their_values_never_appear() -> None:
    reference = _run(("lookup", {"acid": "112774", "token": "SECRET-OLD"}))
    candidate = _run(("lookup", {"acid": "999", "token": "SECRET-NEW", "extra": 1}))

    diff = diff_runs(reference, candidate)

    assert [(c.step_index, c.tool, list(c.changed_arguments)) for c in diff.argument_changes] == [
        (0, "lookup", ["acid", "extra", "token"])
    ]
    assert "SECRET" not in repr(diff) and "112774" not in repr(diff) and "SECRET" not in " ".join(diff.differences)


def test_cost_and_latency_are_reported_as_deltas_and_do_not_decide_process_identity() -> None:
    reference = _run("lookup", cost_usd=0.10, latency_ms=1_000.0)
    candidate = _run("lookup", cost_usd=0.25, latency_ms=900.0)

    diff = diff_runs(reference, candidate)

    assert diff.cost_delta == pytest.approx(0.15)
    assert diff.latency_delta == pytest.approx(-100.0)
    assert diff.identical is True


def test_a_delta_is_unknown_when_either_run_did_not_report_the_figure() -> None:
    assert diff_runs(_run("lookup", cost_usd=0.1), _run("lookup")).cost_delta is None


def test_guardrails_that_fired_or_stopped_firing_make_the_process_differ() -> None:
    reference = _run("lookup", extra={"guardrails_triggered": ["pii_filter"]})
    candidate = _run("lookup", extra={"guardrails_triggered": ["budget_cap"]})

    diff = diff_runs(reference, candidate)

    assert (diff.guardrails_added, diff.guardrails_removed) == (("budget_cap",), ("pii_filter",))
    assert diff.process_identical is False


def test_a_policy_finding_only_the_candidate_has_is_reported() -> None:
    policy = PolicyCheck(forbidden_tools=["delete"])

    diff = diff_runs(_run("lookup"), _run("lookup", "delete"), policy=policy)

    assert diff.policy_codes_added == ("forbidden_tool_called",)
    assert diff.policy_codes_removed == ()


def test_a_run_can_be_compared_with_a_golden_trajectory() -> None:
    golden = _golden()

    same = diff_runs(golden, _run(("lookup", {"acid": "112774"}), ("report", {"format": "short"})))
    drifted = diff_runs(golden, _run(("lookup", {"acid": "112774"}), ("report", {"format": "long"})))

    assert same.identical is True
    assert [(c.tool, list(c.changed_arguments)) for c in drifted.argument_changes] == [("report", ["format"])]


def test_a_different_answer_is_reported_even_when_the_path_is_the_same() -> None:
    diff = diff_runs(_run("lookup", answer="400 clicks."), _run("lookup", answer="300 clicks."))

    assert diff.same_answer is False
    assert diff.process_identical is True
    assert diff.differences == ["the answer differs"]


def _conformance_trajectories(tmp_path: Path) -> dict[str, GenericTrajectory]:
    path = Path(__file__).parents[1] / "learning_control_plane" / "test_provider_conformance.py"
    spec = importlib.util.spec_from_file_location("conformance_cases_golden", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = {}
    for name, build in module.CASES.items():
        adapter, native_run = build(tmp_path)
        result[name] = adapter.to_generic_trajectory(native_run)
    return result


def test_recorded_runs_of_different_frameworks_diff_as_a_person_would_expect(tmp_path: Path) -> None:
    runs = _conformance_trajectories(tmp_path)
    penguiflow, langchain = runs["penguiflow"], runs["langchain"]

    same = diff_runs(penguiflow, langchain)
    with_extra_call = diff_runs(
        penguiflow,
        GenericTrajectory(
            query=langchain.query,
            steps=(*langchain.steps, GenericStep("verify", {"acid": "112774"})),
            final_answer=langchain.final_answer,
        ),
    )

    assert same.identical is True
    assert with_extra_call.same_answer is True
    assert with_extra_call.tools_added == ("verify",)
    assert with_extra_call.first_divergence == 1
    assert with_extra_call.differences == ["the tool sequence differs from step 1 (removed [], added ['verify'])"]


CASES = [EvaluationCase(f"c{n}", {"query": f"q{n}"}) for n in range(1, 5)]
PRODUCTION = {
    case.case_id: _run(("lookup", {"q": case.case_id}), "report", answer=f"answer {case.case_id}") for case in CASES
}


async def test_a_shadow_report_shows_what_an_answer_only_comparison_would_miss() -> None:
    def candidate(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
        # Same answer everywhere; c1 and c2 reach it by another route.
        calls = ("lookup", "search_web", "report") if case.case_id in ("c1", "c2") else ("lookup", "report")
        return _run(*[(c, {"q": case.case_id}) if c == "lookup" else c for c in calls], answer=f"answer {case.case_id}")

    report = await shadow_compare(PRODUCTION, CASES, candidate)

    assert report.answer_agreement_rate == 1.0
    assert report.full_agreement_rate == 0.5
    assert report.process_only_differences == ("c1", "c2")
    assert (report.compared, report.failed) == (4, 0)


async def test_the_candidate_is_told_it_is_a_dry_run() -> None:
    seen: list[object] = []

    def candidate(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
        seen.append(variant.config.get("dry_run"))
        return PRODUCTION[case.case_id]

    await shadow_compare(PRODUCTION, CASES, candidate)

    assert seen == [True] * 4


async def test_a_candidate_that_matches_production_exactly_agrees_fully() -> None:
    report = await shadow_compare(PRODUCTION, CASES, lambda case, variant: PRODUCTION[case.case_id])

    assert (report.answer_agreement_rate, report.full_agreement_rate) == (1.0, 1.0)
    assert report.process_only_differences == ()


async def test_a_candidate_run_that_fails_is_listed_with_its_error_and_not_dropped() -> None:
    def candidate(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
        if case.case_id == "c3":
            raise RuntimeError("tool down")
        return PRODUCTION[case.case_id]

    report = await shadow_compare(PRODUCTION, CASES, candidate)

    assert (report.compared, report.failed) == (3, 1)
    failed = next(case for case in report.cases if case.error)
    assert (failed.case_id, failed.error) == ("c3", "RuntimeError: tool down")


async def test_a_case_with_no_recorded_production_run_is_an_error_not_a_skip() -> None:
    partial = {key: value for key, value in PRODUCTION.items() if key != "c2"}

    with pytest.raises(ValueError, match=r"no recorded production run for cases \['c2'\]"):
        await shadow_compare(partial, CASES, lambda case, variant: PRODUCTION[case.case_id])


async def test_a_shadow_comparison_runs_each_case_once() -> None:
    with pytest.raises(ValueError, match="runs each case once"):
        await shadow_compare(PRODUCTION, CASES, lambda c, v: PRODUCTION[c.case_id], settings=RunSettings(repeats=2))


async def test_a_shadow_comparison_can_check_policy_on_both_sides() -> None:
    def candidate(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
        return _run(("lookup", {"q": case.case_id}), "delete", answer=f"answer {case.case_id}")

    report = await shadow_compare(PRODUCTION, CASES[:1], candidate, policy=PolicyCheck(forbidden_tools=["delete"]))

    diff = report.cases[0].diff
    assert diff is not None
    assert diff.policy_codes_added == ("forbidden_tool_called",)


async def test_a_shadow_report_with_nothing_compared_has_no_agreement_rate() -> None:
    def candidate(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
        raise RuntimeError("down")

    report = await shadow_compare(PRODUCTION, CASES[:1], candidate)

    assert (report.answer_agreement_rate, report.full_agreement_rate) == (0.0, 0.0)
