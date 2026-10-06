"""Tool selection, call order, argument correctness, invariants, and the success-versus-selection gap."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from _paths import CONFORMANCE_CASES

import pytest

from agent_evals import (
    AllowedTools,
    ArgumentCorrectness,
    EvaluationCase,
    EvaluationVariant,
    Forbidden,
    GenericStep,
    GenericTrajectory,
    InvariantsHold,
    MaxCalls,
    PredictionResult,
    Required,
    RunSettings,
    SequenceMatch,
    ToolArguments,
    ToolSelection,
    Tracked,
    argument_codes,
    check_arguments,
    check_invariants,
    run_repeated,
    selection_gap,
    sequence_matches,
    tool_selection_accuracy,
    trajectory_of,
)


def _trajectory(*calls: str | tuple[str, dict]) -> GenericTrajectory:
    steps = [GenericStep(call) if isinstance(call, str) else GenericStep(call[0], call[1]) for call in calls]
    return GenericTrajectory(query="q", steps=steps, final_answer="a")


def _case(tools: list[str] | None = None, case_id: str = "c1") -> EvaluationCase:
    return EvaluationCase(case_id, {}, expected={"tools": tools} if tools is not None else None)


def _output(*calls: str | tuple[str, dict]) -> PredictionResult:
    return PredictionResult(answer="a", trajectory=_trajectory(*calls))


def test_selection_is_matched_calls_over_the_larger_of_expected_and_actual() -> None:
    assert tool_selection_accuracy(["a", "b"], ["a", "b"]) == 1.0
    assert tool_selection_accuracy(["a", "b"], ["a", "b", "c"]) == pytest.approx(2 / 3)  # a wasted call
    assert tool_selection_accuracy(["a", "b"], ["a"]) == 0.5  # a missing call
    assert tool_selection_accuracy(["a", "b"], ["x", "y"]) == 0.0  # the wrong calls
    assert tool_selection_accuracy([], []) == 1.0


def test_each_expected_call_is_matched_at_most_once() -> None:
    assert tool_selection_accuracy(["a"], ["a", "a", "a"]) == pytest.approx(1 / 3)
    assert tool_selection_accuracy(["a", "a"], ["a"]) == 0.5


def test_tool_names_are_compared_ignoring_case_and_spaces() -> None:
    assert tool_selection_accuracy(["Lookup"], [" lookup "]) == 1.0


def test_the_tool_selection_scorer_reads_the_reference_from_the_case() -> None:
    scorer = ToolSelection()

    assert scorer(_case(["lookup", "report"]), _output("lookup", "report")) == {"tool_selection_accuracy": 1.0}
    assert scorer(_case(["lookup", "report"]), _output("lookup")) == {"tool_selection_accuracy": 0.5}


def test_a_case_with_no_reference_tools_is_an_error_not_a_score() -> None:
    with pytest.raises(ValueError, match="case c1 has no expected"):
        ToolSelection()(_case(None), _output("lookup"))


def test_a_run_with_no_trajectory_cannot_be_scored_on_its_path() -> None:
    with pytest.raises(ValueError, match="has no trajectory to score"):
        ToolSelection()(_case(["a"]), PredictionResult(answer="a"))
    with pytest.raises(ValueError, match="has no trajectory to score"):
        trajectory_of("just text")


def test_a_generic_trajectory_can_be_scored_directly() -> None:
    assert ToolSelection()(_case(["a"]), _trajectory("a")) == {"tool_selection_accuracy": 1.0}


def test_a_scorer_can_be_given_another_way_to_find_the_reference() -> None:
    scorer = ToolSelection(expected_tools=lambda case: case.inputs["plan"], name="plan_selection")

    assert scorer(EvaluationCase("c1", {"plan": ["a"]}), _output("a")) == {"plan_selection": 1.0}


def test_the_three_order_modes_on_one_reference() -> None:
    reference = ["a", "b"]

    assert [sequence_matches(reference, ["a", "b"], mode) for mode in ("exact", "in_order", "any_order")] == [True] * 3
    assert [sequence_matches(reference, ["a", "x", "b"], mode) for mode in ("exact", "in_order", "any_order")] == [
        False,
        True,
        True,
    ]
    assert [sequence_matches(reference, ["b", "a"], mode) for mode in ("exact", "in_order", "any_order")] == [
        False,
        False,
        True,
    ]
    assert [sequence_matches(reference, ["a"], mode) for mode in ("exact", "in_order", "any_order")] == [False] * 3


def test_any_order_respects_how_often_the_reference_names_a_tool() -> None:
    assert sequence_matches(["a", "a"], ["a"], "any_order") is False
    assert sequence_matches(["a", "a"], ["a", "b", "a"], "any_order") is True


def test_an_unknown_order_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        sequence_matches(["a"], ["a"], "loose")  # type: ignore[arg-type]


def test_the_sequence_scorer_takes_a_mode_and_scores_zero_or_one() -> None:
    assert SequenceMatch("in_order")(_case(["a", "b"]), _output("a", "x", "b")) == {"sequence_match": 1.0}
    assert SequenceMatch("exact")(_case(["a", "b"]), _output("a", "x", "b")) == {"sequence_match": 0.0}


SPEC = ToolArguments(
    required=("campaign_id", "days"),
    types={"campaign_id": str, "days": int},
    ranges={"days": (1, 90)},
    allowed_values={"metric": ("spend", "clicks")},
    allow_extra=False,
)


def test_a_well_formed_call_has_no_codes() -> None:
    assert argument_codes(SPEC, {"campaign_id": "112774", "days": 7, "metric": "spend"}) == ([], [])


def test_syntactic_problems_are_reported_as_codes_that_name_the_argument() -> None:
    syntactic, semantic = argument_codes(SPEC, {"campaign_id": 112774, "surprise": 1})

    assert sorted(syntactic) == ["missing_argument:days", "unexpected_argument:surprise", "wrong_type:campaign_id"]
    assert semantic == []


def test_semantic_problems_are_reported_separately() -> None:
    syntactic, semantic = argument_codes(SPEC, {"campaign_id": "1", "days": 400, "metric": "revenue"})

    assert syntactic == []
    assert sorted(semantic) == ["out_of_range:days", "value_not_allowed:metric"]


def test_a_bool_is_not_accepted_where_a_number_is_asked_for() -> None:
    syntactic, _ = argument_codes(SPEC, {"campaign_id": "1", "days": True})

    assert syntactic == ["wrong_type:days"]


def test_a_range_may_be_open_at_one_end() -> None:
    open_ended = ToolArguments(ranges={"x": (0, None)})

    assert argument_codes(open_ended, {"x": 10_000}) == ([], [])
    assert argument_codes(open_ended, {"x": -1}) == ([], ["out_of_range:x"])


def test_codes_never_contain_the_values_the_agent_passed() -> None:
    secret = "acct-SECRET-9999"
    trajectory = _trajectory(("send", {"token": secret, "days": 999}))

    checks = check_arguments({"send": ToolArguments(types={"token": int}, ranges={"days": (1, 7)})}, trajectory)

    assert checks[0].syntactic_codes == ["wrong_type:token"]
    assert secret not in repr(checks)
    assert "999" not in repr(checks)


def test_only_calls_to_tools_with_a_spec_are_checked() -> None:
    trajectory = _trajectory("other", ("lookup", {"campaign_id": "1", "days": 7}), ("lookup", {"days": 7}))

    checks = check_arguments({"lookup": SPEC}, trajectory)

    assert [(check.step_index, check.correct) for check in checks] == [(1, True), (2, False)]


def test_the_argument_scorer_gives_an_overall_score_and_one_for_each_level() -> None:
    scorer = ArgumentCorrectness({"lookup": SPEC})
    output = _output(
        ("lookup", {"campaign_id": "1", "days": 7}),
        ("lookup", {"campaign_id": "1", "days": 400}),  # syntactically fine, semantically wrong
        ("lookup", {"days": 7}),  # syntactically wrong
        ("lookup", {"campaign_id": "1", "days": 7}),
    )

    assert scorer(_case(), output) == {
        "argument_correctness": 0.5,
        "argument_correctness_syntax": 0.75,
        "argument_correctness_semantics": 0.75,
    }


def test_a_run_with_no_checked_calls_has_nothing_to_get_wrong() -> None:
    scorer = ArgumentCorrectness({"lookup": SPEC})

    assert scorer(_case(), _output("other")) == {
        "argument_correctness": 1.0,
        "argument_correctness_syntax": 1.0,
        "argument_correctness_semantics": 1.0,
    }


def test_an_authorization_must_come_before_a_write_but_only_when_there_is_a_write() -> None:
    rule = Required("authorize", before="write")

    assert check_invariants([rule], _trajectory("authorize", "write")).passed
    assert check_invariants([rule], _trajectory("read", "read")).passed  # no write, nothing to authorize
    assert check_invariants([rule], _trajectory("write", "authorize")).codes == ["required_tool_not_called_before"]
    assert check_invariants([rule], _trajectory("write")).codes == ["required_tool_not_called_before"]


def test_diagnostic_steps_in_any_order_are_fine_because_no_order_is_asserted() -> None:
    rules = [Required("check_a"), Required("check_b"), Required("check_c")]

    assert check_invariants(rules, _trajectory("check_c", "check_a", "check_b")).passed
    assert check_invariants(rules, _trajectory("check_b", "check_c", "check_a")).passed


def test_a_forbidden_tool_and_a_tool_outside_the_allowed_list_are_violations() -> None:
    assert check_invariants([Forbidden("delete")], _trajectory("read", "delete")).codes == ["forbidden_tool_called"]
    report = check_invariants([AllowedTools(["read"])], _trajectory("read", "write"))
    assert report.codes == ["disallowed_tool_called"]
    assert [violation.tool for violation in report.violations] == ["write"]


def test_a_call_cap_is_a_violation_only_above_the_limit() -> None:
    cap = MaxCalls("lookup", 2)

    assert check_invariants([cap], _trajectory("lookup", "lookup")).passed
    assert check_invariants([cap], _trajectory("lookup", "lookup", "lookup")).codes == [
        "maximum_tool_call_count_exceeded"
    ]


def test_a_negative_call_cap_is_refused() -> None:
    with pytest.raises(ValueError, match="limit must not be negative"):
        MaxCalls("lookup", -1)


def test_a_tracked_tool_is_counted_and_never_fails_a_run() -> None:
    report = check_invariants([Tracked("cache_hit")], _trajectory("cache_hit", "cache_hit", "read"))

    assert report.passed
    assert report.tracked_counts == {"cache_hit": 2}


def test_the_invariant_scorer_reports_each_invariant_and_the_kinds_of_violation() -> None:
    scorer = InvariantsHold([Required("lookup"), Forbidden("delete"), MaxCalls("lookup", 1)])

    score = scorer(_case(), _output("lookup", "lookup", "delete"))

    assert score.score == 0.0
    assert score.checks == {"required:lookup": True, "forbidden:delete": False, "max_calls:lookup": False}
    assert score.feedback == "forbidden_tool_called, maximum_tool_call_count_exceeded"
    assert scorer(_case(), _output("lookup")).score == 1.0


async def test_the_invariant_scorer_works_inside_a_run() -> None:
    scorer = InvariantsHold([Forbidden("delete")])

    def runner(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
        return _output("read", "delete") if case.case_id == "bad" else _output("read")

    cases = [EvaluationCase("ok", {}), EvaluationCase("bad", {})]
    run = await run_repeated(cases, [EvaluationVariant("v")], runner, scorer)

    assert [row.result.metrics for row in run.rows] == [{"invariants_hold": 1.0}, {"invariants_hold": 0.0}]
    assert run.rows[1].result.score_details["invariants_hold"]["feedback"] == "forbidden_tool_called"


# Recorded from the campaign's `_execution_policy_check` on the same tool calls and the same policy.
POLICY = [
    AllowedTools(["lookup", "aggregate_report", "list_tables"]),
    Required("aggregate_report"),
    MaxCalls("aggregate_report", 2),
    MaxCalls("lookup", 3),
]
POLICY_GOLDEN = {
    "clean": ["lookup", "aggregate_report"],
    "missing_required": ["lookup", "list_tables"],
    "disallowed": ["lookup", "aggregate_report", "send_email"],
    "over_max": ["aggregate_report", "aggregate_report", "aggregate_report"],
    "over_max_lookup": ["lookup", "lookup", "lookup", "lookup", "aggregate_report"],
    "case_and_space": [" Lookup ", "AGGREGATE_REPORT"],
    "two_violations": ["send_email", "send_email"],
    "no_calls": [],
}
POLICY_EXPECTED_CODES = {
    "clean": [],
    "missing_required": ["required_tool_not_called"],
    "disallowed": ["disallowed_tool_called"],
    "over_max": ["maximum_tool_call_count_exceeded"],
    "over_max_lookup": ["maximum_tool_call_count_exceeded"],
    "case_and_space": [],
    "two_violations": ["disallowed_tool_called", "required_tool_not_called"],
    "no_calls": ["required_tool_not_called"],
}


@pytest.mark.parametrize("scenario", list(POLICY_GOLDEN))
def test_the_invariants_find_the_violations_the_campaign_policy_check_found(scenario: str) -> None:
    report = check_invariants(POLICY, _trajectory(*POLICY_GOLDEN[scenario]))

    assert report.codes == POLICY_EXPECTED_CODES[scenario]
    assert report.passed == (POLICY_EXPECTED_CODES[scenario] == [])


async def _run_with(outcomes: dict[str, tuple[float, float]]):
    """Per case: (outcome score, selection score)."""

    cases = [EvaluationCase(case_id, {}) for case in [outcomes] for case_id in case]

    def runner(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
        return PredictionResult(answer="x")

    def outcome(case: EvaluationCase, output: PredictionResult) -> dict[str, float]:
        return {"success": outcomes[case.case_id][0]}

    def selection(case: EvaluationCase, output: PredictionResult) -> dict[str, float]:
        return {"selection": outcomes[case.case_id][1]}

    return await run_repeated(cases, [EvaluationVariant("v")], runner, [outcome, selection], RunSettings())


async def test_a_90_percent_success_next_to_60_percent_selection_is_flagged_as_something_to_audit() -> None:
    run = await _run_with({f"c{n}": (1.0 if n < 9 else 0.0, 1.0 if n < 6 else 0.0) for n in range(10)})

    report = selection_gap(run, "v", outcome_metric="success", selection_metric="selection")

    assert report.outcome_mean == pytest.approx(0.9)
    assert report.selection_mean == pytest.approx(0.6)
    assert report.gap == pytest.approx(0.3)
    assert report.flagged is True
    assert "audit signal, not a verdict" in report.note


async def test_matching_success_and_selection_are_not_flagged() -> None:
    run = await _run_with({f"c{n}": (1.0, 1.0) for n in range(4)})

    report = selection_gap(run, "v", outcome_metric="success", selection_metric="selection")

    assert report.gap == 0.0
    assert report.flagged is False
    assert "no gap worth auditing" in report.note


async def test_the_flag_threshold_can_be_moved() -> None:
    run = await _run_with({f"c{n}": (1.0, 0.85) for n in range(4)})

    assert selection_gap(run, "v", outcome_metric="success", selection_metric="selection").flagged is False
    assert selection_gap(run, "v", outcome_metric="success", selection_metric="selection", flag_above=0.1).flagged


def _conformance_cases():
    """The four native runs of the LCP conformance suite (PenguiFlow, LangChain twice, a mock)."""

    path = CONFORMANCE_CASES
    if path is None:
        pytest.skip("this needs the monorepo; `python -m agent_evals.selfcheck` is the standalone proof")
    spec = importlib.util.spec_from_file_location("conformance_cases", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CASES


@pytest.mark.parametrize("framework", ["penguiflow", "mock", "langchain", "langchain-messages"])
def test_the_same_scorers_give_the_same_answers_for_every_frameworks_run(framework: str, tmp_path: Path) -> None:
    adapter, native_run = _conformance_cases()[framework](tmp_path)
    output = PredictionResult(answer="400 clicks.", trajectory=adapter.to_generic_trajectory(native_run))

    assert ToolSelection()(_case(["lookup"]), output) == {"tool_selection_accuracy": 1.0}
    assert SequenceMatch("exact")(_case(["lookup"]), output) == {"sequence_match": 1.0}
    assert ArgumentCorrectness({"lookup": ToolArguments(required=("acid",), types={"acid": str})})(_case(), output) == {
        "argument_correctness": 1.0,
        "argument_correctness_syntax": 1.0,
        "argument_correctness_semantics": 1.0,
    }
    invariants = InvariantsHold([Required("lookup"), Forbidden("delete"), AllowedTools(["lookup"])])
    assert invariants(_case(), output).score == 1.0
    assert ToolSelection()(_case(["lookup", "report"]), output) == {"tool_selection_accuracy": 0.5}
