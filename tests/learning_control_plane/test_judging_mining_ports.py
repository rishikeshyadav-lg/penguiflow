"""Judge fixes found while mining: confirmed no-data answers from empty lookups, and invented benchmarks."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from judging_fixtures import INVENTORY_VOCABULARY

from learning_control_plane.evaluation.verification import SafeStepEvidence, VerificationCheck
from learning_control_plane.judging import (
    AgentRun,
    AgentStep,
    BenchmarkClaimRule,
    EmptyLookupRule,
    OutcomeLadder,
    RubricJudgment,
    no_data_confirmed_by_empty_lookups,
    states_unsupported_benchmark,
)

EMPTY_LOOKUPS = EmptyLookupRule(
    is_scoped_lookup=lambda step: step.tool in {"query_stock", "list_stores"} and bool(step.args.get("filters")),
    returned_nothing=lambda step: isinstance(step.result, Mapping) and not step.result.get("rows"),
    vocabulary=INVENTORY_VOCABULARY,
)
BENCHMARKS = BenchmarkClaimRule(
    metric_names=re.compile(r"\b(?:sell[- ]through|return rate)\b", re.IGNORECASE),
    benchmark_tools=frozenset({"industry_benchmarks"}),
)
SCOPED_EMPTY = AgentStep("query_stock", {"filters": [{"store": "4411"}]}, {"rows": []})


def _run(answer: str, *steps: AgentStep) -> AgentRun:
    return AgentRun(question="How did store 4411 do?", steps=steps, final_answer=answer)


def test_a_no_data_answer_every_empty_lookup_confirms_is_accepted() -> None:
    run = _run("Store 4411 was not found; there were no matching rows.", SCOPED_EMPTY)

    assert no_data_confirmed_by_empty_lookups(run, EMPTY_LOOKUPS, reference=None, findings=())


def test_a_lookup_that_found_rows_rules_out_a_no_data_answer() -> None:
    found = AgentStep("list_stores", {"filters": [{"store": "4411"}]}, {"rows": [{"store": "4411"}]})
    run = _run("Store 4411 was not found.", SCOPED_EMPTY, found)

    assert not no_data_confirmed_by_empty_lookups(run, EMPTY_LOOKUPS, reference=None, findings=())


def test_a_reference_with_rows_rules_out_a_no_data_answer() -> None:
    run = _run("Store 4411 was not found.", SCOPED_EMPTY)

    assert not no_data_confirmed_by_empty_lookups(run, EMPTY_LOOKUPS, reference={"4411": {}}, findings=())


def test_stated_zero_totals_rule_out_a_no_data_answer() -> None:
    run = _run("Store 4411 was not found; units sold: 0.", SCOPED_EMPTY)

    assert not no_data_confirmed_by_empty_lookups(run, EMPTY_LOOKUPS, reference=None, findings=())


def test_a_meaning_finding_beyond_not_answered_rules_out_a_no_data_answer() -> None:
    run = _run("Store 4411 was not found.", SCOPED_EMPTY)

    assert no_data_confirmed_by_empty_lookups(run, EMPTY_LOOKUPS, reference=None, findings=("question_not_answered",))
    assert not no_data_confirmed_by_empty_lookups(run, EMPTY_LOOKUPS, reference=None, findings=("invented_figure",))


def test_without_a_scoped_lookup_there_is_nothing_to_confirm() -> None:
    unscoped = AgentStep("query_stock", {}, {"rows": []})

    assert not no_data_confirmed_by_empty_lookups(
        _run("Store 4411 was not found.", unscoped), EMPTY_LOOKUPS, reference=None, findings=()
    )


def test_an_answer_that_does_not_say_no_data_is_not_confirmed() -> None:
    assert not no_data_confirmed_by_empty_lookups(
        _run("Store 4411 is doing well.", SCOPED_EMPTY), EMPTY_LOOKUPS, reference=None, findings=()
    )


class _NeverJudges:
    def step_evidence(self, run: AgentRun) -> Sequence[SafeStepEvidence]:
        return [SafeStepEvidence(0, "query_stock", result_checks=(VerificationCheck("tool_execution", "passed"),))]

    def clarification_candidates(self, run: AgentRun) -> Sequence[str]:
        return ()

    def service_unavailable(self, run: AgentRun) -> bool:
        return False

    def judge(self, run: AgentRun, reference: Any) -> RubricJudgment | None:
        raise AssertionError("a confirmed no-data answer is settled before the domain judgment")


def test_the_ladder_handles_a_confirmed_no_data_answer_before_the_domain_judgment() -> None:
    ladder = OutcomeLadder(_NeverJudges(), empty_lookups=EMPTY_LOOKUPS)

    verification = ladder.judge(_run("Store 4411 was not found.", SCOPED_EMPTY), meaning_findings=())

    assert verification.outcome == "handled_correctly"
    assert verification.final_answer is not None
    assert verification.final_answer.hard_failure_codes == ("no_data_confirmed",)


def test_a_benchmark_comparison_no_tool_returned_is_unsupported() -> None:
    run = _run("A sell-through of 42% is in the typical range for spring stock.")

    assert states_unsupported_benchmark(run, BENCHMARKS)


def test_benchmark_wording_without_a_metric_is_not_a_claim() -> None:
    run = _run("That is typical for a store winding down its season.")

    assert not states_unsupported_benchmark(run, BENCHMARKS)


def test_a_benchmark_tool_that_returned_rows_supports_the_comparison() -> None:
    benchmarks = AgentStep("industry_benchmarks", {}, {"rows": [{"sell_through": 0.4}]})
    run = _run("Sell-through is above the industry average.", benchmarks)

    assert not states_unsupported_benchmark(run, BENCHMARKS)


def test_a_benchmark_tool_that_returned_nothing_does_not_support_the_comparison() -> None:
    benchmarks = AgentStep("industry_benchmarks", {}, {"rows": []})
    run = _run("Sell-through is above the industry average.", benchmarks)

    assert states_unsupported_benchmark(run, BENCHMARKS)
