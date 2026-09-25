"""Judge fixes found while mining: confirmed no-data answers from empty lookups, and invented benchmarks."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from judging_fixtures import INVENTORY_VOCABULARY

from learning_control_plane.evaluation.verification import SafeStepEvidence, VerificationCheck
from learning_control_plane.judging import (
    NO_DATA_PHRASES,
    AgentRun,
    AgentStep,
    BenchmarkClaimRule,
    EmptyLookupRule,
    OutcomeLadder,
    RenderedOutput,
    RubricJudgment,
    no_data_confirmed_by_empty_lookups,
    stated_values,
    states_as_fact,
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


def test_a_rank_column_is_not_taken_for_a_markdown_rows_name() -> None:
    answer = "| # | Store | Units |\n|---|---|---|\n| 1 | North | 1,452 |\n| 2 | South | 306 |\n"

    values = {(value.label, value.value) for value in stated_values(answer, INVENTORY_VOCABULARY)}

    assert values == {("North", 1452.0), ("South", 306.0)}


def test_a_rank_column_is_not_taken_for_a_rendered_rows_name() -> None:
    table = RenderedOutput(
        "table",
        {
            "columns": [{"field": "rank", "header": "Rank"}, {"field": "store"}, {"field": "units"}],
            "rows": [{"rank": 1, "store": "North", "units": 5}],
        },
    )

    values = stated_values("", INVENTORY_VOCABULARY, [table])

    assert [(value.label, value.value) for value in values] == [("North", 5.0)]


def test_a_search_for_part_of_a_name_may_find_others_to_suggest() -> None:
    rule = EmptyLookupRule(
        is_scoped_lookup=EMPTY_LOOKUPS.is_scoped_lookup,
        returned_nothing=EMPTY_LOOKUPS.returned_nothing,
        vocabulary=INVENTORY_VOCABULARY,
        lookup_terms=lambda step: {str(item["store"]).casefold() for item in step.args.get("filters", ())},
    )
    whole = AgentStep("query_stock", {"filters": [{"store": "North_Outlet_2026"}]}, {"rows": []})
    part = AgentStep("list_stores", {"filters": [{"store": "North"}]}, {"rows": [{"store": "North Store"}]})
    run = _run("No store matching North_Outlet_2026 came up empty; did you mean North Store?", whole, part)

    assert no_data_confirmed_by_empty_lookups(run, rule, reference=None, findings=())
    assert not no_data_confirmed_by_empty_lookups(run, EMPTY_LOOKUPS, reference=None, findings=())


def test_new_no_data_wording_is_recognised() -> None:
    assert NO_DATA_PHRASES.search("The search came up empty.")
    assert NO_DATA_PHRASES.search("There is no store named Zeta.")


def test_a_claim_after_a_request_to_check_is_not_stated_as_fact() -> None:
    on_track = re.compile(r"\bis on track\b")

    assert not states_as_fact("Please verify the restock is on track.", on_track)
    assert states_as_fact("The restock is on track. Please verify it.", on_track)


def test_exceeding_the_norms_is_benchmark_wording() -> None:
    assert states_unsupported_benchmark(_run("Sell-through exceeds seasonal norms."), BENCHMARKS)
