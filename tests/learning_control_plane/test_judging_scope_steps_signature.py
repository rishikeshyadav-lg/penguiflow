from __future__ import annotations

from judging_fixtures import SEASON_SUFFIX

from learning_control_plane.judging import (
    AgentStep,
    SignatureRules,
    entity_named,
    name_forms,
    question_scope_check,
    recovered_step_indexes,
    step_failed,
)

STORE = "Acme_North Store_Q3-2026"


def test_a_name_matches_on_letters_and_digits_only() -> None:
    assert entity_named("Totals for acme north-store q3 2026:", STORE)


def test_a_name_matches_without_its_droppable_suffix() -> None:
    assert name_forms(STORE, suffix_patterns=(SEASON_SUFFIX,)) == ("acmenorthstoreq32026", "acmenorthstore")
    assert entity_named("Acme North Store sold 5 units.", STORE, suffix_patterns=(SEASON_SUFFIX,))
    assert not entity_named("Acme North Store sold 5 units.", STORE)


def test_naming_any_one_question_term_passes_scope() -> None:
    check = question_scope_check("Store 4411 sold 5 units.", question_terms=["4411", "4412"])

    assert check.status == "passed"


def test_naming_no_question_term_fails_scope() -> None:
    check = question_scope_check("Store 9999 sold 5 units.", question_terms=["4411"])

    assert check.status == "failed"
    assert check.reason_codes == ("requested_scope_missing",)


def test_the_agents_filters_stand_in_only_when_the_question_names_nothing_and_all_must_be_named() -> None:
    both = question_scope_check("North and South", question_terms=[], fallback_terms=["North", "South"])
    one = question_scope_check("North only", question_terms=[], fallback_terms=["North", "South"])

    assert both.status == "passed"
    assert one.status == "failed"


def test_scope_does_not_apply_when_nothing_names_it() -> None:
    check = question_scope_check("Everything is fine.", question_terms=[])

    assert check.status == "not_applicable"


def _is_data_step(step: AgentStep) -> bool:
    return step.tool == "query_stock"


def test_a_failed_call_followed_by_a_completed_data_call_is_recovered() -> None:
    steps = [
        AgentStep("query_stock", {"field": "colour"}, error="unknown field"),
        AgentStep("query_stock", {"field": "color"}, result={"rows": [1]}),
    ]

    assert recovered_step_indexes(steps, is_data_step=_is_data_step) == frozenset({0})


def test_a_failure_the_agent_never_moved_past_is_not_recovered() -> None:
    steps = [
        AgentStep("query_stock", result={"rows": [1]}),
        AgentStep("query_stock", result={"error": "timeout"}),
    ]

    assert step_failed(steps[1])
    assert recovered_step_indexes(steps, is_data_step=_is_data_step) == frozenset()


def test_an_empty_result_followed_by_a_data_call_is_recovered_when_the_integration_says_so() -> None:
    steps = [AgentStep("query_stock", result={"rows": []}), AgentStep("query_stock", result={"rows": [1]})]

    recovered = recovered_step_indexes(
        steps, is_data_step=_is_data_step, came_back_empty=lambda step: step.result == {"rows": []}
    )

    assert recovered == frozenset({0})


def test_a_later_non_data_call_does_not_recover_a_failure() -> None:
    steps = [AgentStep("query_stock", error="boom"), AgentStep("render_table", result={})]

    assert recovered_step_indexes(steps, is_data_step=_is_data_step) == frozenset()


RULES = SignatureRules(
    left_out=frozenset({"describe_schema"}),
    left_out_prefixes=("render_",),
    renamed={"list_store_names": "lookup"},
)


def test_the_signature_keeps_only_the_work_in_order() -> None:
    steps = [
        {"node": "describe_schema", "status": "completed"},
        {"node": "list_store_names", "status": "completed"},
        {"node": "list_store_names", "status": "completed"},
        {"node": "query_stock", "status": "failed"},
        {"node": "query_stock", "status": "completed"},
        {"node": "render_table", "status": "completed"},
    ]

    assert RULES(steps) == ("lookup", "query_stock")


def test_a_recovered_step_is_left_out_of_the_signature() -> None:
    recovered = {
        "node": "query_stock",
        "status": "completed",
        "result_checks": [{"check_id": "tool_execution", "reason_codes": ["tool_error_recovered"]}],
    }

    assert RULES([recovered, {"node": "query_stock"}]) == ("query_stock",)


def test_repeats_are_kept_when_collapsing_is_off() -> None:
    rules = SignatureRules(collapse_repeats=False)

    assert rules([{"node": "query_stock"}, {"node": "query_stock"}]) == ("query_stock", "query_stock")
