from __future__ import annotations

from judging_fixtures import INVENTORY_VOCABULARY, SEASON_SUFFIX

from learning_control_plane.judging import (
    OVERALL,
    Expectation,
    RelativeTolerance,
    StatedValue,
    anchored_values,
    judge_expectation,
    judgment_passes,
    row_is_named,
    row_name_forms,
)

TOLERANCE = RelativeTolerance(default=0.005)


def _value(metric: str, value: float, label: str = "", *, row: bool = False, explicit: bool = True) -> StatedValue:
    return StatedValue(metric, value, False, label, "text_line", row=row, explicit=explicit)


def test_a_total_is_compared_with_total_statements_not_group_rows() -> None:
    expectation = Expectation(rows={OVERALL: {"units": 1000.0}}, source="agent")

    judgment = judge_expectation(
        expectation, [_value("units", 400.0, "Week 1", row=True)], shown_text="", tolerance=TOLERANCE
    )

    assert judgment.numerical.status == "not_applicable"
    assert judgment.completeness.reason_codes == ("required_values_missing",)


def test_a_wrong_total_fails_as_an_incorrect_primary_result() -> None:
    expectation = Expectation(rows={OVERALL: {"units": 1000.0}}, source="reference")

    judgment = judge_expectation(expectation, [_value("units", 999_000.0)], shown_text="", tolerance=TOLERANCE)

    assert judgment.numerical.status == "failed"
    assert judgment.hard_failures == ("primary_result_incorrect",)
    assert not judgment_passes(judgment)


def test_a_right_total_matched_to_the_reference_is_grounded_in_it() -> None:
    expectation = Expectation(rows={OVERALL: {"units": 1000.0}}, source="reference")

    judgment = judge_expectation(expectation, [_value("units", 1000.0)], shown_text="", tolerance=TOLERANCE)

    assert judgment.numerical.reason_codes == ("stated_values_match_reference",)
    assert judgment.grounding.reason_codes == ("values_match_independent_reference",)
    assert judgment_passes(judgment)


def test_a_rounded_value_within_tolerance_matches() -> None:
    expectation = Expectation(rows={OVERALL: {"units": 1_234_567.0}}, source="agent")
    shorthand = StatedValue("units", 1_230_000.0, True, "", "text_line")

    judgment = judge_expectation(expectation, [shorthand], shown_text="", tolerance=TOLERANCE)

    assert judgment.numerical.reason_codes == ("stated_values_match_agent",)
    assert judgment.grounding.reason_codes == ("values_match_agent_tool_results",)


def test_an_inferred_value_can_confirm_but_never_contradict() -> None:
    expectation = Expectation(rows={OVERALL: {"units": 1000.0}}, source="agent")

    judgment = judge_expectation(
        expectation, [_value("units", 3.0, explicit=False)], shown_text="", tolerance=TOLERANCE
    )

    assert judgment.numerical.status == "not_applicable"


def test_an_unlabelled_value_can_confirm_one_of_several_rows() -> None:
    expectation = Expectation(rows={"North": {"units": 5.0}, "South": {"units": 7.0}}, source="reference")
    values = [_value("units", 5.0, "North"), _value("units", 7.0)]

    judgment = judge_expectation(expectation, values, shown_text="North and South", tolerance=TOLERANCE)

    assert judgment_passes(judgment)
    assert judgment.scope.reason_codes == ("requested_scope_present",)


def test_a_store_left_out_of_a_several_store_answer_is_incomplete() -> None:
    expectation = Expectation(rows={"North": {"units": 5.0}, "South": {"units": 7.0}}, source="reference")

    judgment = judge_expectation(
        expectation, [_value("units", 5.0, "North")], shown_text="North: 5 units", tolerance=TOLERANCE
    )

    assert "campaign_missing_from_answer" in judgment.completeness.reason_codes


def test_a_group_left_out_of_a_breakdown_is_a_missing_group() -> None:
    expectation = Expectation(rows={"Shoes": {"units": 5.0}, "Hats": {"units": 7.0}}, source="reference")

    judgment = judge_expectation(
        expectation, [_value("units", 5.0, "Shoes")], shown_text="Shoes: 5", tolerance=TOLERANCE, grouped_rows=True
    )

    assert "group_missing_from_answer" in judgment.completeness.reason_codes
    assert judgment.scope.reason_codes == ("scope_checked_separately",)


def test_an_answer_naming_none_of_the_requested_stores_has_the_wrong_scope() -> None:
    expectation = Expectation(rows={"North": {"units": 5.0}, "South": {"units": 7.0}}, source="reference")

    judgment = judge_expectation(expectation, [], shown_text="West sold 3 units", tolerance=TOLERANCE)

    assert judgment.scope.status == "failed"
    assert "wrong_scope" in judgment.hard_failures


def test_several_stores_judged_only_from_the_agents_own_calls_cannot_verify() -> None:
    expectation = Expectation(rows={"North": {"units": 5.0}, "South": {"units": 7.0}}, source="agent")
    values = [_value("units", 5.0, "North"), _value("units", 7.0, "South")]

    judgment = judge_expectation(
        expectation, values, shown_text="North and South", tolerance=TOLERANCE, needs_own_lookup=True
    )

    assert judgment.hard_failures == ("scope_unconfirmed",)


def test_a_metric_the_agent_never_fetched_is_the_agents_gap_but_a_reference_gap_is_not() -> None:
    values = [_value("units", 5.0)]
    required = ["units", "returns"]

    from_agent = judge_expectation(
        Expectation(rows={OVERALL: {"units": 5.0}}, source="agent"),
        values,
        shown_text="",
        tolerance=TOLERANCE,
        required_metrics=required,
    )
    from_reference = judge_expectation(
        Expectation(rows={OVERALL: {"units": 5.0}}, source="reference"),
        values,
        shown_text="",
        tolerance=TOLERANCE,
        required_metrics=required,
    )

    assert "required_data_not_fetched" in from_agent.completeness.reason_codes
    assert from_reference.completeness.status == "passed"


def test_a_roll_up_matches_any_part_and_needs_two_matches() -> None:
    expectation = Expectation(rows={"a": {"units": 5.0}, "b": {"units": 7.0}}, source="agent", label_free=True)
    values = [_value("units", 5.0, "Online"), _value("units", 7.0, "In store")]

    judgment = judge_expectation(expectation, values, shown_text="", tolerance=TOLERANCE)

    assert judgment_passes(judgment)


def test_a_row_may_be_named_by_its_last_part_or_trailing_pieces() -> None:
    forms = row_name_forms("Acme_Region East - Store North - Late_Q3-2026", suffix_patterns=(SEASON_SUFFIX,))

    assert "storenorthlate" in forms
    assert row_name_forms(OVERALL) == ()
    assert row_is_named(
        "Store North – Late sold 5", "Acme_Region East - Store North - Late_Q3-2026", suffix_patterns=(SEASON_SUFFIX,)
    )


def test_prose_values_are_tied_to_the_store_named_before_them() -> None:
    values = anchored_values("North had 1,200 units and South had 800 units.", ["North", "South"], INVENTORY_VOCABULARY)

    assert {(value.label, value.value) for value in values} == {("North", 1200.0), ("South", 800.0)}


def test_relative_tolerance_widens_for_shorthand_values() -> None:
    assert TOLERANCE("units", 1_200_000.0, 1_234_567.0, abbreviated=True)
    assert not TOLERANCE("units", 1_200_000.0, 1_234_567.0)
    assert RelativeTolerance(per_metric={"units": 0.05})("units", 1_200_000.0, 1_234_567.0)
