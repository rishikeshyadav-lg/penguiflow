from __future__ import annotations

import pytest
from judging_fixtures import INVENTORY_VOCABULARY

from learning_control_plane.judging import MetricVocabulary, RenderedOutput, metric_named, stated_values


def _values(answer: str, *rendered: RenderedOutput, implicit: str | None = None) -> set[tuple[str, float, str]]:
    return {
        (value.metric, round(value.value, 6), value.label)
        for value in stated_values(answer, INVENTORY_VOCABULARY, rendered, implicit_metric=implicit)
    }


def test_a_tall_metric_table_gives_each_metric_its_value_under_the_heading() -> None:
    answer = "### North Store\n\n| Metric | Value |\n|---|---|\n| **Units sold** | 1,872 |\n| Sell-through | 42% |\n"

    assert _values(answer) == {("units", 1872.0, "North Store"), ("sell_through", 0.42, "North Store")}


def test_a_wide_table_gives_each_row_its_own_values() -> None:
    answer = "| Store | Units | Revenue |\n|---|---|---|\n| North | 1,452 | $9,000 |\n| South | 306 | $1,200 |\n"

    assert _values(answer) == {
        ("units", 1452.0, "North"),
        ("revenue", 9000.0, "North"),
        ("units", 306.0, "South"),
        ("revenue", 1200.0, "South"),
    }


def test_a_tall_table_with_one_column_per_store_labels_values_by_column() -> None:
    answer = "| Metric | North | South |\n|---|---|---|\n| Revenue | $1,000 | $2,500.50 |\n"

    assert _values(answer) == {("revenue", 1000.0, "North"), ("revenue", 2500.5, "South")}


def test_values_in_a_rendered_table_count_as_stated() -> None:
    table = RenderedOutput(
        "table",
        {
            "columns": [{"field": "store", "header": "Store"}, {"field": "units", "header": "Units"}],
            "rows": [{"store": "North", "units": 1452}],
        },
    )

    assert _values("Neither store is named here.", table) == {("units", 1452.0, "North")}


def test_values_in_a_rendered_report_section_are_labelled_by_the_section_title() -> None:
    report = RenderedOutput(
        "report",
        {
            "sections": [
                {
                    "title": "Store 4411",
                    "content": "| Metric | Total |\n|---|---|\n| Returns | 268 |",
                    "subsections": [{"title": "South", "content": "Units sold: 12"}],
                }
            ]
        },
    )

    assert _values("See the report.", report) == {("returns", 268.0, "Store 4411"), ("units", 12.0, "South")}


def test_a_metric_value_line_is_read_with_its_heading() -> None:
    answer = "### East Depot\n- **Units:** 13,696 | **Returns:** 250"

    assert _values(answer) == {("units", 13696.0, "East Depot"), ("returns", 250.0, "East Depot")}


def test_a_line_starting_with_a_name_is_labelled_by_that_name() -> None:
    assert _values("West Depot: units 1,500") == {("units", 1500.0, "West Depot")}


def test_in_prose_each_number_belongs_to_the_metric_named_next_to_it() -> None:
    assert _values("It sold 16.7K units and made $251K revenue.") == {("units", 16700.0, ""), ("revenue", 251000.0, "")}


def test_a_latest_week_value_is_not_read_as_a_total() -> None:
    assert {metric for metric, _, _ in _values("- **Latest Units:** 2 | **Returns:** 2")} == {"returns"}


def test_a_share_column_is_not_read_as_a_metric() -> None:
    answer = "| Store | Units | Share of Units |\n|---|---|---|\n| North | 100 | 82.6% |\n"

    assert _values(answer) == {("units", 100.0, "North")}


def test_digits_inside_a_date_are_not_read_as_values() -> None:
    assert _values("Units were 1,000 for the season 9/14/26 - 9/27/26.") == {("units", 1000.0, "")}


def test_a_rate_written_as_a_bare_number_keeps_both_readings() -> None:
    answer = "| Metric | Value |\n|---|---|\n| Sell-through | 0.42 |\n"

    assert _values(answer) == {("sell_through", 0.42, ""), ("sell_through", 0.0042, "")}


def test_a_percent_on_a_count_metric_is_not_a_value() -> None:
    assert _values("| Metric | Value |\n|---|---|\n| Units | 40% |\n") == set()


def test_a_number_with_no_metric_named_is_read_as_the_only_metric_asked_for() -> None:
    values = stated_values("The north store sold 1,200.", INVENTORY_VOCABULARY, implicit_metric="units")

    assert [(value.metric, value.value, value.explicit) for value in values] == [("units", 1200.0, False)]


def test_a_number_with_no_metric_named_is_skipped_when_several_metrics_are_asked_for() -> None:
    assert _values("The north store sold 1,200.") == set()


def test_metric_names_map_to_canonical_metrics_and_qualified_names_do_not() -> None:
    assert metric_named("**Units Sold**", INVENTORY_VOCABULARY) == "units"
    assert metric_named("Sell-through rate (%)", INVENTORY_VOCABULARY) == "sell_through"
    assert metric_named("WoW change in units", INVENTORY_VOCABULARY) is None
    assert metric_named("Store", INVENTORY_VOCABULARY) is None


def test_a_vocabulary_needs_at_least_one_alias() -> None:
    with pytest.raises(ValueError, match="at least one alias"):
        MetricVocabulary(aliases={"units": ()})


def test_a_rendered_table_without_columns_states_nothing() -> None:
    assert _values("", RenderedOutput("table", {"rows": [{"units": 3}]})) == set()
