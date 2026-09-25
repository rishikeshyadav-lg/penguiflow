from __future__ import annotations

from learning_control_plane.judging import (
    AgentRun,
    RenderedOutput,
    asks_user_to_choose,
    entity_named,
    is_shortened_list_note,
    looks_truncated,
    rendered_text,
    shown_to_user,
    strip_shortened_list_note,
)


def _run(answer: str, *rendered: RenderedOutput) -> AgentRun:
    return AgentRun(question="How did the north store do?", steps=(), final_answer=answer, rendered=rendered)


def test_an_answer_that_stops_mid_sentence_looks_truncated() -> None:
    assert looks_truncated("The north store sold 1,200 units and the south store ")


def test_an_empty_answer_looks_truncated() -> None:
    assert looks_truncated("   \n ")


def test_an_answer_ending_in_punctuation_is_complete() -> None:
    assert not looks_truncated("The north store sold 1,200 units.")


def test_an_answer_ending_in_a_table_row_or_list_item_is_complete() -> None:
    assert not looks_truncated("| Metric | Value |\n|---|---|\n| Units | 5 |")
    assert not looks_truncated("- **Revenue:** $1,000")


def test_an_answer_ending_in_a_link_is_complete() -> None:
    assert not looks_truncated("See the stock sheet at https://example.com/stock")


def test_asking_which_of_the_listed_stores_is_a_clarification() -> None:
    answer = "I found North Store and North Outlet. Which one do you mean?"

    assert asks_user_to_choose(answer, ["North Store", "North Outlet"], is_named=entity_named)


def test_an_answer_about_one_store_is_not_a_clarification() -> None:
    answer = "North Store sold 1,200 units. Would you like me to add returns?"

    assert not asks_user_to_choose(answer, ["North Store", "North Outlet"], is_named=entity_named)


def test_a_choice_that_names_fewer_than_two_candidates_is_not_a_clarification() -> None:
    answer = "Which one do you mean: North Store?"

    assert not asks_user_to_choose(answer, ["North Store", "North Outlet"], is_named=entity_named)


def test_the_shortened_list_note_is_not_an_item() -> None:
    items = [{"sku": "A-1"}, "... [12 more items]"]

    assert is_shortened_list_note("... [12 more items]")
    assert not is_shortened_list_note("12 more items")
    assert strip_shortened_list_note(items) == [{"sku": "A-1"}]


def test_what_the_user_saw_includes_rendered_table_rows_and_report_sections() -> None:
    table = RenderedOutput(
        "table",
        {
            "columns": [{"field": "store", "header": "Store"}, {"field": "units"}],
            "rows": [{"store": "North", "units": 5}],
        },
    )
    report = RenderedOutput("report", {"sections": [{"title": "Summary", "content": "Stock is healthy."}]})

    shown = shown_to_user(_run("See below.", table, report))

    assert shown == (
        "See below.\n\n[Rendered table]\nStore | units\nNorth | 5\n\n[Rendered report]\n## Summary\nStock is healthy."
    )


def test_rendered_text_flattens_every_rendered_output() -> None:
    table = RenderedOutput("table", {"rows": [{"store": "North Store"}]})

    assert "North Store" in rendered_text(_run("See the table.", table))


def test_a_run_without_an_answer_shows_only_what_was_rendered() -> None:
    run = AgentRun(question="q", steps=(), final_answer=None)

    assert shown_to_user(run) == ""
