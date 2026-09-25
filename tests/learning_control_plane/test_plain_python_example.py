"""The plain-Python example runs the whole loop without PenguiFlow, MLflow or a model."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

from examples.lcp_plain_python_agent.flow import InventoryAgent, inventory_reference, judge_run, run_example
from learning_control_plane.judging import outcome_of

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "lcp_plain_python_agent" / "flow.py"


def test_the_example_judges_mines_gates_and_reaches_the_review_queue(tmp_path: Path) -> None:
    summary = asyncio.run(run_example(tmp_path))

    assert summary["history_outcomes"] == {
        "How many units did North Store sell?": "verified",
        "How many units did South Store sell?": "verified",
        "How many units did East Depot sell?": "verified",
        "What revenue did West Depot make?": "agent_error",
        "How is North doing?": "handled_correctly",
        "Which SKUs does East Depot carry?": "verified",
    }
    assert summary["published_documents"] == 6
    assert summary["mined_pattern"] == "units:query_stock"
    assert summary["golden_set"]["passed"]
    assert summary["gate"]["approved"]
    assert summary["gate"]["verified_success"]["excluded_case_ids"] == ["held-out-3"]
    assert summary["review_queue"] == [summary["candidate_id"]]


def test_rerunning_the_example_in_the_same_store_publishes_nothing_new(tmp_path: Path) -> None:
    asyncio.run(run_example(tmp_path))

    # The same runs publish the same documents again, so a rerun in the same place is a no-op.
    summary = asyncio.run(run_example(tmp_path))

    assert summary["published_documents"] == 6


def test_a_wrong_figure_fails_the_judge() -> None:
    run = InventoryAgent().run("How many units did North Store sell?")
    wrong = type(run)(run.question, run.steps, "North Store sold 1,900 units.")

    assert outcome_of(judge_run(wrong, asyncio.run(inventory_reference(wrong)))) == "failed"


def test_a_sku_answer_that_names_the_note_instead_of_the_items_fails() -> None:
    run = InventoryAgent().run("Which SKUs does East Depot carry?")
    wrong = type(run)(run.question, run.steps, "East Depot carries ... [7 more items].")

    assert outcome_of(judge_run(wrong, asyncio.run(inventory_reference(wrong)))) == "failed"


def test_the_example_runs_as_a_script() -> None:
    completed = subprocess.run([sys.executable, str(EXAMPLE)], capture_output=True, text=True, check=True, timeout=120)

    assert '"review_queue"' in completed.stdout
