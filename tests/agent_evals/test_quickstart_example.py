"""The quickstart example runs, and evaluating a plain function really does take a few dozen lines of glue."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from _paths import EXAMPLES

EXAMPLE = EXAMPLES / "agent_evals_quickstart" / "flow.py"


def _load():
    spec = importlib.util.spec_from_file_location("agent_evals_quickstart_flow", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _section(source: str, start_marker: str, end_marker: str) -> str:
    """The text after the marker's own line and before the end marker."""

    after_start = source.split(start_marker)[1].split("\n", 1)[1]
    return after_start.split(end_marker)[0]


async def test_the_quickstart_evaluates_a_plain_function_and_reports_the_scorecard() -> None:
    markdown = await _load().main()

    assert "Result: passed (100.0% of 3 cases passed)" in markdown
    assert "| Task success rate | outcome | 1.000 rate |" in markdown
    assert "| Tool selection accuracy | trajectory | 1.000 score |" in markdown
    assert "| Plan adherence | trajectory | not measured |" in markdown
    assert "| Policy violations | policy | 0.000 share of runs |" in markdown


async def test_the_quickstart_agent_knows_nothing_about_the_framework() -> None:
    source = EXAMPLE.read_text()
    agent_section = _section(source, "# ---- the agent under test", "# ---- the glue")

    assert "agent_evals" not in agent_section
    assert "PredictionResult" not in agent_section


def test_the_glue_that_connects_a_function_to_the_framework_is_a_few_dozen_lines() -> None:
    source = EXAMPLE.read_text()
    glue = _section(source, "# ---- the glue", 'if __name__ == "__main__"')

    lines = [line for line in glue.splitlines() if line.strip() and not line.strip().startswith("#")]

    assert len(lines) <= 60, f"{len(lines)} lines of glue"
