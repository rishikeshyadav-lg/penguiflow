"""The proof a consumer can run without this repository.

The five-agent proof next door reads cases from a sibling package and exercises PenguiFlow and LangChain
adapters, so it cannot run against a standalone install. `agent_evals.selfcheck` ships inside the package and
makes the same two claims checkable by whoever installed it.
"""

from __future__ import annotations

import asyncio

from agent_evals import selfcheck


def test_importing_the_package_pulls_in_no_framework_or_backend() -> None:
    """Checked in a fresh interpreter, so modules another test already imported cannot hide a leak."""

    assert selfcheck.check_imports_stay_clean() == []


def test_agents_written_in_unrelated_shapes_score_the_same() -> None:
    """A plain function, a coroutine function and a callable object implement one behaviour; one scorecard."""

    scores = asyncio.run(selfcheck.check_every_agent_shape_scores_the_same())

    assert set(scores) == {"plain function", "coroutine function", "callable object"}
    assert selfcheck._disagreements(scores) == []
    assert scores["plain function"] == {"success_rate": 1.0, "tool_selection": 1.0}


def test_measured_time_and_spend_are_left_out_of_the_comparison() -> None:
    """Two identical runs differ in wall-clock time, so comparing those would fail for the wrong reason."""

    assert "p95_latency" in selfcheck.MEASURED_NOT_DERIVED
    assert "p95_latency" not in asyncio.run(selfcheck._scores_for(selfcheck._plain_function))


def test_the_check_reports_a_disagreement_rather_than_passing_quietly() -> None:
    """The error path: if one shape scored differently, the run must say which and fail."""

    scores = {"plain function": {"success_rate": 1.0}, "odd one out": {"success_rate": 0.5}}

    disagreements = selfcheck._disagreements(scores)

    assert len(disagreements) == 1
    assert "odd one out" in disagreements[0]


def test_it_exits_non_zero_when_a_claim_fails(monkeypatch) -> None:
    """A consumer's CI relies on the exit code, not on reading the output."""

    monkeypatch.setattr(selfcheck, "check_imports_stay_clean", lambda: ["penguiflow"])

    assert asyncio.run(selfcheck.main()) == 1


def test_it_exits_zero_when_both_claims_hold() -> None:
    assert asyncio.run(selfcheck.main()) == 0
