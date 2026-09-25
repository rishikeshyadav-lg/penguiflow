from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import pytest

from learning_control_plane.judging import (
    AgentStep,
    ClassificationRead,
    MeaningCheck,
    MeaningQuestion,
    freeze_classification,
    majority,
    mining_skip_reason,
    tool_evidence,
)
from learning_control_plane.providers.typesafe_meaning import TypeSafeMeaningJudge, governance_acknowledged

QUESTIONS = (
    MeaningQuestion("invented_figure", "Quotes a figure no tool returned.", "yes", "no", "invented_figure", bar=0.75),
    MeaningQuestion("contradiction", "States a fact the tools contradict.", "yes", "no", "contradicts_tool_results"),
)
STEPS = (AgentStep("query_stock", {"store": "North"}, {"rows": [{"units": 5}]}),)


class _ScriptedJudge:
    """Returns one scripted read per call; an exception in the script is raised as a failed read."""

    def __init__(self, reads: Sequence[Mapping[str, Any] | Exception]) -> None:
        self._reads = list(reads)
        self.states: list[Mapping[str, str]] = []

    def read(self, state: Mapping[str, str], questions: Sequence[MeaningQuestion]) -> Mapping[str, Any]:
        self.states.append(state)
        scripted = self._reads.pop(0)
        if isinstance(scripted, Exception):
            raise scripted
        return scripted


def _check(reads: Sequence[Mapping[str, Any] | Exception]) -> MeaningCheck:
    return MeaningCheck(_ScriptedJudge(reads), QUESTIONS, reads=len(reads))


def test_a_finding_is_reported_when_the_mean_of_the_reads_reaches_the_bar() -> None:
    check = _check([{"contradiction": 0.9}, {"contradiction": 0.84}, {"contradiction": 0.86}])

    findings = check.check("How many units?", "North sold 5 units.", STEPS)

    assert findings.codes == ("contradicts_tool_results",)
    assert findings.notes["probabilities"]["contradiction"] == pytest.approx(0.8667, abs=1e-4)


def test_each_question_is_compared_with_its_own_bar() -> None:
    check = _check([{"invented_figure": 0.8, "contradiction": 0.8}])

    assert check.check("q", "a", STEPS).codes == ("invented_figure",)
    assert check.bars == {"invented_figure": 0.75, "contradiction": 0.85}


def test_a_failed_read_is_left_out_of_the_mean() -> None:
    check = _check([{"contradiction": 0.9}, TimeoutError("read timed out"), {"contradiction": 0.9}])

    findings = check.check("q", "a", STEPS)

    assert findings.codes == ("contradicts_tool_results",)
    assert len(findings.notes["reads"]) == 2


def test_when_every_read_fails_there_are_no_findings() -> None:
    check = _check([ConnectionError("down"), ConnectionError("down"), ConnectionError("down")])

    findings = check.check("q", "a", STEPS)

    assert findings.codes == ()
    assert findings.notes == {"error": "no_read_succeeded"}


def test_unknown_questions_and_non_numeric_answers_are_ignored() -> None:
    check = _check([{"contradiction": "high", "made_up": 1.0, "invented_figure": True}])

    findings = check.check("q", "a", STEPS)

    assert findings.codes == ()
    assert findings.notes["probabilities"] == {}


def test_the_judge_sees_the_question_the_shown_answer_and_the_tool_results() -> None:
    judge = _ScriptedJudge([{}])

    MeaningCheck(judge, QUESTIONS, reads=1).check("How many units?", "North sold 5 units.", STEPS)

    assert judge.states[0] == {
        "question": "How many units?",
        "answer_shown_to_user": "North sold 5 units.",
        "tool_results": '1. query_stock args={"store": "North"} -> {"rows": [{"units": 5}]}',
    }


def test_a_meaning_check_needs_a_read_and_uniquely_named_questions() -> None:
    with pytest.raises(ValueError, match="at least one read"):
        MeaningCheck(_ScriptedJudge([]), QUESTIONS, reads=0)
    with pytest.raises(ValueError, match="uniquely named"):
        MeaningCheck(_ScriptedJudge([]), (QUESTIONS[0], QUESTIONS[0]))


def test_tool_evidence_leaves_out_render_calls_and_cuts_long_lists() -> None:
    steps = (
        AgentStep("list_stores", {}, {"values": list(range(100))}),
        AgentStep("render_table", {"rows": []}, None),
        AgentStep("query_stock", {}, "x" * 3000),
    )

    evidence = tool_evidence(steps)

    assert "render_table" not in evidence
    assert "39]" in evidence and "40," not in evidence
    assert evidence.splitlines()[1].startswith("3. query_stock")
    assert evidence.endswith("…[cut]")


def test_typesafe_is_refused_without_the_governance_acknowledgement() -> None:
    assert TypeSafeMeaningJudge.from_environment({"TYPESAFE_API_KEY": "key"}) is None
    assert not governance_acknowledged({"TYPESAFE_GOVERNANCE_ACKNOWLEDGED": "no"})


def test_typesafe_is_refused_without_a_key() -> None:
    assert TypeSafeMeaningJudge.from_environment({"TYPESAFE_GOVERNANCE_ACKNOWLEDGED": "true"}) is None


def test_typesafe_is_used_when_acknowledged_and_keyed() -> None:
    judge = TypeSafeMeaningJudge.from_environment(
        {"TYPESAFE_GOVERNANCE_ACKNOWLEDGED": "yes", "TYPESAFE_API_KEY": " key ", "LCP_MEANING_JUDGE_MODEL": "jev-2"}
    )

    assert judge is not None
    assert judge.model_ref == "jev-2"


def test_an_empty_typesafe_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="API key"):
        TypeSafeMeaningJudge(api_key="  ")


def test_a_typesafe_read_asks_each_question_and_returns_its_probability(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_sdk = SimpleNamespace(
        Noul=lambda instructions, criteria: ("noul", instructions, criteria),
        NoulCriteria=lambda true, false: (true, false),
    )
    monkeypatch.setitem(sys.modules, "typesafe_sdk", fake_sdk)
    calls: list[dict[str, Any]] = []

    def system_one(**request: Any) -> SimpleNamespace:
        calls.append(request)
        return SimpleNamespace(
            answers={"contradiction": SimpleNamespace(noul=0.9), "invented_figure": SimpleNamespace(noul=None)}
        )

    judge = TypeSafeMeaningJudge(api_key="key", system_one=system_one)

    probabilities = judge.read({"question": "q"}, QUESTIONS)

    assert probabilities == {"contradiction": 0.9}
    assert calls[0]["model"] == "jev-latest"
    assert set(calls[0]["questions"]) == {"invented_figure", "contradiction"}


def test_a_meaning_check_over_typesafe_skips_a_read_the_sdk_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_sdk = SimpleNamespace(Noul=lambda **_: None, NoulCriteria=lambda **_: None)
    monkeypatch.setitem(sys.modules, "typesafe_sdk", fake_sdk)

    def system_one(**_: Any) -> None:
        raise RuntimeError("rate limited")

    check = MeaningCheck(TypeSafeMeaningJudge(api_key="key", system_one=system_one), QUESTIONS)

    assert check.check("q", "a", STEPS).notes == {"error": "no_read_succeeded"}


def test_the_category_is_the_confident_majority_of_the_reads() -> None:
    reads = [
        ClassificationRead("stock_level", 0.9, {"scope": "one_store", "metrics": ["units"]}),
        ClassificationRead("stock_level", 0.8, {"scope": "one_store", "metrics": ["units", "returns"]}),
        ClassificationRead("forecast", 0.99, {"scope": "all_stores", "metrics": ["units"]}),
    ]

    frozen = freeze_classification(reads, fields=("scope",), list_fields=("metrics",), min_confidence=0.7)

    assert frozen["category"] == "stock_level"
    assert frozen["confidence"] == 0.85
    assert frozen["profile"] == {"scope": "one_store", "metrics": ["units"]}
    assert frozen["reads"][2] == {"category": "forecast", "confidence": 0.99}


def test_a_majority_below_the_confidence_bar_is_kept_only_as_the_raw_category() -> None:
    reads = [ClassificationRead("stock_level", 0.6), ClassificationRead("stock_level", 0.65)]

    frozen = freeze_classification(reads, fields=(), min_confidence=0.7)

    assert frozen["category"] is None
    assert frozen["raw_category"] == "stock_level"


def test_there_is_no_majority_without_more_than_half_the_reads() -> None:
    assert majority(["a", "b"]) is None
    assert majority([None, None]) is None
    assert majority(["a", "a", "b"]) == "a"


def test_a_question_is_replayed_for_mining_only_with_a_mineable_frozen_category() -> None:
    mineable = {"stock_level"}

    assert mining_skip_reason(None, mineable_categories=mineable) == "not_classified"
    assert mining_skip_reason({"category": None}, mineable_categories=mineable) == "no_confident_category"
    assert mining_skip_reason({"category": "forecast"}, mineable_categories=mineable) == "not_mineable:forecast"
    assert mining_skip_reason({"category": "stock_level"}, mineable_categories=mineable) is None
