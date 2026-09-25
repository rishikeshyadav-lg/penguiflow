"""A fail-only check of what an answer means, for mistakes value matching cannot see.

Matching stated values to the data cannot tell that an answer never says which item a figure
belongs to, quotes a benchmark no tool returned, calls something "on track" with no goal data, or
states a fact the tool results contradict. A `MeaningCheck` puts yes/no questions about those to a
`MeaningJudge` (a model that returns a probability per question) and reports a finding when the
mean probability reaches that question's bar.

Lessons behind its shape:
- a single read is unstable: the same answer sat either side of the bar from run to run, so the
  answer is read several times and the mean is compared with the bar;
- one bar does not fit every question, so each question may carry its own;
- a failed read is skipped, and when every read fails there are simply no findings;
- findings can only turn a verified run into a failed one, never verify a run.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol

from .runs import AgentStep

FINDING_BAR = 0.85
_EVIDENCE_CHARACTER_LIMIT = 14_000
_ANSWER_CHARACTER_LIMIT = 9_000
_ARGUMENT_CHARACTER_LIMIT = 600
_RESULT_CHARACTER_LIMIT = 2_500
_ROWS_SHOWN = 40


@dataclass(frozen=True, slots=True)
class MeaningQuestion:
    """One yes/no question about an answer, the reason code a "yes" produces, and its own bar."""

    name: str
    instruction: str
    yes: str
    no: str
    reason_code: str
    bar: float | None = None


class MeaningJudge(Protocol):
    """A model that answers yes/no questions about one answer with a probability of "yes" each."""

    def read(self, state: Mapping[str, str], questions: Sequence[MeaningQuestion]) -> Mapping[str, float]:
        """Return question name -> probability of "yes" for one read; raise when the read fails."""
        ...


@dataclass(frozen=True, slots=True)
class MeaningFindings:
    """Reason codes for the judge, and the probabilities behind them for a reviewer."""

    codes: tuple[str, ...]
    notes: Mapping[str, Any] = field(default_factory=dict)


DEFAULT_MEANING_QUESTIONS: tuple[MeaningQuestion, ...] = (
    MeaningQuestion(
        name="question_not_answered",
        instruction=(
            "The answer leaves out a specific thing the question asked for: for example it gives a split without "
            "saying which figure belongs to which unit the question named, or it says the requested items could "
            "not be identified and gives other figures instead."
        ),
        yes="A specific requested item or split is missing from the answer.",
        no="The answer addresses what was asked, or asks the user to choose between several matching campaigns.",
        reason_code="question_not_answered",
    ),
    MeaningQuestion(
        name="unsupported_pacing",
        instruction=(
            "The answer states that delivery or pacing is on track, ahead or behind, or states under- or "
            "over-delivery against a goal, plan or budget, although no tool result contains a goal, plan or budget."
        ),
        yes="It asserts a pacing or delivery-against-goal verdict with no goal data.",
        no="It makes no such verdict, says pacing cannot be assessed, or only recommends checking pacing.",
        reason_code="unsupported_claim",
    ),
    MeaningQuestion(
        name="invented_benchmark",
        instruction=(
            "The answer states an industry benchmark, a typical value or range, or what research shows, as a "
            "number or range that no tool result returned."
        ),
        yes="It gives a benchmark, typical or research figure or range that is not in the tool results.",
        no="Every benchmark-like figure comes from a tool, or there is none; describing the returned numbers is fine.",
        reason_code="invented_figure",
        # A qualitative benchmark claim reads near 0.8, while correct answers stayed at or below about 0.63.
        bar=0.75,
    ),
    MeaningQuestion(
        name="contradiction",
        instruction=(
            "The answer states a concrete fact (a date, a name, a count, a total, which item is highest or lowest, "
            "or that something does not exist) that the tool results contradict. Rounding is not a contradiction."
        ),
        yes="A concrete stated fact conflicts with the tool results.",
        no="No stated fact conflicts with the tool results.",
        reason_code="contradicts_tool_results",
    ),
)


class MeaningCheck:
    """Read an answer several times with a `MeaningJudge` and report the questions whose mean reaches its bar."""

    def __init__(
        self,
        judge: MeaningJudge,
        questions: Sequence[MeaningQuestion] = DEFAULT_MEANING_QUESTIONS,
        *,
        reads: int = 3,
        finding_bar: float = FINDING_BAR,
    ) -> None:
        if reads < 1:
            raise ValueError("a meaning check needs at least one read")
        names = [question.name for question in questions]
        if not names or len(names) != len(set(names)):
            raise ValueError("meaning questions must be non-empty and uniquely named")
        self._judge = judge
        self.questions = tuple(questions)
        self.reads = reads
        self.finding_bar = finding_bar

    @property
    def bars(self) -> dict[str, float]:
        """Return each question's bar: its own, or the check's finding bar."""

        return {
            question.name: question.bar if question.bar is not None else self.finding_bar for question in self.questions
        }

    def check(self, question: str, shown_answer: str, steps: Sequence[AgentStep]) -> MeaningFindings:
        """Return the findings for one answer from the mean of its reads; failed reads are left out."""

        state = {
            "question": question,
            "answer_shown_to_user": _trimmed(shown_answer, _ANSWER_CHARACTER_LIMIT),
            "tool_results": tool_evidence(steps),
        }
        known = {meaning_question.name for meaning_question in self.questions}

        def read(_: int) -> dict[str, float] | None:
            try:
                probabilities = self._judge.read(state, self.questions)
            except Exception:  # noqa: BLE001 -- the check can only fail runs, so a failed read is skipped
                return None
            return {
                name: float(probability)
                for name, probability in probabilities.items()
                if name in known and isinstance(probability, (int, float)) and not isinstance(probability, bool)
            }

        with ThreadPoolExecutor(self.reads) as pool:
            reads = [probabilities for probabilities in pool.map(read, range(self.reads)) if probabilities is not None]
        if not reads:
            return MeaningFindings(codes=(), notes={"error": "no_read_succeeded"})
        names = {name for probabilities in reads for name in probabilities}
        mean = {
            name: sum(values) / len(values)
            for name in names
            if (values := [probabilities[name] for probabilities in reads if name in probabilities])
        }
        findings = self.findings_from_probabilities(mean)
        return MeaningFindings(codes=findings.codes, notes={**findings.notes, "reads": reads})

    def findings_from_probabilities(self, probabilities: Mapping[str, float]) -> MeaningFindings:
        """Return the reason codes of the questions whose probability reaches their bar."""

        bars = self.bars
        codes = tuple(
            dict.fromkeys(
                question.reason_code
                for question in self.questions
                if probabilities.get(question.name, 0.0) >= bars[question.name]
            )
        )
        return MeaningFindings(codes=codes, notes={"probabilities": dict(probabilities), "bars": bars})


def tool_evidence(steps: Sequence[AgentStep]) -> str:
    """Return a compact record of each tool call: its name, its arguments and what it returned.

    Render calls are left out (their content is already in the shown answer), and long row or value
    lists are cut to their first rows, so the record stays within a model's reading budget.
    """

    lines: list[str] = []
    for index, step in enumerate(steps, 1):
        if step.tool.startswith("render"):
            continue
        result = step.result
        if isinstance(result, Mapping) and isinstance(result.get("rows"), list):
            result = {**result, "rows": result["rows"][:_ROWS_SHOWN]}
        if isinstance(result, Mapping) and isinstance(result.get("values"), list):
            result = {**result, "values": result["values"][:_ROWS_SHOWN]}
        arguments = _trimmed(step.args, _ARGUMENT_CHARACTER_LIMIT)
        lines.append(f"{index}. {step.tool} args={arguments} -> {_trimmed(result, _RESULT_CHARACTER_LIMIT)}")
    return _trimmed("\n".join(lines), _EVIDENCE_CHARACTER_LIMIT)


def _trimmed(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    if len(text) <= limit:
        return text
    return text[:limit] + " …[cut]"


__all__ = [
    "DEFAULT_MEANING_QUESTIONS",
    "FINDING_BAR",
    "MeaningCheck",
    "MeaningFindings",
    "MeaningJudge",
    "MeaningQuestion",
    "tool_evidence",
]
