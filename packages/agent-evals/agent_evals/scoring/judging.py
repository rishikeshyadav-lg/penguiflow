"""Plug in a domain judge, and prove it is trustworthy before its verdicts count.

Some agents cannot be scored by a string comparison: whether an answer is right may need the agent's own
rubric, a reference computed from the real data, or a model reading the answer. That is a domain judge. This
module gives it a place to plug in (`DomainJudge`), turns its verdicts into scores (`JudgeScorer`), and
measures how far it can be trusted: run it on cases a person already labelled and count how often it agrees.
A judge whose agreement is not measured should not decide anything.

The agreement harness is lifted from the campaign's golden-set check and its re-judge table.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Awaitable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..core.evaluation import EvaluationCase, _await_value


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    """What a judge concluded about one run: an outcome label, and the reason codes behind it."""

    outcome: str
    codes: Sequence[str] = ()

    def __post_init__(self) -> None:
        if not self.outcome.strip():
            raise ValueError("outcome must be non-empty")
        object.__setattr__(self, "codes", tuple(self.codes))


class DomainJudge(Protocol):
    """Judges one run of one case. It may be a plain function or a coroutine function."""

    def __call__(self, case: EvaluationCase, output: Any) -> JudgeVerdict | Awaitable[JudgeVerdict]: ...


class JudgeClient(Protocol):
    """A language model as a judge needs it: text in, text out. No provider is assumed.

    Implement it over whichever client you use. It may be a plain function or a coroutine function.
    """

    def __call__(self, prompt: str) -> str | Awaitable[str]: ...


@dataclass(frozen=True, slots=True)
class JudgeScorer:
    """Makes a scorer from a judge and a table of what each outcome is worth.

    `scores` maps an outcome label to a score, for example `{"verified": 1.0, "handled_correctly": 1.0,
    "failed": 0.0}`. An outcome the table does not name is an error, not a zero: a label nobody planned for
    is a judge that changed under you.
    """

    judge: DomainJudge
    scores: Mapping[str, float]
    name: str = "judged_success"

    async def __call__(self, case: EvaluationCase, output: Any) -> dict[str, float]:
        verdict = await _await_value(self.judge(case, output))
        if verdict.outcome not in self.scores:
            raise ValueError(f"the judge gave outcome {verdict.outcome!r}, which has no score")
        return {self.name: float(self.scores[verdict.outcome])}


@dataclass(frozen=True, slots=True)
class LabelledExample:
    """A run a person labelled: what a correct judge should conclude about it."""

    key: str
    case: EvaluationCase
    output: Any
    expected: str
    group: str | None = None


@dataclass(frozen=True, slots=True)
class Comparison:
    """One case: the label, what the judge said, and why."""

    key: str
    expected: str
    got: str
    codes: Sequence[str] = ()
    group: str | None = None

    @property
    def agrees(self) -> bool:
        return self.expected == self.got


@dataclass(frozen=True, slots=True)
class AgreementReport:
    """How often a judge agreed with the labels, and where it did not."""

    comparisons: Sequence[Comparison]

    @property
    def total(self) -> int:
        return len(self.comparisons)

    @property
    def agree(self) -> int:
        return sum(1 for comparison in self.comparisons if comparison.agrees)

    @property
    def rate(self) -> float:
        return self.agree / self.total if self.total else 0.0

    @property
    def disagreements(self) -> tuple[Comparison, ...]:
        return tuple(comparison for comparison in self.comparisons if not comparison.agrees)

    @property
    def by_expected(self) -> dict[str, dict[str, int]]:
        """Per expected outcome: how many cases, and how many the judge agreed on."""

        table: dict[str, Counter[str]] = {}
        for comparison in self.comparisons:
            row = table.setdefault(comparison.expected, Counter())
            row["cases"] += 1
            row["agree"] += comparison.agrees
        return {outcome: dict(counts) for outcome, counts in table.items()}

    @property
    def confusion(self) -> dict[tuple[str, str], int]:
        """How often each (expected, judged) pair of outcomes occurred."""

        return dict(Counter((c.expected, c.got) for c in self.comparisons))

    def by_group(self, success_outcomes: Collection[str]) -> dict[str, dict[str, int]]:
        """Per group: cases, agreements, and which way the disagreements went.

        `judge_passed_label_failed`: the judge called a success what the label did not. `judge_failed_label_passed`:
        the reverse. `different_outcome`: both failed or both succeeded but under different labels.
        """

        table: dict[str, Counter[str]] = {}
        for comparison in self.comparisons:
            row = table.setdefault(comparison.group or "", Counter())
            row["cases"] += 1
            if comparison.agrees:
                row["agree"] += 1
            elif comparison.got in success_outcomes:
                row["judge_passed_label_failed"] += 1
            elif comparison.expected in success_outcomes:
                row["judge_failed_label_passed"] += 1
            else:
                row["different_outcome"] += 1
        return {group: dict(counts) for group, counts in table.items()}

    def summary(self) -> str:
        return f"agreement: {self.agree} of {self.total} ({self.rate:.0%})"


def agreement_report(comparisons: Sequence[Comparison]) -> AgreementReport:
    """A report from verdicts that are already in hand, for example ones stored from an earlier run."""

    return AgreementReport(tuple(comparisons))


async def validate_judge(examples: Sequence[LabelledExample], judge: DomainJudge) -> AgreementReport:
    """Run a judge on labelled examples and report its agreement.

    A judge that raises is recorded as a disagreement with the outcome `judge_error:<type>`; it is never skipped.
    """

    comparisons = []
    for example in examples:
        try:
            verdict = await _await_value(judge(example.case, example.output))
            got, codes = verdict.outcome, tuple(verdict.codes)
        except Exception as error:
            got, codes = f"judge_error:{type(error).__name__}", ()
        comparisons.append(Comparison(example.key, example.expected, got, codes, example.group))
    return AgreementReport(tuple(comparisons))


@dataclass(frozen=True, slots=True)
class StoredVerdicts:
    """A judge that returns verdicts recorded earlier, by case id, for checking the harness on old runs."""

    verdicts: Mapping[str, JudgeVerdict] = field(default_factory=dict)

    def __call__(self, case: EvaluationCase, output: Any) -> JudgeVerdict:
        if case.case_id not in self.verdicts:
            raise KeyError(f"no stored verdict for {case.case_id}")
        return self.verdicts[case.case_id]


__all__ = [
    "AgreementReport",
    "Comparison",
    "DomainJudge",
    "JudgeClient",
    "JudgeScorer",
    "JudgeVerdict",
    "LabelledExample",
    "StoredVerdicts",
    "agreement_report",
    "validate_judge",
]
