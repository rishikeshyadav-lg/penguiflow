"""Shadow comparison: run a candidate on recorded production inputs and diff it against what production did.

Shadow traffic that compares only final answers tests half the pipeline. This runs the candidate over the
inputs of recorded production runs, with its variant marked `dry_run`, and diffs each result against the
recorded run by decisions: tools, arguments, cost, guardrails, policy findings.

`dry_run` is a request, not a guarantee. Nothing here sandboxes tools or replays their results: the agent
must honour `variant.config["dry_run"]` itself so its actions take no effect. This module builds the
comparison and the report, and says how often an answer-only check would have been fooled.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from .diffing import TrajectoryDiff, diff_runs
from .evaluation import EvaluationCase, EvaluationVariant, RunOne
from .execution import RunSettings, run_repeated
from .golden import GoldenTrajectory
from .policy import PolicyCheck
from .prediction import PredictionResult

ShadowReference = PredictionResult | GoldenTrajectory


@dataclass(frozen=True, slots=True)
class ShadowCase:
    """One recorded input: how the candidate's run differs from production, or why it could not be compared."""

    case_id: str
    diff: TrajectoryDiff | None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ShadowReport:
    """The comparison across every recorded input."""

    cases: Sequence[ShadowCase]

    @property
    def compared(self) -> int:
        return sum(1 for case in self.cases if case.diff is not None)

    @property
    def failed(self) -> int:
        return len(self.cases) - self.compared

    @property
    def answer_agreement_rate(self) -> float:
        """The share of compared cases with the same answer: all an answer-only shadow test would see."""

        return self._rate(lambda diff: diff.same_answer)

    @property
    def full_agreement_rate(self) -> float:
        """The share of compared cases with the same answer by the same process."""

        return self._rate(lambda diff: diff.identical)

    @property
    def process_only_differences(self) -> tuple[str, ...]:
        """Cases an answer-only comparison would call identical but whose process differs."""

        return tuple(
            case.case_id
            for case in self.cases
            if case.diff is not None and case.diff.same_answer and not case.diff.process_identical
        )

    def _rate(self, wanted: Callable[[TrajectoryDiff], bool]) -> float:
        diffs = [case.diff for case in self.cases if case.diff is not None]
        if not diffs:
            return 0.0
        return sum(1 for diff in diffs if wanted(diff)) / len(diffs)


async def _ran(case: EvaluationCase, output: PredictionResult) -> dict[str, float]:
    return {"ran": 1.0}


async def shadow_compare(
    recorded: Mapping[str, ShadowReference],
    cases: Sequence[EvaluationCase],
    run_one: RunOne,
    *,
    settings: RunSettings = RunSettings(),
    policy: PolicyCheck | None = None,
    variant_id: str = "shadow",
) -> ShadowReport:
    """Run `run_one` over each case with a `dry_run` variant and diff it against the recorded production run.

    Every case must have a recorded run to compare with; a missing one is an error, not a skipped case.
    A candidate run that fails is listed with its error and is never dropped.
    """

    missing = [case.case_id for case in cases if case.case_id not in recorded]
    if missing:
        raise ValueError(f"no recorded production run for cases {missing}")
    if settings.repeats != 1:
        raise ValueError("a shadow comparison runs each case once")
    variant = EvaluationVariant(variant_id, config={"dry_run": True})
    run = await run_repeated(cases, [variant], run_one, _ran, settings)
    results = []
    for row in run.rows:
        if row.result.error is not None:
            results.append(ShadowCase(row.case_id, None, row.result.error))
            continue
        results.append(ShadowCase(row.case_id, diff_runs(recorded[row.case_id], row.result.output, policy=policy)))
    return ShadowReport(cases=tuple(results))


__all__ = ["ShadowCase", "ShadowReference", "ShadowReport", "shadow_compare"]
