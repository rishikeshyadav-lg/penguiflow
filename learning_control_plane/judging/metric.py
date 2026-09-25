"""Let the promotion gate re-judge every baseline and candidate run with the judge kit.

The gate used to compare only the integrator's own metric numbers; nothing re-judged a candidate's
answers the way mined runs were judged. `VerificationMetric` judges each evaluated run and reports:

- `verified_success`: 1 when the run verified;
- `hard_failure`: 1 when a judged run failed on a hard failure (a wrong result, wrong scope, an
  invented figure, a truncated answer);
- `handled_correctly`: 1 when the run was a correct non-answer (a clarification, a confirmed "no
  data", a service that is off);
- `judged`: 1 when the run had something to check, 0 when it was handled correctly or not judgeable.

`verification_policy` builds a promotion policy on those numbers: verified success must improve
and hard failures must not rise, both measured only over pairs judged on both arms (`judged` is
their denominator), so a run handled correctly on one arm never counts as a failure on it.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from ..control_plane.control_plane import PromotionPolicy
from ..evaluation.evaluation import EvaluationCase, Metric, MetricSpecification
from ..evaluation.verification import InvestigationVerification
from .outcomes import outcome_of
from .runs import AgentRun

VERIFIED_SUCCESS = "verified_success"
HARD_FAILURE = "hard_failure"
HANDLED_CORRECTLY = "handled_correctly"
JUDGED = "judged"
_JUDGED_OUTCOMES = frozenset({"verified", "failed", "agent_error"})
# Judges one run against the reference (None when there is none), e.g. `OutcomeLadder.judge`.
RunJudge = Callable[[AgentRun, Any], InvestigationVerification]


class VerificationMetric:
    """A `Metric` that judges each evaluated run and reports its outcome as numbers."""

    def __init__(
        self,
        judge: RunJudge,
        to_run: Callable[[EvaluationCase, Any], AgentRun],
        reference: Callable[[AgentRun], Awaitable[Any]] | None = None,
    ) -> None:
        self._judge = judge
        self._to_run = to_run
        self._reference = reference

    async def __call__(self, case: EvaluationCase, output: Any) -> Mapping[str, float]:
        """Judge one run's output and return its outcome metrics."""

        run = self._to_run(case, output)
        reference = await self._reference(run) if self._reference is not None else None
        return outcome_metrics(self._judge(run, reference))


def outcome_metrics(verification: InvestigationVerification) -> dict[str, float]:
    """Return the gate's numbers for one judged run."""

    outcome = outcome_of(verification)
    judged = outcome in _JUDGED_OUTCOMES
    hard_failures = verification.final_answer.hard_failure_codes if verification.final_answer else ()
    return {
        VERIFIED_SUCCESS: float(outcome == "verified"),
        HARD_FAILURE: float(judged and outcome != "verified" and bool(hard_failures)),
        HANDLED_CORRECTLY: float(outcome == "handled_correctly"),
        JUDGED: float(judged),
    }


class CombinedMetric:
    """Run several metrics on each case and merge their numbers; two metrics may not report the same name."""

    def __init__(self, *metrics: Metric) -> None:
        if not metrics:
            raise ValueError("a combined metric needs at least one metric")
        self._metrics = metrics

    async def __call__(self, case: EvaluationCase, output: Any) -> Mapping[str, float]:
        """Return every metric's numbers for one case."""

        combined: dict[str, float] = {}
        for metric in self._metrics:
            values = metric(case, output)
            if inspect.isawaitable(values):
                values = await values
            overlap = set(combined) & set(values)
            if overlap:
                raise ValueError(f"combined metrics report the same names: {sorted(overlap)}")
            combined.update(values)
        return combined


def verification_policy(
    policy_version: str,
    *,
    minimum_verified_improvement: float = 0.0,
    require_golden_set: bool = True,
    **policy_options: Any,
) -> PromotionPolicy:
    """Return a promotion policy on re-judged outcomes: more verified runs, and no more hard failures.

    Both metrics count only pairs that were judged on both arms. Further `PromotionPolicy` options
    (minimum cases, confidence intervals, other metrics) pass through unchanged.
    """

    extra_specifications = tuple(policy_options.pop("metric_specifications", ()))
    protected = tuple(policy_options.pop("protected_metrics", ()))
    return PromotionPolicy(
        policy_version=policy_version,
        primary_metric=VERIFIED_SUCCESS,
        minimum_primary_improvement=minimum_verified_improvement,
        metric_specifications=(
            MetricSpecification(VERIFIED_SUCCESS, "higher_is_better", denominator=JUDGED),
            MetricSpecification(HARD_FAILURE, "lower_is_better", denominator=JUDGED),
            *extra_specifications,
        ),
        protected_metrics=(HARD_FAILURE, *protected),
        require_golden_set=require_golden_set,
        **policy_options,
    )


__all__ = [
    "CombinedMetric",
    "HANDLED_CORRECTLY",
    "HARD_FAILURE",
    "JUDGED",
    "RunJudge",
    "VERIFIED_SUCCESS",
    "VerificationMetric",
    "outcome_metrics",
    "verification_policy",
]
