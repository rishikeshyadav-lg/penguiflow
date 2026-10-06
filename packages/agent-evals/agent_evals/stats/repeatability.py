"""How repeatable a result is: pass@k, pass^k, and whether each case is consistent.

A single success rate hides how an agent behaves on its own next run. pass@k asks whether at least one
of k attempts succeeded; pass^k asks whether all k did, and it falls fast: an agent that succeeds 75% of
the time per attempt has about a 42% chance of three wins in a row. The customer does not experience the
average run, they experience their own.

With n recorded runs of a case and c successes, the unbiased estimators are
pass@k = 1 - C(n-c, k) / C(n, k) and pass^k = C(c, k) / C(n, k). A run counts as a success when it did
not fail and its score reaches `success_threshold`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import comb
from typing import Literal

from ..running.execution import RepeatedRun

Consistency = Literal["consistently_correct", "consistently_incorrect", "intermittent"]


def _check_counts(n: int, c: int, k: int) -> None:
    if n < 1 or not 0 <= c <= n:
        raise ValueError("need at least one run and 0 <= successes <= runs")
    if not 1 <= k <= n:
        raise ValueError("k must be at least 1 and must not exceed the number of runs")


def pass_at_k(n: int, c: int, k: int) -> float:
    """The chance that at least one of k attempts succeeds, given c successes in n runs."""

    _check_counts(n, c, k)
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """The chance that all k attempts succeed (pass^k), given c successes in n runs."""

    _check_counts(n, c, k)
    return comb(c, k) / comb(n, k)


def expected_pass_hat_k(rate: float, k: int) -> float:
    """pass^k for an agent that succeeds with probability `rate` on each independent attempt."""

    if not 0 <= rate <= 1 or k < 1:
        raise ValueError("rate must be between 0 and 1 and k at least 1")
    return rate**k


def consistency_label(scores: Sequence[float], *, failed: int = 0, missing: int = 0) -> Consistency:
    """Classify one case's repeated scores. Meaningful for a binary metric.

    `consistently_correct` needs every run to score exactly 1.0, `consistently_incorrect` every run exactly 0.0;
    anything else, or any failed or missing run, is `intermittent`. Lifted from the campaign's calibration.
    """

    if failed or missing or not scores:
        return "intermittent"
    if min(scores) == 1.0:
        return "consistently_correct"
    if max(scores) == 0.0:
        return "consistently_incorrect"
    return "intermittent"


@dataclass(frozen=True, slots=True)
class CaseRepeatability:
    runs: int
    successes: int
    pass_at_k: float
    pass_hat_k: float
    consistency: Consistency


@dataclass(frozen=True, slots=True)
class RepeatabilityReport:
    """pass@k and pass^k averaged over cases, with each case's own numbers and consistency."""

    k: int
    metric: str
    success_threshold: float
    pass_at_k: float
    pass_hat_k: float
    cases: Mapping[str, CaseRepeatability]

    def consistency_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {"consistently_correct": 0, "intermittent": 0, "consistently_incorrect": 0}
        for case in self.cases.values():
            counts[case.consistency] += 1
        return counts


def repeatability(
    run: RepeatedRun, variant_id: str, metric: str, *, k: int, success_threshold: float = 1.0
) -> RepeatabilityReport:
    """pass@k, pass^k and consistency for one variant of a repeated run."""

    # A failed run is recorded as None; it is never a success and never has a score.
    by_case: dict[str, list[float | None]] = {}
    for row in run.rows_for(variant_id):
        if row.result.error is not None:
            outcome = None
        elif metric in row.result.metrics:
            outcome = row.result.metrics[metric]
        else:
            raise ValueError(f"the run of {row.key} did not produce metric {metric!r}")
        by_case.setdefault(row.case_id, []).append(outcome)
    if not by_case:
        raise ValueError(f"the run has no rows for variant {variant_id!r}")

    cases: dict[str, CaseRepeatability] = {}
    for case_id, outcomes in by_case.items():
        runs = len(outcomes)
        if runs < k:
            raise ValueError(f"case {case_id} has {runs} runs, fewer than k={k}")
        scores = [outcome for outcome in outcomes if outcome is not None]
        successes = sum(1 for score in scores if score >= success_threshold)
        cases[case_id] = CaseRepeatability(
            runs=runs,
            successes=successes,
            pass_at_k=pass_at_k(runs, successes, k),
            pass_hat_k=pass_hat_k(runs, successes, k),
            consistency=consistency_label(scores, failed=runs - len(scores)),
        )
    return RepeatabilityReport(
        k=k,
        metric=metric,
        success_threshold=success_threshold,
        pass_at_k=sum(case.pass_at_k for case in cases.values()) / len(cases),
        pass_hat_k=sum(case.pass_hat_k for case in cases.values()) / len(cases),
        cases=cases,
    )


__all__ = [
    "CaseRepeatability",
    "Consistency",
    "RepeatabilityReport",
    "consistency_label",
    "expected_pass_hat_k",
    "pass_at_k",
    "pass_hat_k",
    "repeatability",
]
