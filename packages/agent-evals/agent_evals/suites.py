"""Two kinds of suite, two rules: a regression suite guards, a capability suite measures.

A regression suite protects what already works, so it wants a hard threshold near 100%: a case
passes when its score reaches `case_pass_score`, and the suite passes when enough cases do. A
capability suite exists to raise the ceiling, so a hard threshold would hide small gains: it reports the
mean partial-credit score with a case-clustered interval and gives no pass or fail. Putting both jobs
into one number is the mistake this keeps apart.

A case's score is the mean of its repeats, and a repeat that failed counts as 0, so a failure can
neither hide nor be averaged away by the runs that worked.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .datasets import DatasetManifest
from .evaluation import EvaluationDataset, EvaluationVariant, RunOne
from .execution import JsonlRowSink, RepeatedRun, RunSettings, run_repeated
from .runner import Scorers
from .statistics import BootstrapInterval, PairedPrompt, paired_bootstrap


@dataclass(frozen=True, slots=True)
class SuiteRule:
    """How a regression suite is judged. A capability suite ignores the two thresholds."""

    metric: str
    case_pass_score: float = 1.0
    required_pass_rate: float = 1.0
    confidence: float = 0.95
    resamples: int = 10_000

    def __post_init__(self) -> None:
        if not self.metric.strip():
            raise ValueError("metric must be non-empty")
        for name, value in (("case_pass_score", self.case_pass_score), ("required_pass_rate", self.required_pass_rate)):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class SuiteVerdict:
    """What a suite's rule made of a run. `passed` is None for a capability suite, which has no threshold."""

    suite: str
    metric: str
    case_count: int
    mean_score: float
    passed: bool | None
    pass_rate: float | None
    failing_case_ids: Sequence[str]
    interval: BootstrapInterval | None
    rule: str


def case_scores(run: RepeatedRun, variant_id: str, metric: str) -> dict[str, float]:
    """Each case's mean score over its repeats, with a failed repeat counted as 0."""

    scores: dict[str, list[float]] = {}
    for row in run.rows_for(variant_id):
        if row.result.error is not None:
            value = 0.0
        elif metric in row.result.metrics:
            value = row.result.metrics[metric]
        else:
            raise ValueError(f"the run of {row.key} did not produce metric {metric!r}")
        scores.setdefault(row.case_id, []).append(value)
    return {case_id: sum(values) / len(values) for case_id, values in scores.items()}


def suite_verdict(run: RepeatedRun, variant_id: str, manifest: DatasetManifest, rule: SuiteRule) -> SuiteVerdict:
    """Judge one variant's run by the rule its dataset's suite type calls for."""

    scores = case_scores(run, variant_id, rule.metric)
    if not scores:
        raise ValueError(f"the run has no rows for variant {variant_id!r}")
    mean_score = sum(scores.values()) / len(scores)
    if manifest.suite == "regression":
        failing = tuple(case_id for case_id, score in scores.items() if score < rule.case_pass_score)
        pass_rate = 1 - len(failing) / len(scores)
        return SuiteVerdict(
            suite="regression",
            metric=rule.metric,
            case_count=len(scores),
            mean_score=mean_score,
            passed=pass_rate >= rule.required_pass_rate,
            pass_rate=pass_rate,
            failing_case_ids=failing,
            interval=None,
            rule=(
                f"regression: a case passes at a mean {rule.metric} of at least {rule.case_pass_score:g}; "
                f"{rule.required_pass_rate:.0%} of cases must pass"
            ),
        )
    interval = paired_bootstrap(
        [PairedPrompt(case_id, 0.0, score) for case_id, score in scores.items()],
        resamples=rule.resamples,
        confidence=rule.confidence,
    )
    return SuiteVerdict(
        suite="capability",
        metric=rule.metric,
        case_count=len(scores),
        mean_score=mean_score,
        passed=None,
        pass_rate=None,
        failing_case_ids=(),
        interval=interval,
        rule=(
            f"capability: partial credit; mean {rule.metric} with a {rule.confidence:.0%} case-clustered "
            "interval; no pass or fail"
        ),
    )


async def run_suite(
    dataset: EvaluationDataset,
    manifest: DatasetManifest,
    variants: Sequence[EvaluationVariant],
    run_one: RunOne,
    scorers: Scorers,
    *,
    metric_id: str,
    metric_version: str,
    settings: RunSettings = RunSettings(),
    sink: JsonlRowSink | None = None,
) -> RepeatedRun:
    """Run a frozen dataset, after checking it is the frozen one and the metric is the one it expects.

    Both checks happen before any case runs, so a mismatch costs nothing.
    """

    manifest.check(dataset)
    manifest.check_metric(metric_id, metric_version)
    return await run_repeated(dataset.cases, variants, run_one, scorers, settings, sink)


__all__ = ["SuiteRule", "SuiteVerdict", "case_scores", "run_suite", "suite_verdict"]
