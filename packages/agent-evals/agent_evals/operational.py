"""Operational scorers: is the run fast, cheap and sensibly sized enough to ship?

Accuracy alone does not make a product. An agent at 90% success costing $0.50 and 45 s is not the same
product as one at 90% costing $0.05 and 8 s. This module measures steps, cost and latency as first-class
results: tail latency and cost percentiles with case-clustered intervals, step counts and an expected
step band for a class of task, an efficiency ratio, and a detector for an agent stuck repeating itself.

Fewer steps are not automatically better: an agent can shorten its run by skipping a verification it
should not skip. So a step count is judged against the band typical for that kind of task, and a run
*under* the band is as much an alarm as one over it.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from .evaluation import EvaluationCase
from .execution import RepeatedRun, RunRow
from .statistics import percentile
from .steps import GenericStep
from .trajectory import trajectory_of


@dataclass(frozen=True, slots=True)
class PercentileEstimate:
    """One percentile with a case-clustered interval: repeats of a case are resampled together."""

    probability: float
    estimate: float
    lower: float
    upper: float
    sample_size: int


def clustered_percentile(
    groups: Mapping[str, Sequence[float]],
    probability: float,
    *,
    resamples: int = 2_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> PercentileEstimate:
    """A percentile of all values, with an interval from resampling whole groups (cases) with replacement."""

    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    keys = sorted(key for key, values in groups.items() if values)
    if not keys:
        raise ValueError("there are no values to take a percentile of")
    pooled = [value for key in keys for value in groups[key]]
    generator = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        drawn = [generator.choice(keys) for _ in keys]
        estimates.append(percentile([value for key in drawn for value in groups[key]], probability))
    tail = (1 - confidence) / 2
    return PercentileEstimate(
        probability=probability,
        estimate=percentile(pooled, probability),
        lower=percentile(estimates, tail),
        upper=percentile(estimates, 1 - tail),
        sample_size=len(pooled),
    )


@dataclass(frozen=True, slots=True)
class OperationalSummary:
    """What a variant's runs cost and how long they took. A value is None when no run reported it."""

    variant_id: str
    runs: int
    failed_runs: int
    latency_ms: Mapping[str, PercentileEstimate] | None
    cost_usd: Mapping[str, PercentileEstimate] | None
    mean_cost_per_task_usd: float | None
    mean_step_count: float | None


_PERCENTILES = {"p50": 0.50, "p95": 0.95, "p99": 0.99}


def operational_summary(
    run: RepeatedRun, variant_id: str, *, resamples: int = 2_000, confidence: float = 0.95, seed: int = 0
) -> OperationalSummary:
    """p50, p95 and p99 of latency and cost per run, mean cost and steps, over a variant's successful runs."""

    rows = run.rows_for(variant_id)
    if not rows:
        raise ValueError(f"the run has no rows for variant {variant_id!r}")
    succeeded = [row for row in rows if row.result.error is None]
    latency: dict[str, list[float]] = {}
    cost: dict[str, list[float]] = {}
    for row in succeeded:
        if row.result.latency_ms is not None:
            latency.setdefault(row.case_id, []).append(row.result.latency_ms)
        if row.result.cost_usd is not None:
            cost.setdefault(row.case_id, []).append(row.result.cost_usd)

    def estimates(groups: dict[str, list[float]]) -> dict[str, PercentileEstimate] | None:
        if not groups:
            return None
        return {
            name: clustered_percentile(groups, probability, resamples=resamples, confidence=confidence, seed=seed)
            for name, probability in _PERCENTILES.items()
        }

    costs = [value for values in cost.values() for value in values]
    steps = [len(row.tool_calls) for row in succeeded]
    return OperationalSummary(
        variant_id=variant_id,
        runs=len(rows),
        failed_runs=len(rows) - len(succeeded),
        latency_ms=estimates(latency),
        cost_usd=estimates(cost),
        mean_cost_per_task_usd=sum(costs) / len(costs) if costs else None,
        mean_step_count=sum(steps) / len(steps) if steps else None,
    )


RegressionStatus = Literal["ok", "alert", "block"]


def regression_status(baseline: float, candidate: float, *, alert: float, block: float) -> RegressionStatus:
    """How a candidate's mean cost (or latency) compares with the baseline's, as a relative increase.

    Above `block` it blocks, above `alert` it alerts. When the baseline is zero any increase counts as +100%,
    the same convention the gate uses.
    """

    if not 0 <= alert <= block:
        raise ValueError("alert and block must satisfy 0 <= alert <= block")
    if baseline <= 0:
        relative = 1.0 if candidate > baseline else 0.0
    else:
        relative = (candidate - baseline) / baseline
    if _exceeds(relative, block):
        return "block"
    if _exceeds(relative, alert):
        return "alert"
    return "ok"


def _exceeds(value: float, limit: float) -> bool:
    # (1.10 - 1.0) / 1.0 is 0.10000000000000009: a value at the limit must not count as over it.
    return value > limit and not math.isclose(value, limit, rel_tol=1e-9, abs_tol=1e-12)


def _steps(output: Any) -> Sequence[GenericStep]:
    return trajectory_of(output).steps


@dataclass(frozen=True, slots=True)
class StepCount:
    """Scores how many steps a run took, and how many of them failed."""

    name: str = "step_count"

    def __call__(self, case: EvaluationCase, output: Any) -> dict[str, float]:
        steps = _steps(output)
        failed = sum(1 for step in steps if step.error is not None or step.failure is not None)
        return {self.name: float(len(steps)), f"failed_{self.name}": float(failed)}


def _optimal_steps_from_case(case: EvaluationCase) -> int:
    """The default reference: `case.expected["optimal_steps"]`."""

    if not isinstance(case.expected, Mapping) or "optimal_steps" not in case.expected:
        raise ValueError(f"case {case.case_id} has no expected['optimal_steps'] to compare the run with")
    return int(case.expected["optimal_steps"])


@dataclass(frozen=True, slots=True)
class ExecutionEfficiency:
    """Optimal steps over the steps taken, at most 1.0: how much of the work was not wasted.

    A run that beats the stated optimum scores 1.0; that is not proof it was right, which is why this
    is read together with the step band and with outcome scores.
    """

    optimal_steps: Callable[[EvaluationCase], int] = _optimal_steps_from_case
    name: str = "execution_efficiency"

    def __call__(self, case: EvaluationCase, output: Any) -> dict[str, float]:
        optimal = self.optimal_steps(case)
        if optimal < 0:
            raise ValueError(f"case {case.case_id} has a negative optimal step count")
        actual = len(_steps(output))
        if actual == 0:
            return {self.name: 1.0 if optimal == 0 else 0.0}
        return {self.name: min(1.0, optimal / actual)}


@dataclass(frozen=True, slots=True)
class StepBand:
    """The step counts normal for one kind of task, inclusive at both ends."""

    low: int
    high: int

    def __post_init__(self) -> None:
        if self.low < 0 or self.high < self.low:
            raise ValueError("a step band needs 0 <= low <= high")

    def position(self, steps: int) -> Literal["under", "within", "over"]:
        if steps < self.low:
            return "under"
        return "over" if steps > self.high else "within"


POOLED_BAND = "*"


def _row_category(row: RunRow) -> str:
    return row.category or POOLED_BAND


def step_band_from_baseline(
    run: RepeatedRun,
    variant_id: str,
    *,
    group_of: Callable[[RunRow], str] = _row_category,
    low_quantile: float = 0.10,
    high_quantile: float = 0.90,
    minimum_runs: int = 5,
) -> dict[str, StepBand]:
    """The expected step band per kind of task, from a baseline's successful runs.

    A group with fewer than `minimum_runs` runs has too little to say and is left out; the pooled band
    over every run, under the key `"*"`, covers it.
    """

    groups: dict[str, list[int]] = {}
    for row in run.rows_for(variant_id):
        if row.result.error is None:
            groups.setdefault(group_of(row), []).append(len(row.tool_calls))
    pooled = [count for counts in groups.values() for count in counts]
    if not pooled:
        raise ValueError(f"the run has no successful rows for variant {variant_id!r}")

    def band(counts: list[int]) -> StepBand:
        return StepBand(
            math.floor(percentile(counts, low_quantile)),
            math.ceil(percentile(counts, high_quantile)),
        )

    bands = {group: band(counts) for group, counts in groups.items() if len(counts) >= minimum_runs}
    bands[POOLED_BAND] = band(pooled)
    return bands


def _case_category(case: EvaluationCase) -> str:
    return str(case.inputs.get("category", POOLED_BAND))


@dataclass(frozen=True, slots=True)
class WithinStepBand:
    """1.0 when the run's step count is inside the band for its kind of task; 0.0 when under or over it.

    A task kind with no band of its own uses the pooled band.
    """

    bands: Mapping[str, StepBand]
    group_of: Callable[[EvaluationCase], str] = _case_category
    name: str = "within_step_band"

    def __call__(self, case: EvaluationCase, output: Any) -> dict[str, float]:
        band = self.bands.get(self.group_of(case)) or self.bands[POOLED_BAND]
        return {self.name: 1.0 if band.position(len(_steps(output))) == "within" else 0.0}


def _fingerprint(tool: str, arguments: Mapping[str, Any]) -> str:
    encoded = json.dumps([tool.strip().lower(), arguments], sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def action_fingerprint(step: GenericStep) -> str:
    """A hash of what an action was: the tool and its arguments."""

    return _fingerprint(step.tool, step.args)


@dataclass(frozen=True, slots=True)
class Loop:
    """A block of `window` actions that was repeated `repeats` times back to back, starting at `start_index`."""

    start_index: int
    window: int
    repeats: int


def _repeats_at(hashes: Sequence[str], start: int, window: int, repeats: int) -> bool:
    block = hashes[start : start + window]
    return all(hashes[start + r * window : start + (r + 1) * window] == block for r in range(1, repeats))


def find_loop(steps: Sequence[GenericStep], *, max_window: int = 3, repeats: int = 3) -> Loop | None:
    """The earliest place the agent repeated itself: the same action, or the same short cycle of actions,
    `repeats` times back to back. Actions are compared by tool and arguments, so the same tool with
    different arguments is progress, not a loop."""

    if max_window < 1 or repeats < 2:
        raise ValueError("max_window must be at least 1 and repeats at least 2")
    hashes = [action_fingerprint(step) for step in steps]
    for start in range(len(hashes)):
        for window in range(1, max_window + 1):
            if start + window * repeats <= len(hashes) and _repeats_at(hashes, start, window, repeats):
                return Loop(start, window, repeats)
    return None


@dataclass(slots=True)
class LoopGuard:
    """Feed it each action as it happens; it says when the agent has just repeated itself.

    A runner can call `record` and stop early when it returns a `Loop`. The stop itself belongs in the
    agent; this only detects. The guard sees the same repeats `find_loop` does.
    """

    max_window: int = 3
    repeats: int = 3
    _hashes: list[str] = field(default_factory=list)

    def record(self, tool: str, arguments: Mapping[str, Any]) -> Loop | None:
        """Note one action; return the loop it completes, or None."""

        self._hashes.append(_fingerprint(tool, arguments))
        for window in range(1, self.max_window + 1):
            start = len(self._hashes) - window * self.repeats
            if start >= 0 and _repeats_at(self._hashes, start, window, self.repeats):
                return Loop(start, window, self.repeats)
        return None


@dataclass(frozen=True, slots=True)
class NoLoop:
    """1.0 when the run never repeated itself (see `find_loop`); 0.0 when it did."""

    max_window: int = 3
    repeats: int = 3
    name: str = "no_loop"

    def __call__(self, case: EvaluationCase, output: Any) -> dict[str, float]:
        loop = find_loop(_steps(output), max_window=self.max_window, repeats=self.repeats)
        return {self.name: 0.0 if loop else 1.0}


__all__ = [
    "ExecutionEfficiency",
    "Loop",
    "LoopGuard",
    "NoLoop",
    "OperationalSummary",
    "POOLED_BAND",
    "PercentileEstimate",
    "RegressionStatus",
    "StepBand",
    "StepCount",
    "WithinStepBand",
    "action_fingerprint",
    "clustered_percentile",
    "find_loop",
    "operational_summary",
    "regression_status",
    "step_band_from_baseline",
]
