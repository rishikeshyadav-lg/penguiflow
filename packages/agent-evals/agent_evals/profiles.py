"""Threshold profiles: the bars a run is held to in CI and in production.

A profile is a set of bars for the scorecard: a success floor, a tail-latency bound, and how much a mean
cost may rise before it alerts or blocks. Two kinds ship, and they are labelled differently on purpose.

`example` profiles hold starting numbers from the article this framework was checked against. The article
calls them its own proposals, not measured standards; use them to see the shape, then calibrate.
`derived` profiles come from a calibration of the agent being judged (`propose_thresholds`) and are the ones
to hold a real decision to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .operational import OperationalSummary, RegressionStatus, regression_status
from .thresholds import PromotionThresholds

ProfileOrigin = Literal["example", "derived"]
CheckStatus = Literal["pass", "fail", "alert", "not_measured"]
_COST_STATUS: dict[RegressionStatus, CheckStatus] = {"ok": "pass", "alert": "alert", "block": "fail"}


@dataclass(frozen=True, slots=True)
class ThresholdProfile:
    """The bars for one setting (CI gate, production SLO). A bar left None is not checked."""

    name: str
    origin: ProfileOrigin
    note: str
    minimum_success_rate: float | None = None
    maximum_p95_latency_ms: float | None = None
    cost_alert: float | None = None
    cost_block: float | None = None

    def __post_init__(self) -> None:
        if self.cost_alert is not None and self.cost_block is None:
            raise ValueError("cost_alert needs cost_block")
        if self.cost_alert is not None and self.cost_block is not None and self.cost_alert > self.cost_block:
            raise ValueError("cost_alert must not exceed cost_block")


_ARTICLE_NOTE = (
    "Starting numbers from the evaluation article; its author calls them a proposal, not a measured standard. "
    "Calibrate before relying on them."
)
EXAMPLE_PROFILES = {
    "ci_gate": ThresholdProfile(
        "ci_gate", "example", _ARTICLE_NOTE, minimum_success_rate=0.85, maximum_p95_latency_ms=4_000.0,
        cost_alert=0.10, cost_block=0.15,
    ),
    "production_slo": ThresholdProfile(
        "production_slo", "example", _ARTICLE_NOTE, minimum_success_rate=0.90, maximum_p95_latency_ms=3_000.0,
        cost_alert=0.10, cost_block=0.15,
    ),
}  # fmt: skip


def derived_profile(
    name: str, thresholds: PromotionThresholds, *, baseline: OperationalSummary | None = None
) -> ThresholdProfile:
    """A profile from a calibration: the candidate floor as the success floor, the owner's cost cap as the block
    level, and, when the baseline's p95 latency is known, that p95 plus the owner's latency cap as the bound."""

    p95_bound = None
    if baseline is not None and baseline.latency_ms is not None:
        p95_bound = baseline.latency_ms["p95"].estimate * (1 + thresholds.maximum_latency_regression)
    return ThresholdProfile(
        name=name,
        origin="derived",
        note="Derived from a calibration of this agent's own baseline runs.",
        minimum_success_rate=thresholds.minimum_candidate_correct_rate,
        maximum_p95_latency_ms=p95_bound,
        cost_block=thresholds.maximum_cost_regression,
    )


@dataclass(frozen=True, slots=True)
class ProfileCheck:
    """One bar and how the run stood against it."""

    name: str
    status: CheckStatus
    detail: str


@dataclass(frozen=True, slots=True)
class ProfileVerdict:
    """A run held to one profile. `passed` is None when no bar could be checked."""

    profile: str
    origin: ProfileOrigin
    note: str
    checks: tuple[ProfileCheck, ...]

    @property
    def passed(self) -> bool | None:
        judged = [check for check in self.checks if check.status != "not_measured"]
        if not judged:
            return None
        return all(check.status in ("pass", "alert") for check in judged)


def evaluate_profile(
    profile: ThresholdProfile,
    *,
    success_rate: float | None,
    summary: OperationalSummary,
    baseline: OperationalSummary | None = None,
) -> ProfileVerdict:
    """Hold a variant's success rate, p95 latency and mean cost to a profile.

    Cost is judged against the baseline's mean cost per task, so without a baseline it is not measured.
    """

    checks: list[ProfileCheck] = []
    if profile.minimum_success_rate is not None:
        if success_rate is None:
            checks.append(ProfileCheck("success_rate", "not_measured", "no success rate was measured"))
        else:
            ok = success_rate >= profile.minimum_success_rate
            checks.append(
                ProfileCheck(
                    "success_rate",
                    "pass" if ok else "fail",
                    f"{success_rate:.3f} against a floor of {profile.minimum_success_rate:.3f}",
                )
            )
    if profile.maximum_p95_latency_ms is not None:
        if summary.latency_ms is None:
            checks.append(ProfileCheck("p95_latency", "not_measured", "no latency was measured"))
        else:
            p95 = summary.latency_ms["p95"].estimate
            checks.append(
                ProfileCheck(
                    "p95_latency", "pass" if p95 <= profile.maximum_p95_latency_ms else "fail",
                    f"{p95:.0f} ms against a bound of {profile.maximum_p95_latency_ms:.0f} ms",
                )
            )  # fmt: skip
    if profile.cost_block is not None:
        candidate_cost = summary.mean_cost_per_task_usd
        baseline_cost = baseline.mean_cost_per_task_usd if baseline else None
        if candidate_cost is None or baseline_cost is None:
            checks.append(ProfileCheck("cost", "not_measured", "needs a cost for both the baseline and this run"))
        else:
            alert = profile.cost_alert if profile.cost_alert is not None else profile.cost_block
            status = regression_status(baseline_cost, candidate_cost, alert=alert, block=profile.cost_block)
            checks.append(
                ProfileCheck(
                    "cost",
                    _COST_STATUS[status],
                    f"{candidate_cost:.4f} against a baseline of {baseline_cost:.4f} per task ({status})",
                )
            )
    return ProfileVerdict(profile.name, profile.origin, profile.note, tuple(checks))


__all__ = [
    "CheckStatus",
    "EXAMPLE_PROFILES",
    "ProfileCheck",
    "ProfileOrigin",
    "ProfileVerdict",
    "ThresholdProfile",
    "derived_profile",
    "evaluate_profile",
]
