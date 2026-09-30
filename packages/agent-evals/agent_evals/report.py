"""A report says what was measured and what was not: one metric per layer, and every gap shown.

The scorecard has eight entries (task success, tool selection, argument correctness, plan adherence,
execution efficiency, cost per task, p95 latency, policy violations), one for each layer of the model plus the
policy constraint that cuts across them. A metric that could not be computed is still listed, as "not
measured" with the reason; it is never left out. Failures and exclusions are listed too, so a number is
never presented without the runs that did not make it in.

A report holds ids, numbers and codes, never case text. Error messages can contain text, so only the error
type is shown unless the caller asks for messages.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from .datasets import DatasetManifest
from .execution import RepeatedRun
from .operational import OperationalSummary, operational_summary
from .policy import policy_flag
from .profiles import ProfileVerdict
from .statistics import PairedPrompt, paired_bootstrap
from .suites import SuiteRule, SuiteVerdict, case_scores, suite_verdict
from .trajectory import selection_gap

REPORT_SCHEMA_VERSION = "agent-evals.report.v1"


@dataclass(frozen=True, slots=True)
class ScorecardMetrics:
    """Which metric name each scorecard entry reads. None means no scorer is configured for it."""

    success: str = "success"
    tool_selection: str | None = "tool_selection_accuracy"
    argument_correctness: str | None = "argument_correctness"
    plan_adherence: str | None = None
    execution_efficiency: str | None = "execution_efficiency"
    policy: str | None = "policy_compliance"


@dataclass(frozen=True, slots=True)
class RunRecord:
    """What produced a report: the run's settings, bundle, dataset and metric versions, and whether it counts."""

    run_id: str
    variant_id: str
    dataset_id: str
    dataset_version: str
    dataset_digest: str
    suite: str
    metric_versions: Mapping[str, str]
    settings: Mapping[str, Any]
    bundle: Mapping[str, Any]
    verdict_grade: bool
    created_at: str

    @classmethod
    def from_run(
        cls,
        run: RepeatedRun,
        manifest: DatasetManifest,
        variant_id: str,
        *,
        run_id: str,
        bundle: Mapping[str, Any] | None = None,
        metric_versions: Mapping[str, str] | None = None,
        verdict_grade: bool = False,
        created_at: str | None = None,
    ) -> RunRecord:
        """Describe a run. `verdict_grade` says whether it ran on the agent as deployed and may decide something."""

        return cls(
            run_id=run_id,
            variant_id=variant_id,
            dataset_id=manifest.dataset_id,
            dataset_version=manifest.version,
            dataset_digest=manifest.digest,
            suite=manifest.suite,
            metric_versions=dict(metric_versions or {}),
            settings=asdict(run.settings),
            bundle=dict(bundle or {}),
            verdict_grade=verdict_grade,
            created_at=created_at or datetime.now(UTC).isoformat(),
        )


@dataclass(frozen=True, slots=True)
class ScorecardEntry:
    """One of the eight metrics, measured or not, with its range and the runs behind it."""

    key: str
    label: str
    layer: str
    unit: str
    measured: bool
    value: float | None
    lower: float | None
    upper: float | None
    runs_used: int
    runs_excluded: int
    reason: str | None = None
    flag: bool | None = None


@dataclass(frozen=True, slots=True)
class FailedRun:
    """A run that did not complete: which case and repeat, and what kind of error."""

    case_id: str
    repeat: int
    error: str


@dataclass(frozen=True, slots=True)
class Scorecard:
    entries: Sequence[ScorecardEntry]
    audit: Sequence[str]
    failures: Sequence[FailedRun]

    def entry(self, key: str) -> ScorecardEntry:
        for entry in self.entries:
            if entry.key == key:
                return entry
        raise ValueError(f"unknown scorecard entry: {key}")


def _interval(case_values: Mapping[str, float], resamples: int, seed: int) -> tuple[float | None, float | None]:
    interval = paired_bootstrap(
        [PairedPrompt(case_id, 0.0, value) for case_id, value in case_values.items()], resamples=resamples, seed=seed
    )
    return (None, None) if interval is None else (interval.lower, interval.upper)


def _case_means(run: RepeatedRun, variant_id: str, metric: str) -> tuple[dict[str, float], int, int]:
    """Each case's mean of a metric over the runs that produced it, and how many runs were used and left out."""

    values: dict[str, list[float]] = {}
    used = excluded = 0
    for row in run.rows_for(variant_id):
        if row.result.error is None and metric in row.result.metrics:
            values.setdefault(row.case_id, []).append(row.result.metrics[metric])
            used += 1
        else:
            excluded += 1
    return {case_id: sum(v) / len(v) for case_id, v in values.items()}, used, excluded


def _not_measured(key: str, label: str, layer: str, unit: str, reason: str, excluded: int = 0) -> ScorecardEntry:
    return ScorecardEntry(key, label, layer, unit, False, None, None, None, 0, excluded, reason)


def _metric_entry(
    run: RepeatedRun, variant_id: str, key: str, label: str, layer: str, metric: str | None, *,
    resamples: int, seed: int, missing_reason: str,
) -> ScorecardEntry:  # fmt: skip
    if metric is None:
        return _not_measured(key, label, layer, "score", missing_reason)
    means, used, excluded = _case_means(run, variant_id, metric)
    if not means:
        return _not_measured(key, label, layer, "score", f"no scorer produced {metric!r}", excluded)
    lower, upper = _interval(means, resamples, seed)
    return ScorecardEntry(
        key, label, layer, "score", True, sum(means.values()) / len(means), lower, upper, used, excluded
    )


def build_scorecard(
    run: RepeatedRun,
    variant_id: str,
    *,
    metrics: ScorecardMetrics = ScorecardMetrics(),
    summary: OperationalSummary | None = None,
    gap_flag_above: float = 0.2,
    resamples: int = 2_000,
    seed: int = 0,
    include_error_messages: bool = False,
) -> Scorecard:
    """Build the eight-entry scorecard for one variant of a run."""

    rows = run.rows_for(variant_id)
    if not rows:
        raise ValueError(f"the run has no rows for variant {variant_id!r}")
    summary = summary or operational_summary(run, variant_id, resamples=resamples, seed=seed)

    success_scores = case_scores(run, variant_id, metrics.success)
    lower, upper = _interval(success_scores, resamples, seed)
    entries = [
        ScorecardEntry(
            "success_rate", "Task success rate", "outcome", "rate", True,
            sum(success_scores.values()) / len(success_scores), lower, upper,
            sum(1 for row in rows if row.result.error is None), sum(1 for row in rows if row.result.error is not None),
        ),
        _metric_entry(
            run, variant_id, "tool_selection", "Tool selection accuracy", "trajectory", metrics.tool_selection,
            resamples=resamples, seed=seed, missing_reason="no tool-selection scorer is configured",
        ),
        _metric_entry(
            run, variant_id, "argument_correctness", "Argument correctness", "trajectory", metrics.argument_correctness,
            resamples=resamples, seed=seed, missing_reason="no argument scorer is configured",
        ),
        _metric_entry(
            run, variant_id, "plan_adherence", "Plan adherence", "trajectory", metrics.plan_adherence,
            resamples=resamples, seed=seed, missing_reason="not configured: plan adherence needs a judge",
        ),
        _metric_entry(
            run, variant_id, "execution_efficiency", "Execution efficiency", "operational", metrics.execution_efficiency,
            resamples=resamples, seed=seed, missing_reason="no efficiency scorer is configured",
        ),
    ]  # fmt: skip

    cost_means: dict[str, list[float]] = {}
    for row in rows:
        if row.result.error is None and row.result.cost_usd is not None:
            cost_means.setdefault(row.case_id, []).append(row.result.cost_usd)
    if cost_means:
        per_case = {case_id: sum(v) / len(v) for case_id, v in cost_means.items()}
        cost_lower, cost_upper = _interval(per_case, resamples, seed)
        used = sum(len(v) for v in cost_means.values())
        entries.append(
            ScorecardEntry(
                "cost_per_task", "Cost per task", "operational", "usd", True, summary.mean_cost_per_task_usd,
                cost_lower, cost_upper, used, len(rows) - used,
            )
        )  # fmt: skip
    else:
        entries.append(
            _not_measured("cost_per_task", "Cost per task", "operational", "usd", "the runner reported no cost")
        )

    if summary.latency_ms is not None:
        p95 = summary.latency_ms["p95"]
        entries.append(
            ScorecardEntry(
                "p95_latency", "p95 latency", "operational", "ms", True, p95.estimate, p95.lower, p95.upper,
                p95.sample_size, len(rows) - p95.sample_size,
            )
        )  # fmt: skip
    else:
        entries.append(_not_measured("p95_latency", "p95 latency", "operational", "ms", "no latency was measured"))

    if metrics.policy is not None and any(metrics.policy in row.result.metrics for row in rows):
        flag = policy_flag(run, variant_id, metric=metrics.policy)
        entries.append(
            ScorecardEntry(
                "policy_violation", "Policy violations", "policy", "share of runs", True,
                flag.violating_runs / flag.runs if flag.runs else 0.0, None, None, flag.runs, len(rows) - flag.runs,
                flag=flag.raised,
                reason=("cases: " + ", ".join(flag.case_ids)) if flag.case_ids else None,
            )
        )  # fmt: skip
    else:
        entries.append(
            _not_measured(
                "policy_violation", "Policy violations", "policy", "share of runs", "no policy check is configured"
            )
        )

    audit = []
    if metrics.tool_selection and entries[1].measured:
        gap = selection_gap(
            run, variant_id, outcome_metric=metrics.success, selection_metric=metrics.tool_selection,
            flag_above=gap_flag_above,
        )  # fmt: skip
        audit.append(gap.note)

    failures = [
        FailedRun(
            row.case_id, row.repeat, row.result.error if include_error_messages else row.result.error.split(":")[0]
        )
        for row in rows
        if row.result.error is not None
    ]
    return Scorecard(entries=tuple(entries), audit=tuple(audit), failures=tuple(failures))


@dataclass(frozen=True, slots=True)
class Report:
    """Everything one run's report says."""

    record: RunRecord
    suite: SuiteVerdict
    scorecard: Scorecard
    operational: OperationalSummary
    profiles: Sequence[ProfileVerdict] = field(default_factory=tuple)


def build_report(
    run: RepeatedRun,
    manifest: DatasetManifest,
    variant_id: str,
    *,
    record: RunRecord,
    metrics: ScorecardMetrics = ScorecardMetrics(),
    rule: SuiteRule | None = None,
    profiles: Sequence[ProfileVerdict] = (),
    resamples: int = 2_000,
    seed: int = 0,
    include_error_messages: bool = False,
) -> Report:
    """Assemble a report: the suite's verdict, the scorecard, and the operational summary."""

    summary = operational_summary(run, variant_id, resamples=resamples, seed=seed)
    return Report(
        record=record,
        suite=suite_verdict(run, variant_id, manifest, rule or SuiteRule(metrics.success, resamples=resamples)),
        scorecard=build_scorecard(
            run, variant_id, metrics=metrics, summary=summary, resamples=resamples, seed=seed,
            include_error_messages=include_error_messages,
        ),
        operational=summary,
        profiles=tuple(profiles),
    )  # fmt: skip


def report_json(report: Report) -> dict[str, Any]:
    """The report as plain JSON data. Every report has the same keys, whatever was measured."""

    suite = report.suite
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run": asdict(report.record),
        "suite": {
            "suite": suite.suite,
            "metric": suite.metric,
            "rule": suite.rule,
            "case_count": suite.case_count,
            "mean_score": suite.mean_score,
            "passed": suite.passed,
            "pass_rate": suite.pass_rate,
            "failing_case_ids": list(suite.failing_case_ids),
            "interval": None if suite.interval is None else [suite.interval.lower, suite.interval.upper],
        },
        "scorecard": {
            "entries": [asdict(entry) for entry in report.scorecard.entries],
            "audit": list(report.scorecard.audit),
            "failures": [asdict(failure) for failure in report.scorecard.failures],
        },
        "operational": {
            "runs": report.operational.runs,
            "failed_runs": report.operational.failed_runs,
            "mean_cost_per_task_usd": report.operational.mean_cost_per_task_usd,
            "mean_step_count": report.operational.mean_step_count,
        },
        "profiles": [
            {
                "profile": verdict.profile,
                "origin": verdict.origin,
                "note": verdict.note,
                "passed": verdict.passed,
                "checks": [asdict(check) for check in verdict.checks],
            }
            for verdict in report.profiles
        ],
    }


def _number(value: float | None, unit: str) -> str:
    if value is None:
        return "-"
    return f"{value:.0f}" if unit == "ms" else f"{value:.3f}" if unit != "usd" else f"{value:.4f}"


def report_markdown(report: Report) -> str:
    """The report as Markdown. A metric that was not measured is shown as such, with its reason."""

    record, suite = report.record, report.suite
    lines = [
        f"# Evaluation report: {record.variant_id}",
        "",
        f"- run `{record.run_id}` created {record.created_at}",
        f"- dataset `{record.dataset_id}` {record.dataset_version} (`{record.dataset_digest}`), suite: {record.suite}",
        f"- verdict grade: {'yes' if record.verdict_grade else 'no'}",
        "",
        "## Suite",
        "",
        f"Rule applied: {suite.rule}",
        "",
    ]
    if suite.passed is not None:
        lines.append(
            f"Result: {'passed' if suite.passed else 'FAILED'} "
            f"({suite.pass_rate:.1%} of {suite.case_count} cases passed"
            + (f"; failing: {', '.join(suite.failing_case_ids)}" if suite.failing_case_ids else "")
            + ")"
        )
    else:
        assert suite.interval is not None
        lines.append(
            f"Result: mean {suite.mean_score:.3f} over {suite.case_count} cases, "
            f"range {suite.interval.lower:.3f} to {suite.interval.upper:.3f}; no pass or fail for a capability suite."
        )
    lines += [
        "",
        "## Scorecard",
        "",
        "| Metric | Layer | Value | 95% range | Runs | Status |",
        "|---|---|---|---|---|---|",
    ]
    for entry in report.scorecard.entries:
        if entry.measured:
            span = (
                "-"
                if entry.lower is None
                else f"{_number(entry.lower, entry.unit)} to {_number(entry.upper, entry.unit)}"
            )
            status = "FLAG RAISED" if entry.flag else "measured"
            if entry.reason and entry.measured:
                status += f" ({entry.reason})"
            lines.append(
                f"| {entry.label} | {entry.layer} | {_number(entry.value, entry.unit)} {entry.unit} | {span} "
                f"| {entry.runs_used} used, {entry.runs_excluded} excluded | {status} |"
            )
        else:
            lines.append(f"| {entry.label} | {entry.layer} | not measured | - | - | {entry.reason} |")
    if report.scorecard.audit:
        lines += ["", "## Audit signals", ""] + [f"- {line}" for line in report.scorecard.audit]
    lines += ["", "## Failed runs", ""]
    if report.scorecard.failures:
        lines += [f"- {f.case_id} repeat {f.repeat}: {f.error}" for f in report.scorecard.failures]
    else:
        lines.append("None.")
    for verdict in report.profiles:
        state = {True: "passed", False: "FAILED", None: "not measured"}[verdict.passed]
        lines += ["", f"## Profile: {verdict.profile} ({verdict.origin}): {state}", "", verdict.note, ""]
        lines += [f"- {check.name}: {check.status}, {check.detail}" for check in verdict.checks]
    return "\n".join(lines) + "\n"


__all__ = [
    "FailedRun",
    "REPORT_SCHEMA_VERSION",
    "Report",
    "RunRecord",
    "Scorecard",
    "ScorecardEntry",
    "ScorecardMetrics",
    "build_report",
    "build_scorecard",
    "report_json",
    "report_markdown",
]
