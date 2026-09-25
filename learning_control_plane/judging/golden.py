"""Re-judge stored runs against their golden labels, so a judge change cannot regress silently.

A golden label says what a correct judge concludes about one stored run. `run_golden_set` re-judges
every labelled run and compares the outcome with the label. Some disagreements are known and
accepted (the label is right, the judge is not yet); they are listed as `AcceptedMiss`es, each with
the outcome the judge is known to give. The set passes when every other label agrees:

- a new miss (a disagreement not accepted, or an accepted miss whose judged outcome changed) fails;
- an accepted miss that now agrees is reported as fixed, so it can be removed from the list.

Run it before any paid replay and after every judge change. The labels and stored runs usually hold
customer text, so they live outside the repository and are passed in as paths:

    python -m learning_control_plane.judging.golden --labels labels.json --runs runs.json \\
        --judge my_package.judging:build_judge [--reference my_package.judging:build_reference] \\
        [--accepted-misses misses.json] [--output results.json]

`--runs` maps each label's `case_key` to a stored run: `question`, `steps` (each with `tool`,
`args`, `result`, `error`), `final_answer`, and optional `rendered` and `classification`. The
`--judge` factory returns a callable `(run, reference) -> InvestigationVerification`; the optional
`--reference` factory returns an async callable `run -> reference`. The command exits non-zero on
a new miss.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import sys
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..evaluation.verification import JUDGE_OUTCOMES, InvestigationVerification, JudgeOutcome
from .outcomes import outcome_of, outcome_reason_codes
from .runs import AgentRun, AgentStep, RenderedOutput

logger = logging.getLogger("learning_control_plane.judging.golden")

GoldenJudge = Callable[[AgentRun, Any], InvestigationVerification]
GoldenReference = Callable[[AgentRun], Awaitable[Any]]


@dataclass(frozen=True, slots=True)
class GoldenLabel:
    """What a correct judge concludes about one stored run."""

    case_key: str
    expected_outcome: JudgeOutcome
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.expected_outcome not in JUDGE_OUTCOMES:
            raise ValueError(f"unsupported expected outcome for {self.case_key!r}: {self.expected_outcome!r}")


@dataclass(frozen=True, slots=True)
class AcceptedMiss:
    """A known disagreement: the label is right, and the judge is known to conclude `judged_outcome`."""

    case_key: str
    judged_outcome: str
    reason: str = ""


@dataclass(frozen=True, slots=True)
class GoldenCaseResult:
    """One label's comparison: the expected and judged outcomes and the judge's reason codes."""

    case_key: str
    expected_outcome: str
    judged_outcome: str
    reason_codes: tuple[str, ...] = ()
    reference_error: str | None = None

    @property
    def agrees(self) -> bool:
        """Return whether the judge reached the labelled outcome."""

        return self.judged_outcome == self.expected_outcome


@dataclass(frozen=True, slots=True)
class GoldenReport:
    """How a judge did on the golden set, and whether it may be trusted with paid runs."""

    results: tuple[GoldenCaseResult, ...]
    accepted_misses: tuple[AcceptedMiss, ...] = ()
    new_misses: tuple[GoldenCaseResult, ...] = field(init=False)
    fixed_misses: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        accepted = {miss.case_key: miss for miss in self.accepted_misses}
        new_misses = tuple(
            result
            for result in self.results
            if not result.agrees
            and (result.case_key not in accepted or accepted[result.case_key].judged_outcome != result.judged_outcome)
        )
        fixed = tuple(result.case_key for result in self.results if result.agrees and result.case_key in accepted)
        object.__setattr__(self, "new_misses", new_misses)
        object.__setattr__(self, "fixed_misses", fixed)

    @property
    def agreement(self) -> int:
        """Return how many labels the judge agreed with."""

        return sum(result.agrees for result in self.results)

    @property
    def disagreements(self) -> tuple[GoldenCaseResult, ...]:
        """Return every label the judge disagreed with, accepted or not."""

        return tuple(result for result in self.results if not result.agrees)

    @property
    def passed(self) -> bool:
        """Return whether every disagreement is an accepted miss with its known outcome."""

        return bool(self.results) and not self.new_misses

    def summary(self) -> str:
        """Return a short human-readable report."""

        lines = [f"agreement: {self.agreement} of {len(self.results)}"]
        by_expected = Counter(result.expected_outcome for result in self.results)
        lines.append(f"by expected outcome: {dict(by_expected)}")
        accepted = {miss.case_key for miss in self.accepted_misses}
        for result in self.disagreements:
            kind = "ACCEPTED" if result.case_key in accepted and result not in self.new_misses else "NEW MISS"
            lines.append(
                f"  {kind} {result.case_key}: expected {result.expected_outcome}, "
                f"judge {result.judged_outcome} {list(result.reason_codes[:4])}"
            )
        for case_key in self.fixed_misses:
            lines.append(f"  FIXED {case_key}: now agrees; remove it from the accepted misses")
        lines.append("golden set: " + ("passed" if self.passed else "FAILED"))
        return "\n".join(lines)


async def run_golden_set(
    labels: Iterable[GoldenLabel],
    load_run: Callable[[GoldenLabel], AgentRun | None],
    judge: GoldenJudge,
    *,
    reference: GoldenReference | None = None,
    accepted_misses: Sequence[AcceptedMiss] = (),
) -> GoldenReport:
    """Re-judge each labelled run and compare its outcome with the label.

    A label whose run cannot be loaded is judged `missing_run`, which never agrees. A reference that
    fails is recorded on the result and the run is judged without it, as the live judge would be.
    The judge runs in a worker thread, since it may call a model.
    """

    results: list[GoldenCaseResult] = []
    for label in labels:
        run = load_run(label)
        if run is None:
            results.append(GoldenCaseResult(label.case_key, label.expected_outcome, "missing_run"))
            continue
        run_reference = None
        reference_error = None
        if reference is not None:
            try:
                run_reference = await reference(run)
            except Exception as error:  # noqa: BLE001 -- judged without it, as the live judge would be
                reference_error = f"{type(error).__name__}: {error}"
                logger.warning("golden reference failed for %s: %s", label.case_key, reference_error)
        verification = await asyncio.to_thread(judge, run, run_reference)
        results.append(
            GoldenCaseResult(
                case_key=label.case_key,
                expected_outcome=label.expected_outcome,
                judged_outcome=outcome_of(verification),
                reason_codes=tuple(outcome_reason_codes(verification)),
                reference_error=reference_error,
            )
        )
    return GoldenReport(results=tuple(results), accepted_misses=tuple(accepted_misses))


def run_from_record(record: Mapping[str, Any]) -> AgentRun:
    """Rebuild a stored run from its JSON record."""

    return AgentRun(
        question=str(record.get("question") or ""),
        steps=[
            AgentStep(
                tool=str(step["tool"]),
                args=dict(step.get("args") or {}),
                result=step.get("result"),
                error=step.get("error"),
            )
            for step in record.get("steps") or ()
        ],
        final_answer=record.get("final_answer"),
        rendered=[RenderedOutput(output["kind"], output["content"]) for output in record.get("rendered") or ()],
        classification=dict(record.get("classification") or {}),
        finish_reason=str(record.get("finish_reason") or "answer_complete"),
    )


def _load_factory(reference: str) -> Any:
    module_name, _, attribute = reference.partition(":")
    if not module_name or not attribute:
        raise ValueError(f"expected module:factory, got {reference!r}")
    return getattr(importlib.import_module(module_name), attribute)()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the golden set from the command line; return 1 on a new miss."""

    parser = argparse.ArgumentParser(description="Re-judge stored runs against their golden labels.")
    parser.add_argument("--labels", type=Path, required=True, help="JSON list of {case_key, expected_outcome}")
    parser.add_argument("--runs", type=Path, required=True, help="JSON object: case_key -> stored run")
    parser.add_argument("--judge", required=True, help="module:factory returning (run, reference) -> verification")
    parser.add_argument("--reference", help="module:factory returning an async run -> reference")
    parser.add_argument("--accepted-misses", type=Path, help="JSON list of {case_key, judged_outcome, reason}")
    parser.add_argument("--output", type=Path, help="Write per-case results as JSON here")
    args = parser.parse_args(argv)

    labels = [
        GoldenLabel(str(item["case_key"]), item["expected_outcome"], item.get("reason"))
        for item in json.loads(args.labels.read_text())
    ]
    runs = json.loads(args.runs.read_text())
    accepted = [
        AcceptedMiss(str(item["case_key"]), str(item["judged_outcome"]), str(item.get("reason") or ""))
        for item in (json.loads(args.accepted_misses.read_text()) if args.accepted_misses else [])
    ]
    judge = _load_factory(args.judge)
    reference = _load_factory(args.reference) if args.reference else None

    def load_run(label: GoldenLabel) -> AgentRun | None:
        record = runs.get(label.case_key)
        return run_from_record(record) if isinstance(record, Mapping) else None

    report = asyncio.run(run_golden_set(labels, load_run, judge, reference=reference, accepted_misses=accepted))
    print(report.summary())
    if args.output:
        args.output.write_text(json.dumps([asdict(result) for result in report.results], indent=1))
    return 0 if report.passed else 1


__all__ = [
    "AcceptedMiss",
    "GoldenCaseResult",
    "GoldenJudge",
    "GoldenLabel",
    "GoldenReference",
    "GoldenReport",
    "main",
    "run_from_record",
    "run_golden_set",
]


if __name__ == "__main__":
    sys.exit(main())
