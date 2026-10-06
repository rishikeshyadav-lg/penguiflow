"""Compare two runs by what they did, not only by what they said.

Two runs can give the same answer by different routes: another tool, other arguments, more cost, a
guardrail that fired, a policy code raised. A comparison that looks only at the final answer calls those
identical and tests less than it appears to. `diff_runs` compares the answer and also the tool
sequence, the arguments (by name, never value), cost, latency, guardrails and policy findings.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from ..scoring.golden import GoldenTrajectory
from ..scoring.outcome import _normalized
from ..scoring.policy import PolicyCheck
from ..core.prediction import PredictionResult
from ..core.steps import GenericStep, GenericTrajectory
from ..scoring.trajectory import trajectory_of

GUARDRAILS_KEY = "guardrails_triggered"


@dataclass(frozen=True, slots=True)
class RunView:
    """What a comparison reads from a run, whichever kind of run it is."""

    answer: str | None
    steps: Sequence[GenericStep]
    cost_usd: float | None
    latency_ms: float | None
    guardrails: Sequence[str]


def view_of(run: PredictionResult | GoldenTrajectory | GenericTrajectory) -> RunView:
    """Read a run as a `RunView`. A `PredictionResult` keeps its guardrails in `extra["guardrails_triggered"]`."""

    if isinstance(run, GoldenTrajectory):
        return RunView(run.final_answer, run.steps, None, None, ())
    if isinstance(run, GenericTrajectory):
        return RunView(run.final_answer, run.steps, None, None, ())
    trajectory = trajectory_of(run)
    return RunView(
        answer=run.answer,
        steps=trajectory.steps,
        cost_usd=run.cost_usd,
        latency_ms=run.latency_ms,
        guardrails=tuple(run.extra.get(GUARDRAILS_KEY, ())),
    )


@dataclass(frozen=True, slots=True)
class ArgumentChange:
    """A call kept the same tool but changed arguments. It names them and never their values."""

    step_index: int
    tool: str
    changed_arguments: Sequence[str]


@dataclass(frozen=True, slots=True)
class TrajectoryDiff:
    """How a candidate run differs from a reference run."""

    same_answer: bool
    tool_sequence_same: bool
    first_divergence: int | None
    tools_removed: Sequence[str]
    tools_added: Sequence[str]
    argument_changes: Sequence[ArgumentChange]
    cost_delta: float | None
    latency_delta: float | None
    guardrails_added: Sequence[str]
    guardrails_removed: Sequence[str]
    policy_codes_added: Sequence[str]
    policy_codes_removed: Sequence[str]

    @property
    def process_identical(self) -> bool:
        """Whether the candidate did the same things: same tools in order, arguments, guardrails and policy findings.
        Cost and latency are reported as deltas and do not decide this."""

        return not (
            not self.tool_sequence_same
            or self.argument_changes
            or self.guardrails_added
            or self.guardrails_removed
            or self.policy_codes_added
            or self.policy_codes_removed
        )

    @property
    def identical(self) -> bool:
        """The same answer by the same process."""

        return self.same_answer and self.process_identical

    @property
    def differences(self) -> list[str]:
        """One line per difference, in words."""

        lines = []
        if not self.same_answer:
            lines.append("the answer differs")
        if not self.tool_sequence_same:
            lines.append(
                f"the tool sequence differs from step {self.first_divergence} "
                f"(removed {list(self.tools_removed)}, added {list(self.tools_added)})"
            )
        lines.extend(
            f"step {change.step_index} ({change.tool}) changed arguments {list(change.changed_arguments)}"
            for change in self.argument_changes
        )
        if self.guardrails_added or self.guardrails_removed:
            lines.append(
                f"guardrails differ (added {list(self.guardrails_added)}, removed {list(self.guardrails_removed)})"
            )
        if self.policy_codes_added or self.policy_codes_removed:
            lines.append(
                f"policy findings differ (added {list(self.policy_codes_added)}, "
                f"removed {list(self.policy_codes_removed)})"
            )
        return lines


def _delta(reference: float | None, candidate: float | None) -> float | None:
    return None if reference is None or candidate is None else candidate - reference


_MISSING: Any = object()


def _changed_argument_names(reference: GenericStep, candidate: GenericStep) -> list[str]:
    names = set(reference.args) | set(candidate.args)
    return sorted(name for name in names if reference.args.get(name, _MISSING) != candidate.args.get(name, _MISSING))


def diff_runs(
    reference: PredictionResult | GoldenTrajectory | GenericTrajectory,
    candidate: PredictionResult | GoldenTrajectory | GenericTrajectory,
    *,
    policy: PolicyCheck | None = None,
) -> TrajectoryDiff:
    """Compare a candidate run with a reference run, which may be a golden trajectory."""

    ref, cand = view_of(reference), view_of(candidate)
    ref_tools = [step.tool.strip().lower() for step in ref.steps]
    cand_tools = [step.tool.strip().lower() for step in cand.steps]
    matcher = SequenceMatcher(None, ref_tools, cand_tools, autojunk=False)
    removed: list[str] = []
    added: list[str] = []
    argument_changes: list[ArgumentChange] = []
    first_divergence: int | None = None
    for opcode, ref_start, ref_end, cand_start, cand_end in matcher.get_opcodes():
        if opcode == "equal":
            for offset in range(ref_end - ref_start):
                changed = _changed_argument_names(ref.steps[ref_start + offset], cand.steps[cand_start + offset])
                if changed:
                    argument_changes.append(
                        ArgumentChange(cand_start + offset, cand_tools[cand_start + offset], changed)
                    )
            continue
        if first_divergence is None:
            first_divergence = ref_start
        removed.extend(ref_tools[ref_start:ref_end])
        added.extend(cand_tools[cand_start:cand_end])

    ref_policy = set(policy.check(ref.steps).codes) if policy else set()
    cand_policy = set(policy.check(cand.steps).codes) if policy else set()
    ref_guardrails, cand_guardrails = set(ref.guardrails), set(cand.guardrails)
    return TrajectoryDiff(
        same_answer=_normalized(ref.answer or "") == _normalized(cand.answer or ""),
        tool_sequence_same=ref_tools == cand_tools,
        first_divergence=first_divergence,
        tools_removed=tuple(removed),
        tools_added=tuple(added),
        argument_changes=tuple(argument_changes),
        cost_delta=_delta(ref.cost_usd, cand.cost_usd),
        latency_delta=_delta(ref.latency_ms, cand.latency_ms),
        guardrails_added=tuple(sorted(cand_guardrails - ref_guardrails)),
        guardrails_removed=tuple(sorted(ref_guardrails - cand_guardrails)),
        policy_codes_added=tuple(sorted(cand_policy - ref_policy)),
        policy_codes_removed=tuple(sorted(ref_policy - cand_policy)),
    )


__all__ = ["ArgumentChange", "GUARDRAILS_KEY", "RunView", "TrajectoryDiff", "diff_runs", "view_of"]
