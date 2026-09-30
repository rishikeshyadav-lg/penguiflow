"""Trajectory scorers: did the agent take a sensible path, without demanding one exact path?

A correct answer can stand on wrong tools, garbage arguments, or a skipped authorization, and a success
rate cannot see that. These scorers read the steps of a `GenericTrajectory` (so they work on any framework's
runs) and check what matters: which tools were called, with what arguments, and whether the invariants
hold. Invariants are the middle position between scoring the exact path (brittle: agents find valid
routes nobody planned for) and scoring only the outcome (blind to how it was reached): name the events
that must happen, must never happen, or are merely tracked.

Tool names are compared ignoring case and surrounding spaces. Reports carry mismatch *codes*, never
argument values, so a report can be shared without leaking what the agent was given.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from .evaluation import EvaluationCase
from .execution import RepeatedRun
from .prediction import PredictionResult, ScoreResult
from .steps import GenericTrajectory
from .suites import case_scores

SequenceMode = Literal["exact", "in_order", "any_order"]


def trajectory_of(output: Any) -> GenericTrajectory:
    """The trajectory a runner returned: a `PredictionResult`'s, or a `GenericTrajectory` itself.

    A run with no trajectory cannot be scored on its path; that is an error, not a zero.
    """

    if isinstance(output, GenericTrajectory):
        return output
    if isinstance(output, PredictionResult) and output.trajectory is not None:
        return output.trajectory
    raise ValueError("the run has no trajectory to score; return a PredictionResult with a GenericTrajectory")


def _name(tool: str) -> str:
    return tool.strip().lower()


def called_tools(trajectory: GenericTrajectory) -> list[str]:
    """The tools called, in order."""

    return [_name(step.tool) for step in trajectory.steps]


def tool_selection_accuracy(expected: Sequence[str], actual: Sequence[str]) -> float:
    """Matched tool calls over the larger of expected and actual calls.

    That denominator punishes all three failure modes at once: too few calls, too many, and the wrong ones.
    Each expected call is matched at most once. No expected calls and no actual calls is a perfect 1.0.
    """

    expected_counts = Counter(_name(tool) for tool in expected)
    actual_counts = Counter(_name(tool) for tool in actual)
    matched = sum((expected_counts & actual_counts).values())
    denominator = max(sum(expected_counts.values()), sum(actual_counts.values()))
    return 1.0 if denominator == 0 else matched / denominator


def sequence_matches(expected: Sequence[str], actual: Sequence[str], mode: SequenceMode) -> bool:
    """Whether the calls made match the reference.

    `exact`: the same calls in the same order. `in_order`: the reference appears in order, other calls
    may lie between. `any_order`: every reference call appears at least as often as the reference has it.
    """

    wanted = [_name(tool) for tool in expected]
    made = [_name(tool) for tool in actual]
    if mode == "exact":
        return made == wanted
    if mode == "in_order":
        remaining = iter(made)
        return all(tool in remaining for tool in wanted)
    if mode == "any_order":
        return not (Counter(wanted) - Counter(made))
    raise ValueError("mode must be 'exact', 'in_order' or 'any_order'")


def _expected_tools_from_case(case: EvaluationCase) -> Sequence[str]:
    """The default reference: `case.expected["tools"]`."""

    if not isinstance(case.expected, Mapping) or "tools" not in case.expected:
        raise ValueError(f"case {case.case_id} has no expected['tools'] to compare the calls with")
    return case.expected["tools"]


@dataclass(frozen=True, slots=True)
class ToolSelection:
    """Scores the share of the right tools called (see `tool_selection_accuracy`) against the case's reference."""

    expected_tools: Callable[[EvaluationCase], Sequence[str]] = _expected_tools_from_case
    name: str = "tool_selection_accuracy"

    def __call__(self, case: EvaluationCase, output: Any) -> dict[str, float]:
        score = tool_selection_accuracy(self.expected_tools(case), called_tools(trajectory_of(output)))
        return {self.name: score}


@dataclass(frozen=True, slots=True)
class SequenceMatch:
    """1.0 when the calls match the case's reference in the chosen `mode`."""

    mode: SequenceMode
    expected_tools: Callable[[EvaluationCase], Sequence[str]] = _expected_tools_from_case
    name: str = "sequence_match"

    def __call__(self, case: EvaluationCase, output: Any) -> dict[str, float]:
        matched = sequence_matches(self.expected_tools(case), called_tools(trajectory_of(output)), self.mode)
        return {self.name: 1.0 if matched else 0.0}


@dataclass(frozen=True, slots=True)
class ToolArguments:
    """What a tool's arguments must look like.

    Syntactic: `required` names must be present, `types` must hold, and with `allow_extra=False` no other
    name may appear. Semantic: `ranges` (inclusive minimum and maximum, either may be None) and
    `allowed_values` bound the values. A bool is not accepted where a number is asked for.
    """

    required: Sequence[str] = ()
    types: Mapping[str, type | tuple[type, ...]] = field(default_factory=dict)
    ranges: Mapping[str, tuple[float | None, float | None]] = field(default_factory=dict)
    allowed_values: Mapping[str, Sequence[Any]] = field(default_factory=dict)
    allow_extra: bool = True


def _has_type(value: Any, expected: type | tuple[type, ...]) -> bool:
    types = expected if isinstance(expected, tuple) else (expected,)
    if isinstance(value, bool):
        return bool in types
    return isinstance(value, types)


def argument_codes(spec: ToolArguments, arguments: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """The syntactic and semantic mismatch codes for one call, each naming the argument and never its value."""

    syntactic: list[str] = []
    semantic: list[str] = []
    for name in spec.required:
        if name not in arguments:
            syntactic.append(f"missing_argument:{name}")
    if not spec.allow_extra:
        known = set(spec.required) | set(spec.types) | set(spec.ranges) | set(spec.allowed_values)
        syntactic.extend(f"unexpected_argument:{name}" for name in arguments if name not in known)
    for name, expected_type in spec.types.items():
        if name in arguments and not _has_type(arguments[name], expected_type):
            syntactic.append(f"wrong_type:{name}")
    for name, (minimum, maximum) in spec.ranges.items():
        value = arguments.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue  # a missing or non-numeric value is a syntactic problem, reported above
        if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
            semantic.append(f"out_of_range:{name}")
    for name, allowed in spec.allowed_values.items():
        if name in arguments and arguments[name] not in allowed:
            semantic.append(f"value_not_allowed:{name}")
    return syntactic, semantic


@dataclass(frozen=True, slots=True)
class CallArgumentCheck:
    """One call's argument check: which step, which tool, and the codes."""

    step_index: int
    tool: str
    syntactic_codes: Sequence[str]
    semantic_codes: Sequence[str]

    @property
    def correct(self) -> bool:
        return not self.syntactic_codes and not self.semantic_codes


def check_arguments(specs: Mapping[str, ToolArguments], trajectory: GenericTrajectory) -> list[CallArgumentCheck]:
    """Check every call to a tool that has a spec; calls to other tools are not checked."""

    normalized = {_name(tool): spec for tool, spec in specs.items()}
    checks = []
    for index, step in enumerate(trajectory.steps):
        spec = normalized.get(_name(step.tool))
        if spec is not None:
            syntactic, semantic = argument_codes(spec, step.args)
            checks.append(CallArgumentCheck(index, _name(step.tool), syntactic, semantic))
    return checks


@dataclass(frozen=True, slots=True)
class ArgumentCorrectness:
    """Scores the share of checked calls whose arguments are right, overall and at each level.

    Calls to tools with no spec are not checked, and a run with no checked calls scores 1.0 on all three:
    there was nothing to get wrong (whether the right tools were called is tool selection's job).
    """

    specs: Mapping[str, ToolArguments]
    name: str = "argument_correctness"

    def __call__(self, case: EvaluationCase, output: Any) -> dict[str, float]:
        checks = check_arguments(self.specs, trajectory_of(output))
        if not checks:
            return {self.name: 1.0, f"{self.name}_syntax": 1.0, f"{self.name}_semantics": 1.0}
        total = len(checks)
        return {
            self.name: sum(check.correct for check in checks) / total,
            f"{self.name}_syntax": sum(not check.syntactic_codes for check in checks) / total,
            f"{self.name}_semantics": sum(not check.semantic_codes for check in checks) / total,
        }


@dataclass(frozen=True, slots=True)
class Violation:
    """One broken invariant: a code for the kind of break and the tool it concerns."""

    code: str
    tool: str


@dataclass(frozen=True, slots=True)
class Required:
    """The tool must be called. With `before`, it applies only when `before` is called, and then the tool
    must have been called earlier ("an authorization before any write")."""

    tool: str
    before: str | None = None

    @property
    def label(self) -> str:
        return f"required:{_name(self.tool)}" + (f":before:{_name(self.before)}" if self.before else "")

    def violations(self, made: Sequence[str]) -> list[Violation]:
        tool = _name(self.tool)
        if self.before is None:
            return [] if tool in made else [Violation("required_tool_not_called", tool)]
        gate = _name(self.before)
        if gate not in made:
            return []
        if tool in made[: made.index(gate)]:
            return []
        return [Violation("required_tool_not_called_before", tool)]


@dataclass(frozen=True, slots=True)
class Forbidden:
    """The tool must never be called."""

    tool: str

    @property
    def label(self) -> str:
        return f"forbidden:{_name(self.tool)}"

    def violations(self, made: Sequence[str]) -> list[Violation]:
        tool = _name(self.tool)
        return [Violation("forbidden_tool_called", tool)] if tool in made else []


@dataclass(frozen=True, slots=True)
class AllowedTools:
    """No tool outside this list may be called."""

    tools: Sequence[str]

    @property
    def label(self) -> str:
        return "allowed_tools"

    def violations(self, made: Sequence[str]) -> list[Violation]:
        allowed = {_name(tool) for tool in self.tools}
        return [Violation("disallowed_tool_called", tool) for tool in sorted(set(made) - allowed)]


@dataclass(frozen=True, slots=True)
class MaxCalls:
    """The tool may be called at most `limit` times."""

    tool: str
    limit: int

    def __post_init__(self) -> None:
        if self.limit < 0:
            raise ValueError("limit must not be negative")

    @property
    def label(self) -> str:
        return f"max_calls:{_name(self.tool)}"

    def violations(self, made: Sequence[str]) -> list[Violation]:
        tool = _name(self.tool)
        return [Violation("maximum_tool_call_count_exceeded", tool)] if made.count(tool) > self.limit else []


@dataclass(frozen=True, slots=True)
class Tracked:
    """The tool is counted and reported but never fails a run."""

    tool: str

    @property
    def label(self) -> str:
        return f"tracked:{_name(self.tool)}"

    def violations(self, made: Sequence[str]) -> list[Violation]:
        return []


Invariant = Required | Forbidden | AllowedTools | MaxCalls | Tracked


@dataclass(frozen=True, slots=True)
class InvariantReport:
    """Which invariants broke, and how often each tracked tool was called."""

    violations: Sequence[Violation]
    results: Mapping[str, bool]
    tracked_counts: Mapping[str, int]

    @property
    def passed(self) -> bool:
        return not self.violations

    @property
    def codes(self) -> list[str]:
        """The distinct kinds of violation, sorted."""

        return sorted({violation.code for violation in self.violations})


def check_invariants(invariants: Sequence[Invariant], trajectory: GenericTrajectory) -> InvariantReport:
    """Check every invariant against the calls a run made."""

    made = called_tools(trajectory)
    violations: list[Violation] = []
    results: dict[str, bool] = {}
    tracked: dict[str, int] = {}
    for invariant in invariants:
        found = invariant.violations(made)
        violations.extend(found)
        results[invariant.label] = not found
        if isinstance(invariant, Tracked):
            tracked[_name(invariant.tool)] = made.count(_name(invariant.tool))
    return InvariantReport(violations=violations, results=results, tracked_counts=tracked)


@dataclass(frozen=True, slots=True)
class InvariantsHold:
    """1.0 when no invariant is broken. The score carries each invariant's result and the violation codes."""

    invariants: Sequence[Invariant]
    name: str = "invariants_hold"

    def __call__(self, case: EvaluationCase, output: Any) -> ScoreResult:
        report = check_invariants(self.invariants, trajectory_of(output))
        return ScoreResult(
            score=1.0 if report.passed else 0.0,
            feedback=", ".join(report.codes) or None,
            checks=report.results,
        )


@dataclass(frozen=True, slots=True)
class GapReport:
    """Task success beside tool selection. A wide gap means the agent reached results by a path nobody audited."""

    outcome_mean: float
    selection_mean: float
    gap: float
    flagged: bool
    note: str


def selection_gap(
    run: RepeatedRun, variant_id: str, *, outcome_metric: str, selection_metric: str, flag_above: float = 0.2
) -> GapReport:
    """Put the success rate and the tool-selection score side by side, per variant of a run.

    The gap is an audit signal, not a verdict: the agent may be substituting its own knowledge for a tool
    it should have called (a compliance hole), or the reference may simply be badly defined.
    """

    outcomes = case_scores(run, variant_id, outcome_metric)
    selections = case_scores(run, variant_id, selection_metric)
    outcome_mean = sum(outcomes.values()) / len(outcomes)
    selection_mean = sum(selections.values()) / len(selections)
    gap = outcome_mean - selection_mean
    flagged = gap > flag_above
    note = f"{outcome_metric} {outcome_mean:.2f} against {selection_metric} {selection_mean:.2f}: " + (
        "audit signal, not a verdict; check whether the agent answers from memory instead of calling its tools, "
        "or whether the reference is badly defined."
        if flagged
        else "no gap worth auditing."
    )
    return GapReport(outcome_mean, selection_mean, gap, flagged, note)


__all__ = [
    "AllowedTools",
    "ArgumentCorrectness",
    "CallArgumentCheck",
    "Forbidden",
    "GapReport",
    "Invariant",
    "InvariantReport",
    "InvariantsHold",
    "MaxCalls",
    "Required",
    "SequenceMatch",
    "SequenceMode",
    "ToolArguments",
    "ToolSelection",
    "Tracked",
    "Violation",
    "argument_codes",
    "called_tools",
    "check_arguments",
    "check_invariants",
    "selection_gap",
    "sequence_matches",
    "tool_selection_accuracy",
    "trajectory_of",
]
