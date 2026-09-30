"""Outcome scorers: did the agent reach the goal, and how much of it?

Every scorer here has the shape `(case, output) -> {metric_name: score}` and works on whatever a runner
returns: a `PredictionResult`, a plain string, or a number. Success should be measured on the state of
the world, not on what the agent says about it; `StateCheck` is that seam. An agent that writes "done"
is right only when the row it promised to create exists.
"""

from __future__ import annotations

import math
import re
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from .evaluation import EvaluationCase
from .prediction import PredictionResult

ToleranceKind = Literal["absolute", "relative"]


def answer_of(output: Any) -> Any:
    """The answer a runner gave: a `PredictionResult`'s answer, or the output itself."""

    if isinstance(output, PredictionResult):
        return output.answer if output.answer is not None else ""
    return output


def _normalized(text: Any) -> str:
    return " ".join(str(text).split()).casefold()


@dataclass(frozen=True, slots=True)
class ExactMatch:
    """1.0 when the answer equals `case.expected`. Text is compared ignoring case and spacing unless `normalize` is off."""

    name: str = "exact_match"
    normalize: bool = True

    def __call__(self, case: EvaluationCase, output: Any) -> dict[str, float]:
        answer = answer_of(output)
        if self.normalize:
            matched = _normalized(answer) == _normalized(case.expected)
        else:
            matched = answer == case.expected
        return {self.name: 1.0 if matched else 0.0}


@dataclass(frozen=True, slots=True)
class Contains:
    """1.0 when the answer contains `case.expected`; when that is a list, every item must appear."""

    name: str = "contains"

    def __call__(self, case: EvaluationCase, output: Any) -> dict[str, float]:
        expected = case.expected
        required = [expected] if isinstance(expected, str) else list(expected)
        answer = _normalized(answer_of(output))
        return {self.name: 1.0 if all(_normalized(item) in answer for item in required) else 0.0}


@dataclass(frozen=True, slots=True)
class RegexMatch:
    """1.0 when `pattern` (or `case.expected`, when no pattern is given) is found in the answer."""

    pattern: str | None = None
    name: str = "regex_match"

    def __call__(self, case: EvaluationCase, output: Any) -> dict[str, float]:
        pattern = self.pattern if self.pattern is not None else case.expected
        return {self.name: 1.0 if re.search(pattern, str(answer_of(output))) else 0.0}


@dataclass(frozen=True, slots=True)
class Tolerance:
    """How close a number must be to count as right.

    `absolute`: within `value` of the expected number. `relative`: within `value` of it proportionally,
    or within `floor` absolutely, whichever is larger. Lifted from the campaign's decision tolerance.
    """

    kind: ToleranceKind
    value: float
    floor: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in ("absolute", "relative"):
            raise ValueError("tolerance kind must be 'absolute' or 'relative'")
        for field_name, number in (("value", self.value), ("floor", self.floor)):
            if not math.isfinite(number) or number < 0:
                raise ValueError(f"tolerance {field_name} must be finite and non-negative")

    def matches(self, claimed: float, expected: float) -> bool:
        """Whether a claimed number is within this tolerance of the expected one."""

        if not math.isfinite(claimed) or not math.isfinite(expected):
            return False
        difference = abs(claimed - expected)
        if self.kind == "absolute":
            allowed = self.value
        else:
            allowed = max(self.value * abs(expected), self.floor)
        # A boundary value can land a float-representation hair past `allowed`
        # (0.0665 - 0.066 == 0.0005000000000000004); isclose absorbs that without widening the tolerance.
        return difference <= allowed or math.isclose(difference, allowed, rel_tol=1e-9, abs_tol=1e-12)


def _as_number(answer: Any) -> float | None:
    if isinstance(answer, bool):
        return None
    if isinstance(answer, (int, float)):
        return float(answer)
    try:
        return float(str(answer).strip().replace(",", ""))
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class NumericMatch:
    """1.0 when the answer is a number within `tolerance` of `case.expected`; 0.0 when it is not a number at all."""

    tolerance: Tolerance
    name: str = "numeric_match"

    def __call__(self, case: EvaluationCase, output: Any) -> dict[str, float]:
        expected = _as_number(case.expected)
        if expected is None:
            raise ValueError(f"case {case.case_id} has no numeric expected value")
        claimed = _as_number(answer_of(output))
        return {self.name: 1.0 if claimed is not None and self.tolerance.matches(claimed, expected) else 0.0}


class StateCheck(Protocol):
    """A check on the environment's state, used as a scorer: it returns whether the world is as it should be.

    It receives the case and whatever the runner returned, and looks at the real state (a database
    row, a file), not at the agent's claim. It may be a plain function or a coroutine function; give it
    a `name` (or a `def` name) so its score has one.
    """

    def __call__(self, case: EvaluationCase, output: Any) -> bool | Awaitable[bool]: ...


@dataclass(frozen=True, slots=True)
class WeightedRubric:
    """Criteria with weights, and the rule that turns their scores into a pass.

    A criterion scored None does not apply to this task and leaves the denominator, so the score is
    the weighted average of the criteria that do. `required_full_score` names criteria that must score
    exactly 1.0 (a criterion that does not apply cannot satisfy that).
    """

    weights: Mapping[str, float]
    minimum_score: float = 0.85
    required_full_score: Sequence[str] = ()

    def __post_init__(self) -> None:
        if not self.weights:
            raise ValueError("a rubric needs at least one criterion")
        for name, weight in self.weights.items():
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError(f"rubric criterion weight must be positive and finite: {name}")
        if not 0 <= self.minimum_score <= 1:
            raise ValueError("minimum_score must be between 0 and 1")
        if not set(self.required_full_score) <= set(self.weights):
            raise ValueError("required_full_score must name rubric criteria")
        object.__setattr__(self, "weights", dict(self.weights))
        object.__setattr__(self, "required_full_score", tuple(self.required_full_score))


@dataclass(frozen=True, slots=True)
class RubricResult:
    """A rubric's score and verdict for one answer."""

    score: float
    passed: bool
    applicable: Sequence[str]
    effective_weights: Mapping[str, float]
    failure_codes: Sequence[str]


def score_rubric(
    rubric: WeightedRubric, scores: Mapping[str, float | None], *, failure_codes: Sequence[str] = ()
) -> RubricResult:
    """Weight the criterion scores (0 to 1, or None when a criterion does not apply) and apply the pass rule.

    Any failure code fails the answer whatever its score.
    """

    missing = sorted(set(rubric.weights) - set(scores))
    extra = sorted(set(scores) - set(rubric.weights))
    if missing or extra:
        raise ValueError(f"criterion scores do not match the rubric; missing={missing}, extra={extra}")
    applicable_scores = {name: value for name, value in scores.items() if value is not None}
    applicable = [name for name in rubric.weights if name in applicable_scores]
    applicable_weight = sum(rubric.weights[name] for name in applicable)
    if applicable_weight == 0:
        score, effective = 0.0, {}
    else:
        score = sum(rubric.weights[name] * applicable_scores[name] for name in applicable) / applicable_weight
        effective = {name: rubric.weights[name] / applicable_weight for name in applicable}
    passed = (
        not failure_codes
        and score >= rubric.minimum_score
        and all(scores[name] == 1.0 for name in rubric.required_full_score)
    )
    return RubricResult(
        score=score,
        passed=passed,
        applicable=tuple(applicable),
        effective_weights=effective,
        failure_codes=tuple(failure_codes),
    )


__all__ = [
    "Contains",
    "ExactMatch",
    "NumericMatch",
    "RegexMatch",
    "RubricResult",
    "StateCheck",
    "Tolerance",
    "WeightedRubric",
    "answer_of",
    "score_rubric",
]
