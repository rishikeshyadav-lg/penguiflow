"""What running an agent returns, what scoring returns, and how a scorer's answer is read.

`PredictionResult` is the shape an agent's runner is encouraged to return. It is not required: a
runner may return anything, and scorers then receive that value. Returning `PredictionResult` is what
lets the framework score tool calls, count cost and latency, and treat a failed or paused run as a
failure instead of scoring an empty answer.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from .steps import GenericTrajectory

PredictionStatus = Literal["ok", "failed", "cancelled", "paused"]
_STATUSES = ("ok", "failed", "cancelled", "paused")


def _finite_non_negative(value: float | None, field_name: str) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field_name} must be a finite, non-negative number")
    return number


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """One agent run on one case, in terms every agent can meet."""

    status: PredictionStatus = "ok"
    answer: str | None = None
    trajectory: GenericTrajectory | None = None
    latency_ms: float | None = None
    cost_usd: float | None = None
    llm_usage: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None
    # Anything else worth keeping with the run: a route, trace ids, the effective contexts.
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in _STATUSES:
            raise ValueError(f"status must be one of {', '.join(_STATUSES)}")
        if self.status == "ok" and self.error is not None:
            raise ValueError("a run with status ok cannot carry an error")
        if self.status == "failed" and not (self.error and self.error.strip()):
            raise ValueError("a run with status failed must say why")
        object.__setattr__(self, "latency_ms", _finite_non_negative(self.latency_ms, "latency_ms"))
        object.__setattr__(self, "cost_usd", _finite_non_negative(self.cost_usd, "cost_usd"))
        object.__setattr__(self, "llm_usage", dict(self.llm_usage))
        object.__setattr__(self, "extra", dict(self.extra))

    @property
    def completed(self) -> bool:
        """Whether the run finished and its output is worth scoring."""

        return self.status == "ok"


@dataclass(frozen=True, slots=True)
class ScoreResult:
    """One score with the reason for it, for scorers that want to explain themselves."""

    score: float
    feedback: str | None = None
    checks: Mapping[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        score = float(self.score)
        if not math.isfinite(score):
            raise ValueError("score must be finite")
        object.__setattr__(self, "score", score)
        if not all(isinstance(passed, bool) for passed in self.checks.values()):
            raise ValueError("every check must be True or False")
        object.__setattr__(self, "checks", dict(self.checks))


ScoreValue = float | int | bool | Mapping[str, float] | ScoreResult


def scorer_name(scorer: Callable[..., Any]) -> str | None:
    """The name a scorer that returns a single score is known by, or None when it has none."""

    name = getattr(scorer, "name", None) or getattr(scorer, "__name__", None)
    if not isinstance(name, str) or not name.strip() or name == "<lambda>":
        return None
    return name.strip()


def normalize_scores(value: ScoreValue, *, scorer: str | None) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    """Read what a scorer returned as `(metrics, details)`.

    A mapping names its own metrics. A single number, a bool (1.0 or 0.0) or a `ScoreResult` is
    reported under the scorer's own name, so such a scorer must have one.
    """

    if isinstance(value, Mapping):
        return {str(name): float(number) for name, number in value.items()}, {}
    if isinstance(value, ScoreResult):
        name = _required_name(scorer)
        details: dict[str, Any] = {}
        if value.feedback is not None:
            details["feedback"] = value.feedback
        if value.checks:
            details["checks"] = dict(value.checks)
        return {name: value.score}, ({name: details} if details else {})
    if isinstance(value, (bool, int, float)):
        return {_required_name(scorer): float(value)}, {}
    raise ValueError(f"a scorer must return a mapping, a number or a ScoreResult, not {type(value).__name__}")


def _required_name(scorer: str | None) -> str:
    if scorer is None:
        raise ValueError("a scorer that returns a single score needs a name (a `name` attribute or a def name)")
    return scorer


__all__ = [
    "PredictionResult",
    "PredictionStatus",
    "ScoreResult",
    "ScoreValue",
    "normalize_scores",
    "scorer_name",
]
