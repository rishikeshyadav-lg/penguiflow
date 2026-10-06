"""Plan adherence and multi-step coherence, scored by a language model. Experimental.

Two things a success rate cannot see: whether the actions the agent took trace back to the plan it announced,
and whether each step is consistent with the steps before it. They separate a good plan poorly executed from
a bad plan executed flawlessly into a wall. Scoring them takes a judge that reads the run, and how far to trust
such a judge is an open question with its own reliability literature.

So these scorers are marked experimental. A report that uses one says how often the judge agreed with labelled
trajectories, or that it has not been measured. Until it has, treat the number as a hint, not a verdict.

The judge is asked for one true or false per step, and the score is the share of true. By default the judge is
shown the announced plan, the tool names, the names of their arguments and whether each step failed, and no
argument values or results, so a trajectory can be judged without sending what the agent handled.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from ..core.evaluation import EvaluationCase, _await_value
from .judging import JudgeClient
from ..core.steps import GenericStep, GenericTrajectory
from .trajectory import trajectory_of

Criterion = Literal["plan_adherence", "multi_step_coherence"]
PLAN_KEY = "plan"
_VALUE_LIMIT = 200

_INSTRUCTIONS: dict[str, str] = {
    "plan_adherence": (
        "An agent announced a plan and then took the actions listed below. For each action, decide whether it "
        "traces back to a step of the announced plan."
    ),
    "multi_step_coherence": (
        "An agent took the actions listed below in order. For each action, decide whether it is logically "
        "consistent with the actions before it: it builds on what came earlier and does not contradict or "
        "ignore it. The first action is consistent unless it is contradictory on its own."
    ),
}


def render_steps(steps: Sequence[GenericStep], *, include_values: bool = False) -> str:
    """The steps as numbered lines: the tool, its argument names, and whether it failed."""

    lines = []
    for index, step in enumerate(steps, start=1):
        status = "failed" if step.error is not None or step.failure is not None else "ok"
        if include_values:
            arguments = ", ".join(f"{name}={str(value)[:_VALUE_LIMIT]}" for name, value in step.args.items())
            result = f" -> {str(step.observation)[:_VALUE_LIMIT]}" if step.observation is not None else ""
            lines.append(f"{index}. {step.tool}({arguments}) [{status}]{result}")
        else:
            lines.append(f"{index}. {step.tool}({', '.join(sorted(step.args))}) [{status}]")
    return "\n".join(lines)


def _prompt(criterion: str, trajectory: GenericTrajectory, *, include_values: bool, include_query: bool) -> str:
    parts = [_INSTRUCTIONS[criterion]]
    if include_query:
        parts.append(f"The task: {trajectory.query}")
    if criterion == "plan_adherence":
        plan = trajectory.llm_context.get(PLAN_KEY)
        if not plan:
            raise ValueError(
                f"the run announced no plan (llm_context[{PLAN_KEY!r}]), so plan adherence cannot be judged"
            )
        parts.append(f"The announced plan:\n{plan}")
    parts.append("The actions:\n" + render_steps(trajectory.steps, include_values=include_values))
    parts.append(
        'Reply with only JSON: {"steps": [true, false, ...]}, one true or false per action, in order, '
        f"{len(trajectory.steps)} in all."
    )
    return "\n\n".join(parts)


def parse_step_verdicts(reply: str, expected_count: int) -> list[bool]:
    """Read `{"steps": [...]}` out of a reply that may have text or a code fence around it."""

    decoder = json.JSONDecoder()
    for index, character in enumerate(reply):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(reply[index:])
        except ValueError:
            continue
        verdicts = payload.get("steps") if isinstance(payload, dict) else None
        if isinstance(verdicts, list) and all(isinstance(item, bool) for item in verdicts):
            if len(verdicts) != expected_count:
                raise ValueError(f"the judge returned {len(verdicts)} verdicts for {expected_count} steps")
            return verdicts
    raise ValueError("the judge's reply did not contain a JSON object with a list of true or false under 'steps'")


@dataclass(frozen=True, slots=True)
class TrajectoryJudge:
    """Scores plan adherence or multi-step coherence as the share of steps the judge approves. Experimental.

    A run with no steps is an error (there is nothing to judge), as is a plan-adherence run that announced no
    plan in `trajectory.llm_context["plan"]`. Neither is scored as zero.
    """

    client: JudgeClient
    criterion: Criterion
    name: str | None = None
    include_values: bool = False
    include_query: bool = False
    experimental: bool = True

    def __post_init__(self) -> None:
        if self.criterion not in _INSTRUCTIONS:
            raise ValueError("criterion must be 'plan_adherence' or 'multi_step_coherence'")

    async def __call__(self, case: EvaluationCase, output: Any) -> dict[str, float]:
        trajectory = trajectory_of(output)
        if not trajectory.steps:
            raise ValueError("the run has no steps to judge")
        prompt = _prompt(
            self.criterion, trajectory, include_values=self.include_values, include_query=self.include_query
        )
        reply = await _await_value(self.client(prompt))
        verdicts = parse_step_verdicts(reply, len(trajectory.steps))
        return {self.name or self.criterion: sum(verdicts) / len(verdicts)}


def plan_adherence(client: JudgeClient, **options: Any) -> TrajectoryJudge:
    """A plan-adherence scorer (experimental) that asks `client` about each step."""

    return TrajectoryJudge(client, "plan_adherence", **options)


def multi_step_coherence(client: JudgeClient, **options: Any) -> TrajectoryJudge:
    """A multi-step-coherence scorer (experimental) that asks `client` about each step."""

    return TrajectoryJudge(client, "multi_step_coherence", **options)


__all__ = [
    "Criterion",
    "PLAN_KEY",
    "TrajectoryJudge",
    "multi_step_coherence",
    "parse_step_verdicts",
    "plan_adherence",
    "render_steps",
]
