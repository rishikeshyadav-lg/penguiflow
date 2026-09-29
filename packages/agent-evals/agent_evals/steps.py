"""A framework-neutral shape for what an agent did during one run.

Every agent framework records a run differently: PenguiFlow's `Trajectory` walks a list of
`TrajectoryStep` objects with a `PlannerAction`; LangChain's `AgentExecutor` returns
`intermediate_steps` as `(AgentAction, observation)` tuples; other frameworks use their own event
shapes. A verifier that reads one framework's native shape directly cannot judge a different
framework's runs without learning that framework's internals.

`GenericStep` and `GenericTrajectory` are the shape a verifier or scorer reads instead. Each
agent integration supplies one function that translates its own native run into this shape once
(the learning control plane calls it `to_generic_trajectory`); a scorer's domain logic -- which tool
names mean what, which arguments matter -- never has to change when a new framework is added.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class GenericStep:
    """One tool call, in every framework's terms: what was asked, and what came back."""

    tool: str
    args: Mapping[str, Any] = field(default_factory=dict)
    observation: Any = None
    error: str | None = None
    failure: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.tool.strip():
            raise ValueError("tool must be non-empty")
        object.__setattr__(self, "args", dict(self.args))
        if self.failure is not None:
            if not isinstance(self.failure, Mapping):
                raise ValueError("failure must be a mapping or None")
            object.__setattr__(self, "failure", dict(self.failure))


@dataclass(frozen=True, slots=True)
class GenericTrajectory:
    """One agent run, in every framework's terms: the question, the steps, the answer."""

    query: str
    steps: Sequence[GenericStep] = ()
    final_answer: str | None = None
    # Carries whatever per-turn context a verifier needs beyond the steps themselves (for campaign,
    # the answer rubric); a plain mapping so no framework's own context type leaks into the verifier.
    llm_context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "llm_context", dict(self.llm_context))


__all__ = ["GenericStep", "GenericTrajectory"]
