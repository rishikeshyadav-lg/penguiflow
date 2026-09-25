"""A framework-neutral record of one agent run, the only input the judge kit reads.

Any agent can be judged once its run is expressed as an `AgentRun`: the question, each tool call
with its arguments, result and error, the final answer, and whatever tables or reports were shown
to the user beside it. The PenguiFlow integration builds one from a trajectory; a plain-Python
agent builds one directly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

RenderedKind = Literal["table", "report"]


@dataclass(frozen=True, slots=True)
class AgentStep:
    """One tool call: its name, arguments, result, the error it raised, and whether it streamed output."""

    tool: str
    args: Mapping[str, Any] = field(default_factory=dict)
    result: Any = None
    error: str | None = None
    streamed: bool = False


@dataclass(frozen=True, slots=True)
class RenderedOutput:
    """A table or report shown to the user beside the answer text.

    A table's `content` has `columns` (each with a `field` and an optional `header`) and `rows`
    (mappings from field to value). A report's `content` has `sections`, each with a `title`, a
    markdown `content` string and optional nested `subsections`.
    """

    kind: RenderedKind
    content: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class AgentRun:
    """Everything the judge needs from one run, with no framework types."""

    question: str
    steps: Sequence[AgentStep]
    final_answer: str | None
    rendered: Sequence[RenderedOutput] = ()
    classification: Mapping[str, Any] = field(default_factory=dict)
    finish_reason: str = "answer_complete"
    # Non-text inputs (images, files) that came with the question; only their count is ever projected.
    input_part_count: int = 0


__all__ = ["AgentRun", "AgentStep", "RenderedKind", "RenderedOutput"]
