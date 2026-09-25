"""Tell a tool failure the agent moved past from one that sank the run.

Probing a field that does not exist, then answering from the right one, is normal exploration.
Treating every failed call as a failed step blocked verification of runs whose answers were right,
so a failed (or empty) call counts as recovered when a later data call completed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from .runs import AgentStep

# Result-check code a projector writes on a step the agent recovered from.
RECOVERED_STEP_CODE = "tool_error_recovered"


def step_failed(step: AgentStep) -> bool:
    """Return whether a tool call failed: it raised, or its result is an error payload."""

    return bool(step.error) or (isinstance(step.result, Mapping) and "error" in step.result)


def recovered_step_indexes(
    steps: Sequence[AgentStep],
    *,
    is_data_step: Callable[[AgentStep], bool],
    came_back_empty: Callable[[AgentStep], bool] | None = None,
) -> frozenset[int]:
    """Return the indexes of failed or empty steps after which a data step completed.

    `is_data_step` names the calls that fetch the data an answer rests on; `came_back_empty`
    optionally marks a completed call whose result holds nothing usable.
    """

    recovered: set[int] = set()
    for index, step in enumerate(steps):
        empty = came_back_empty is not None and not step_failed(step) and came_back_empty(step)
        if not (step_failed(step) or empty):
            continue
        if any(is_data_step(later) and not step_failed(later) for later in steps[index + 1 :]):
            recovered.add(index)
    return frozenset(recovered)


__all__ = ["RECOVERED_STEP_CODE", "recovered_step_indexes", "step_failed"]
