"""The framework-neutral step shape, re-exported from `agent_evals`.

The contract lives in `agent_evals` because scoring an agent's run does not depend on the learning
loop. It is re-exported here so every existing `learning_control_plane.contracts.steps` import keeps
working and refers to the very same classes.
"""

from __future__ import annotations

from agent_evals.core.steps import GenericStep, GenericTrajectory

__all__ = ["GenericStep", "GenericTrajectory"]
