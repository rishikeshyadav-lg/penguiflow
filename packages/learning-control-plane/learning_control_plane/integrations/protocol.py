"""The contract a new agent framework implements to plug into the learning control plane.

Adding a framework means writing one small adapter that satisfies this `Protocol` -- nothing in
`control_plane/`, `evaluation/`, `mining/`, or the contracts changes; requirement 1 of
`docs/proposals/FRAMEWORK_AGNOSTIC_LEARNING_CONTROL_PLANE.md`. Four responsibilities, each already
framework-agnostic on the control-plane side of the boundary:

1. `to_generic_trajectory` -- turn one native run into the framework-neutral step shape
   (`GenericStep` / `GenericTrajectory`, see `contracts/steps.py`) that a verifier reads. This is
   the one translation every adapter must write; everything downstream of it is shared code.
2. `project` -- turn the same native run into `InvestigationTrajectoryV1`, the redacted record
   mining reads back from the trace store.
3. `attach_guidance` -- build whatever this framework's own runtime needs to hand an approved
   skill to a live turn (PenguiFlow: an `LLMContextHook`; LangChain: a callback or prompt step).
4. `deliver` -- write an approved candidate to wherever this framework's agents read skills from,
   and return an `ActivationReceipt` the control plane can record.

`FrameworkAdapter` is a `Protocol` (`@runtime_checkable`), so an implementation needs no import from
this module and no subclassing -- an object that has these four methods satisfies the contract. The
conformance suite (`tests/learning_control_plane/test_provider_conformance.py`) runs the same
scenarios against every adapter that claims to implement it: PenguiFlow's, a synthetic mock with a
run shape nothing like PenguiFlow's, and LangChain's.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from ..contracts.investigation import InvestigationTrajectoryV1
from ..contracts.steps import GenericTrajectory
from ..control_plane.control_plane import ActivationReceipt, AdvisorySkillCandidate, DeliveryAuthorization


@runtime_checkable
class FrameworkAdapter(Protocol):
    """What a framework integration supplies; see the module docstring for each method's job."""

    def to_generic_trajectory(self, native_run: Any) -> GenericTrajectory:
        """Translate one native run into the framework-neutral shape a verifier reads."""
        ...

    def project(
        self,
        native_run: Any,
        *,
        completed_at: datetime | None = None,
        investigation_id: str | None = None,
    ) -> InvestigationTrajectoryV1:
        """Translate one native run into the redacted record mining reads from the trace store."""
        ...

    def attach_guidance(self, *, guidance: str, categories: tuple[str, ...]) -> Any:
        """Build this framework's own runtime hook that injects `guidance` on a matching turn."""
        ...

    def deliver(
        self,
        authorization: DeliveryAuthorization,
        candidate: AdvisorySkillCandidate,
        *,
        now: datetime | None = None,
    ) -> ActivationReceipt:
        """Write the candidate to this framework's skill store and return a receipt."""
        ...


__all__ = ["FrameworkAdapter"]
