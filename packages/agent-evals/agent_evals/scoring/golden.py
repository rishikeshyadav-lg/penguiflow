"""Golden trajectories: a frozen run you compare a candidate against by its decisions, not by a string.

A golden trajectory holds the inputs the run saw, every tool call and what came back, the answer, and a
reference to the environment it ran in, under one digest. Compared with a stored final answer, it lets a
regression be caught when the answer stays the same but the path changes.

A golden trajectory is code that has to be maintained. When an API or a policy changes for a good reason,
the path has to be re-approved on purpose: who approved it, why, and which digest it replaced. A careless
refresh turns a bug into the new "correct" behaviour, so a refresh without that record is refused.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.evaluation import _non_empty
from ..core.steps import GenericStep, GenericTrajectory


class GoldenRefreshError(ValueError):
    """A golden trajectory was replaced without a valid approval."""


def _step_payload(step: GenericStep) -> dict[str, Any]:
    return {
        "tool": step.tool,
        "args": dict(step.args),
        "observation": step.observation,
        "error": step.error,
        "failure": dict(step.failure) if step.failure is not None else None,
    }


def _canonical(payload: Any) -> Any:
    """The payload as plain JSON data; anything not JSON-native becomes its string."""

    return json.loads(json.dumps(payload, sort_keys=True, default=str))


@dataclass(frozen=True, slots=True)
class GoldenApproval:
    """Who re-approved a golden trajectory, why, and which digest it replaced."""

    approver: str
    reason: str
    previous_digest: str
    new_digest: str
    approved_at: str

    def __post_init__(self) -> None:
        for name in ("approver", "reason", "previous_digest", "new_digest", "approved_at"):
            object.__setattr__(self, name, _non_empty(getattr(self, name), name))


@dataclass(frozen=True, slots=True)
class GoldenTrajectory:
    """One frozen run, with a digest over everything that defines it. Approvals are history, not content."""

    case_id: str
    query: str
    steps: Sequence[GenericStep]
    final_answer: str | None
    environment_ref: str
    approvals: Sequence[GoldenApproval] = ()
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _non_empty(self.case_id, "case_id"))
        object.__setattr__(self, "environment_ref", _non_empty(self.environment_ref, "environment_ref"))
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "approvals", tuple(self.approvals))
        content = _canonical(
            {
                "case_id": self.case_id,
                "query": self.query,
                "steps": [_step_payload(step) for step in self.steps],
                "final_answer": self.final_answer,
                "environment_ref": self.environment_ref,
            }
        )
        encoded = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        object.__setattr__(self, "digest", f"sha256:{hashlib.sha256(encoded.encode()).hexdigest()}")


def freeze_golden(case_id: str, trajectory: GenericTrajectory, *, environment_ref: str) -> GoldenTrajectory:
    """Freeze a run as a golden trajectory, tied to the environment it ran in."""

    return GoldenTrajectory(
        case_id=case_id,
        query=trajectory.query,
        steps=trajectory.steps,
        final_answer=trajectory.final_answer,
        environment_ref=environment_ref,
    )


def refresh_golden(
    current: GoldenTrajectory, new: GoldenTrajectory, approval: GoldenApproval | None
) -> GoldenTrajectory:
    """Replace a golden trajectory, only with an approval that names exactly this change.

    The approval must be for the digest being replaced and the digest replacing it, so it cannot be reused
    for another refresh. The new golden keeps the full approval history.
    """

    if approval is None:
        raise GoldenRefreshError("a golden trajectory cannot be refreshed without an approval record")
    if new.case_id != current.case_id:
        raise GoldenRefreshError(f"the new run is for case {new.case_id}, not {current.case_id}")
    if approval.previous_digest != current.digest:
        raise GoldenRefreshError("the approval is not for the golden trajectory being replaced")
    if approval.new_digest != new.digest:
        raise GoldenRefreshError("the approval is not for the new run")
    return GoldenTrajectory(
        case_id=new.case_id,
        query=new.query,
        steps=new.steps,
        final_answer=new.final_answer,
        environment_ref=new.environment_ref,
        approvals=(*current.approvals, approval),
    )


def save_golden(golden: GoldenTrajectory, path: str | Path) -> None:
    """Write a golden trajectory as JSON, with its digest, so a later edit of the file is caught on load."""

    payload = {
        "case_id": golden.case_id,
        "query": golden.query,
        "steps": [_step_payload(step) for step in golden.steps],
        "final_answer": golden.final_answer,
        "environment_ref": golden.environment_ref,
        "approvals": [
            {
                "approver": a.approver,
                "reason": a.reason,
                "previous_digest": a.previous_digest,
                "new_digest": a.new_digest,
                "approved_at": a.approved_at,
            }
            for a in golden.approvals
        ],
        "digest": golden.digest,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_canonical(payload), sort_keys=True, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def load_golden(path: str | Path) -> GoldenTrajectory:
    """Read a golden trajectory and refuse one whose content no longer matches its recorded digest."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    golden = GoldenTrajectory(
        case_id=payload["case_id"],
        query=payload["query"],
        steps=[
            GenericStep(
                tool=step["tool"],
                args=step["args"],
                observation=step["observation"],
                error=step["error"],
                failure=step["failure"],
            )
            for step in payload["steps"]
        ],
        final_answer=payload["final_answer"],
        environment_ref=payload["environment_ref"],
        approvals=[GoldenApproval(**approval) for approval in payload["approvals"]],
    )
    if golden.digest != payload["digest"]:
        raise ValueError(
            f"{path} was edited after it was frozen (digest {golden.digest}, recorded {payload['digest']})"
        )
    return golden


__all__ = [
    "GoldenApproval",
    "GoldenRefreshError",
    "GoldenTrajectory",
    "freeze_golden",
    "load_golden",
    "refresh_golden",
    "save_golden",
]
