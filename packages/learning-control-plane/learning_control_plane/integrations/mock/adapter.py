"""A synthetic framework adapter, structurally unlike PenguiFlow's, for the conformance suite.

Its native run shape (`MockRun`) does not look like PenguiFlow's `Trajectory`: different field
names (`call`/`input`/`output`/`ok` instead of `action.next_node`/`args`/`observation`/`error`), no
concept of a parallel step, and its own tiny in-memory skill store instead of PenguiFlow's. If the
`FrameworkAdapter` protocol (`integrations/protocol.py`) were secretly shaped around PenguiFlow, an
adapter this different could not satisfy it. It does, and the conformance suite
(`tests/learning_control_plane/test_provider_conformance.py`) runs the same scenarios against both
to prove it. This is also the template a real framework's adapter starts from.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any
from uuid import uuid4

from ...contracts.investigation import InvestigationTrajectoryV1, SourceTraceRef
from ...contracts.steps import GenericStep, GenericTrajectory
from ...control_plane.control_plane import ActivationReceipt, AdvisorySkillCandidate, DeliveryAuthorization


@dataclass(frozen=True, slots=True)
class MockEvent:
    """One call in the mock framework's own vocabulary -- not PenguiFlow's."""

    call: str
    input: Mapping[str, Any] = field(default_factory=dict)
    output: Any = None
    ok: bool = True
    error: str | None = None


@dataclass(frozen=True, slots=True)
class MockRun:
    """One run of the mock framework's synthetic agent."""

    question: str
    events: Sequence[MockEvent] = ()
    answer: str | None = None


@dataclass(frozen=True, slots=True)
class MockInvestigationContext:
    """The mock framework's equivalent of `PenguiFlowInvestigationContext`."""

    agent_ref: str
    scope_ref: str
    execution_fingerprint: str
    experiment_id: str = "mock-experiment"
    provider_ref: str = "mock"
    redaction_profile: str = "mock-investigation-safe:v1"


class MockSkillStore:
    """A tiny in-memory skill store -- proof that `deliver` needs no particular backend."""

    def __init__(self) -> None:
        self._skills: dict[str, dict[str, str]] = {}

    def put(self, *, scope_ref: str, candidate_id: str, guidance: str) -> str:
        digest = f"sha256:{sha256(guidance.encode()).hexdigest()}"
        self._skills[f"{scope_ref}:{candidate_id}"] = {"guidance": guidance, "digest": digest}
        return digest

    def get(self, *, scope_ref: str, candidate_id: str) -> dict[str, str] | None:
        return self._skills.get(f"{scope_ref}:{candidate_id}")


class MockGuidanceHook:
    """The object `MockFrameworkAdapter.attach_guidance` returns.

    Deliberately not shaped like PenguiFlow's `LLMContextHook` (a different method name, `apply`
    instead of `before_run`, and a synchronous call): the protocol only promises that *some* object
    comes back from `attach_guidance`, never what a caller does with it -- that part is always
    framework-specific, because each framework's own runtime decides how a hook is invoked.
    """

    def __init__(self, *, guidance: str, categories: tuple[str, ...]) -> None:
        self.guidance = guidance
        self.categories = categories

    def apply(self, context: Mapping[str, Any]) -> Mapping[str, Any] | None:
        if self.categories and context.get("category") not in self.categories:
            return None
        return {"guidance": self.guidance}


class MockFrameworkAdapter:
    """A synthetic `FrameworkAdapter` implementation, unrelated to PenguiFlow's internals."""

    def __init__(self, context: MockInvestigationContext, skill_store: MockSkillStore | None = None) -> None:
        self._context = context
        self._skill_store = skill_store or MockSkillStore()

    def to_generic_trajectory(self, native_run: MockRun) -> GenericTrajectory:
        return GenericTrajectory(
            query=native_run.question,
            steps=tuple(
                GenericStep(
                    tool=event.call,
                    args=event.input,
                    observation=event.output,
                    error=event.error,
                    failure={"error": event.error} if event.error else None,
                )
                for event in native_run.events
            ),
            final_answer=native_run.answer,
        )

    def project(
        self,
        native_run: MockRun,
        *,
        completed_at: datetime | None = None,
        investigation_id: str | None = None,
    ) -> InvestigationTrajectoryV1:
        steps = [{"index": index, "call": event.call, "ok": event.ok} for index, event in enumerate(native_run.events)]
        return InvestigationTrajectoryV1(
            investigation_id=investigation_id or f"investigation_{uuid4().hex[:24]}",
            source_trace_ref=SourceTraceRef(
                tracking_store_ref="mock",
                experiment_id=self._context.experiment_id,
                mlflow_trace_id=f"mock-trace-{uuid4().hex[:12]}",
                deployment_ref=self._context.execution_fingerprint,
            ),
            agent_ref=self._context.agent_ref,
            provider_ref=self._context.provider_ref,
            scope_ref=self._context.scope_ref,
            started_at=datetime.now(UTC),
            completed_at=completed_at,
            status="completed" if native_run.answer else "failed",
            execution_fingerprint=self._context.execution_fingerprint,
            request={"has_text": bool(native_run.question)},
            steps=steps,
            redaction_profile=self._context.redaction_profile,
            step_signature=">".join(str(step["call"]) for step in steps) or "no_steps",
            execution_context={
                "step_count": len(steps),
                "has_final_answer": native_run.answer is not None,
                "verified_success": native_run.answer is not None and all(event.ok for event in native_run.events),
            },
        )

    def attach_guidance(self, *, guidance: str, categories: tuple[str, ...]) -> MockGuidanceHook:
        return MockGuidanceHook(guidance=guidance, categories=categories)

    def deliver(
        self,
        authorization: DeliveryAuthorization,
        candidate: AdvisorySkillCandidate,
        *,
        now: datetime | None = None,
    ) -> ActivationReceipt:
        delivered_at = now or datetime.now(UTC)
        if not authorization.is_active(delivered_at):
            raise ValueError("delivery authorization is expired or revoked")
        if authorization.candidate_id != candidate.candidate_id:
            raise ValueError("candidate does not match delivery authorization")

        digest = self._skill_store.put(
            scope_ref=authorization.scope_ref,
            candidate_id=candidate.candidate_id,
            guidance=candidate.advisory_skill,
        )
        return ActivationReceipt(
            receipt_id=f"receipt_{uuid4().hex}",
            authorization_id=authorization.authorization_id,
            candidate_id=candidate.candidate_id,
            scope_ref=authorization.scope_ref,
            provider_ref=f"mock.skills:{authorization.scope_ref}:{candidate.candidate_id}",
            delivered_at=delivered_at,
            skill_digest=digest,
            investigation_digests=authorization.investigation_digests,
        )


__all__ = [
    "MockEvent",
    "MockFrameworkAdapter",
    "MockGuidanceHook",
    "MockInvestigationContext",
    "MockRun",
    "MockSkillStore",
]
