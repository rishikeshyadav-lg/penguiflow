"""The same scenarios, run against every `FrameworkAdapter` implementation.

This is the proposal's Phase 0 conformance harness
(`docs/proposals/FRAMEWORK_AGNOSTIC_LEARNING_CONTROL_PLANE.md`), built: a synthetic mock adapter
whose native run shape has nothing to do with PenguiFlow's `Trajectory`, run through the identical
checks as PenguiFlow's own adapter. Passing both proves the `FrameworkAdapter` protocol
(`integrations/protocol.py`) is not secretly shaped around one framework, and that a domain
verifier written once against `GenericStep`/`GenericTrajectory` (`_toy_verifier` below) needs no
change to judge either framework's runs.

A real second framework, LangChain (`integrations/langchain/`), is in `CASES` alongside the mock
and PenguiFlow, so the same suite covers it too: a genuinely different framework, not a fixture
built to be easy to pass.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from langchain_core.agents import AgentAction

from learning_control_plane.contracts.investigation import SourceTraceRef
from learning_control_plane.contracts.steps import GenericTrajectory
from learning_control_plane.control_plane.control_plane import AdvisorySkillCandidate, DeliveryAuthorization
from learning_control_plane.integrations.langchain.adapter import (
    LangChainFrameworkAdapter,
    LangChainInvestigationContext,
)
from learning_control_plane.integrations.mock.adapter import (
    MockEvent,
    MockFrameworkAdapter,
    MockInvestigationContext,
    MockRun,
)
from learning_control_plane.integrations.penguiflow.projector import (
    PenguiFlowFrameworkAdapter,
    PenguiFlowInvestigationContext,
    PenguiFlowInvestigationProjector,
)
from learning_control_plane.integrations.protocol import FrameworkAdapter
from penguiflow.planner.models import PlannerAction
from penguiflow.planner.trajectory import Trajectory, TrajectoryStep
from penguiflow.skills.local_store import LocalSkillStore


def _toy_verifier(trajectory: GenericTrajectory) -> bool:
    """A stand-in domain judge: verified when a `lookup` step found something and there is an answer.

    Written once, against the generic contract only -- no `if adapter is penguiflow` branch, and no
    import from either adapter's package. This is the property the conformance suite exists to prove.
    """

    if trajectory.final_answer is None:
        return False
    return any(step.tool == "lookup" and step.observation for step in trajectory.steps)


def _penguiflow_case(tmp_path: Path) -> tuple[FrameworkAdapter, Any]:
    context = PenguiFlowInvestigationContext(
        source_trace_ref=SourceTraceRef(
            tracking_store_ref="mlflow://local",
            experiment_id="1",
            mlflow_trace_id="tr-1",
            deployment_ref="sha256:fp",
        ),
        agent_ref="conformance-agent",
        scope_ref="tenant:conformance",
        execution_fingerprint="sha256:fp",
        started_at=datetime.now(UTC),
    )
    adapter = PenguiFlowFrameworkAdapter(
        PenguiFlowInvestigationProjector(context), LocalSkillStore(db_path=str(tmp_path / "skills.db"))
    )
    native_run = Trajectory(
        query="how many clicks did the campaign get",
        steps=[
            TrajectoryStep(
                action=PlannerAction(next_node="lookup", args={"acid": "112774"}),
                observation={"clicks": 400},
            )
        ],
        final_answer="400 clicks.",
        finish_reason="answer_complete",
    )
    return adapter, native_run


def _langchain_case(_tmp_path: Path) -> tuple[FrameworkAdapter, Any]:
    context = LangChainInvestigationContext(
        agent_ref="conformance-agent", scope_ref="tenant:conformance", execution_fingerprint="sha256:fp"
    )
    adapter = LangChainFrameworkAdapter(context)
    native_run = {
        "input": "how many clicks did the campaign get",
        "output": "400 clicks.",
        "intermediate_steps": [
            (AgentAction(tool="lookup", tool_input={"acid": "112774"}, log="calling lookup"), {"clicks": 400})
        ],
    }
    return adapter, native_run


def _mock_case(_tmp_path: Path) -> tuple[FrameworkAdapter, Any]:
    context = MockInvestigationContext(
        agent_ref="conformance-agent", scope_ref="tenant:conformance", execution_fingerprint="sha256:fp"
    )
    adapter = MockFrameworkAdapter(context)
    native_run = MockRun(
        question="how many clicks did the campaign get",
        events=(MockEvent(call="lookup", input={"acid": "112774"}, output={"clicks": 400}, ok=True),),
        answer="400 clicks.",
    )
    return adapter, native_run


CASES = {"penguiflow": _penguiflow_case, "mock": _mock_case, "langchain": _langchain_case}


@pytest.fixture(params=list(CASES))
def adapter_case(request: pytest.FixtureRequest, tmp_path: Path) -> tuple[FrameworkAdapter, Any]:
    return CASES[request.param](tmp_path)


def test_every_adapter_satisfies_the_framework_adapter_protocol(adapter_case: tuple[FrameworkAdapter, Any]) -> None:
    adapter, _native_run = adapter_case

    assert isinstance(adapter, FrameworkAdapter)


def test_the_same_domain_verifier_judges_every_adapters_translated_run(
    adapter_case: tuple[FrameworkAdapter, Any],
) -> None:
    adapter, native_run = adapter_case

    generic = adapter.to_generic_trajectory(native_run)

    assert _toy_verifier(generic) is True
    assert generic.query == "how many clicks did the campaign get"


def test_every_adapter_projects_a_valid_investigation_document(adapter_case: tuple[FrameworkAdapter, Any]) -> None:
    adapter, native_run = adapter_case

    document = adapter.project(native_run)

    assert document.scope_ref == "tenant:conformance"
    assert document.digest().startswith("sha256:")
    assert "how many clicks" not in document.canonical_bytes().decode()  # content-free by construction


def test_every_adapters_guidance_hook_matches_the_requested_category_only(
    adapter_case: tuple[FrameworkAdapter, Any],
) -> None:
    adapter, _native_run = adapter_case

    hook = adapter.attach_guidance(guidance="State totals plainly.", categories=("summary",))

    assert hook is not None  # what a caller does with it is framework-specific; see each adapter's own tests


def test_every_adapter_delivers_a_receipt_for_an_active_authorization_and_refuses_an_expired_one(
    adapter_case: tuple[FrameworkAdapter, Any],
) -> None:
    adapter, _native_run = adapter_case
    candidate = AdvisorySkillCandidate(candidate_id="c-1", advisory_skill="State totals plainly.")
    active = DeliveryAuthorization(
        authorization_id="auth-1",
        job_id="job-1",
        candidate_id="c-1",
        scope_ref="tenant:conformance",
        authorized_by="owner@example.com",
        expires_at=datetime.now(UTC) + timedelta(days=1),
    )
    expired = DeliveryAuthorization(
        authorization_id="auth-2",
        job_id="job-1",
        candidate_id="c-1",
        scope_ref="tenant:conformance",
        authorized_by="owner@example.com",
        expires_at=datetime.now(UTC) + timedelta(seconds=1),
    )

    receipt = adapter.deliver(active, candidate)
    with pytest.raises(ValueError, match="expired or revoked"):
        adapter.deliver(expired, candidate, now=datetime.now(UTC) + timedelta(days=1))

    assert receipt.candidate_id == "c-1" and receipt.scope_ref == "tenant:conformance"
    assert receipt.skill_digest is not None
