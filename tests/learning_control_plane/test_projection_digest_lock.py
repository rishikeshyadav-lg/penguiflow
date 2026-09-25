"""The PenguiFlow projector's output is locked byte for byte across its move onto the generic projector."""

from __future__ import annotations

from datetime import UTC, datetime

from learning_control_plane.contracts.investigation import SourceTraceRef
from learning_control_plane.evaluation.verification import (
    InvestigationVerification,
    SafeStepEvidence,
    VerificationCheck,
    score_final_answer,
)
from learning_control_plane.integrations.penguiflow.projector import (
    PenguiFlowInvestigationContext,
    PenguiFlowInvestigationProjector,
)
from penguiflow.planner.models import PlannerAction
from penguiflow.planner.trajectory import Trajectory, TrajectoryStep

# Digest of the document the projector produced for `_fixed_trajectory` before the generic projector existed.
LOCKED_DIGEST = "sha256:2932250d9cb78de827cea3ecabef193408773cc581de87ee0750f17e83338a1f"


def _verification(trajectory: Trajectory) -> InvestigationVerification:
    checks = {
        criterion: VerificationCheck(criterion, "passed", ("synthetic_check",))
        for criterion in (
            "factual_numerical_correctness",
            "scope_correctness",
            "evidence_grounding",
            "completeness",
            "interpretation_correctness",
        )
    }
    return InvestigationVerification(
        step_evidence=[
            SafeStepEvidence(
                step_index=index,
                node_name=step.action.next_node if step.action.next_node != "unlisted_tool" else "redacted_node",
                argument_facts={"arg_count": len(step.action.args)},
                decision_reason_codes=("synthetic_reason",),
                result_checks=(VerificationCheck("tool_execution", "failed" if step.error else "passed"),),
            )
            for index, step in enumerate(trajectory.steps)
        ],
        final_answer=score_final_answer(checks),
    )


def _fixed_trajectory() -> Trajectory:
    return Trajectory(
        query="How many units did the north store sell?",
        final_answer="The north store sold 1,200 units.",
        finish_reason="answer_complete",
        steps=[
            TrajectoryStep(
                action=PlannerAction(next_node="list_stores", args={"prefix": "north"}),
                observation={"values": ["North Store"]},
                streams={"progress": ({"chunk": "1"},)},
            ),
            TrajectoryStep(
                action=PlannerAction(next_node="query_stock", args={"field": "colour"}),
                error="unknown field",
            ),
            TrajectoryStep(
                action=PlannerAction(next_node="parallel", args={}),
                observation={
                    "branches": [
                        {"node": "query_stock", "args": {"store": "North"}, "observation": {"units": 1200}},
                        {"node": "unlisted_tool", "args": {}, "error": "boom", "failure": {"code": "x"}},
                    ],
                    "join": {"node": "merge_results", "args": {}, "observation": {"ok": True}},
                },
            ),
            TrajectoryStep(action=PlannerAction(next_node="render_table", args={"rows": []}), observation=None),
        ],
    )


def _context() -> PenguiFlowInvestigationContext:
    return PenguiFlowInvestigationContext(
        source_trace_ref=SourceTraceRef(
            tracking_store_ref="mlflow://local",
            experiment_id="7",
            mlflow_trace_id="tr-lock-1",
            deployment_ref="sha256:agent-v1",
            native_trace_id="native-lock-1",
        ),
        agent_ref="inventory_agent",
        scope_ref="tenant:demo",
        execution_fingerprint="sha256:agent-v1",
        started_at=datetime(2026, 9, 1, 12, tzinfo=UTC),
        allowed_node_names=frozenset({"list_stores", "query_stock", "merge_results", "render_table"}),
        intent_descriptor={"class": "stock_level"},
        verification_projector=_verification,
        signature_normalizer=lambda steps: [str(step["node"]) for step in steps if step["status"] == "completed"],
    )


def test_the_penguiflow_projection_of_a_fixed_trajectory_keeps_its_digest() -> None:
    document = PenguiFlowInvestigationProjector(_context()).project(
        _fixed_trajectory(), completed_at=datetime(2026, 9, 1, 12, 5, tzinfo=UTC)
    )

    assert document.digest() == LOCKED_DIGEST


def test_the_penguiflow_projection_keeps_its_generated_investigation_id() -> None:
    document = PenguiFlowInvestigationProjector(_context()).project(_fixed_trajectory())

    assert document.investigation_id.startswith("investigation_")
    assert len(document.investigation_id) == len("investigation_") + 24
