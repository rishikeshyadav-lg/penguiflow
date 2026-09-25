from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import pytest

from learning_control_plane.contracts.investigation import InvestigationTrajectoryV1, SourceTraceRef
from learning_control_plane.evaluation.verification import SafeStepEvidence, VerificationCheck
from learning_control_plane.integrations.generic import RunContext, RunPublisher, project_run
from learning_control_plane.integrations.penguiflow.projector import agent_run_from_trajectory
from learning_control_plane.judging import (
    AgentRun,
    AgentStep,
    MeaningCheck,
    MeaningQuestion,
    OutcomeLadder,
    RubricJudgment,
    SignatureRules,
)
from penguiflow.planner.models import PlannerAction
from penguiflow.planner.trajectory import Trajectory, TrajectoryStep

PASSED = RubricJudgment(
    numerical=VerificationCheck("factual_numerical_correctness", "passed", ("stated_values_match_reference",)),
    completeness=VerificationCheck("completeness", "passed", ("all_required_values_stated",)),
    scope=VerificationCheck("scope_correctness", "passed", ("requested_scope_present",)),
    grounding=VerificationCheck("evidence_grounding", "passed", ("values_match_independent_reference",)),
)
CONTEXT = RunContext(
    source_trace_ref=SourceTraceRef(
        tracking_store_ref="local",
        experiment_id="exp-1",
        mlflow_trace_id="trace-1",
        deployment_ref="sha256:inventory-v1",
    ),
    agent_ref="inventory_agent",
    scope_ref="tenant:demo",
    execution_fingerprint="sha256:inventory-v1",
    started_at=datetime(2026, 9, 1, tzinfo=UTC),
    allowed_node_names=frozenset({"query_stock", "list_stores"}),
)


def _run(answer: str | None = "North Store sold 1,200 units.") -> AgentRun:
    return AgentRun(
        question="How many units did North Store sell?",
        steps=(
            AgentStep("list_stores", {"prefix": "North"}, {"values": ["North Store"]}, streamed=True),
            AgentStep("query_stock", {"store": "North Store"}, error="timeout"),
            AgentStep("query_stock", {"store": "North Store"}, {"units": 1200}),
            AgentStep("secret_tool", {"token": "abc"}, {"raw": "content"}),
        ),
        final_answer=answer,
    )


class _InventoryJudge:
    def __init__(self) -> None:
        self.references: list[Any] = []

    def step_evidence(self, run: AgentRun) -> Sequence[SafeStepEvidence]:
        return [
            SafeStepEvidence(index, step.tool, result_checks=(VerificationCheck("tool_execution", "passed"),))
            for index, step in enumerate(run.steps)
            if step.tool in CONTEXT.allowed_node_names
        ]

    def clarification_candidates(self, run: AgentRun) -> Sequence[str]:
        return ()

    def service_unavailable(self, run: AgentRun) -> bool:
        return False

    def judge(self, run: AgentRun, reference: Any) -> RubricJudgment | None:
        self.references.append(reference)
        return PASSED


class _Publisher:
    def __init__(self, error: Exception | None = None) -> None:
        self.documents: list[InvestigationTrajectoryV1] = []
        self._error = error

    def publish(self, document: InvestigationTrajectoryV1) -> str:
        if self._error is not None:
            raise self._error
        self.documents.append(document)
        return document.digest()


class _AssessmentPublisher:
    def __init__(self, error: Exception | None = None) -> None:
        self._error = error

    def publish(self, document: InvestigationTrajectoryV1) -> tuple[str, ...]:
        if self._error is not None:
            raise self._error
        return ("assessment-1",)


def test_a_projected_run_keeps_only_allowlisted_names_and_no_content() -> None:
    document = project_run(_run(), CONTEXT)

    assert [step["node"] for step in document.steps] == ["list_stores", "query_stock", "query_stock", "redacted_node"]
    assert [step["status"] for step in document.steps] == ["completed", "failed", "completed", "completed"]
    assert document.steps[0]["has_streams"] is True
    assert document.status == "completed"
    assert document.request == {"has_text": True, "input_part_count": 0}
    assert document.execution_context["verified_success"] is False
    for secret in ("North Store", "abc", "content", "1,200"):
        assert secret not in document.canonical_bytes().decode()


def test_a_judged_run_carries_its_verification_and_assessment_reference() -> None:
    ladder = OutcomeLadder(_InventoryJudge())

    document = project_run(_run(), CONTEXT, judge=ladder.judge)

    assert document.extensions["learning.verification"]["outcome"] == "verified"
    assert document.execution_context["verified_success"] is True
    assert document.steps[2]["verified"] is True
    assert len(document.assessment_refs) == 1


def test_a_judge_that_raises_still_leaves_a_document() -> None:
    def broken(run: AgentRun) -> Any:
        raise RuntimeError("judge bug")

    document = project_run(_run(), CONTEXT, judge=broken)

    assert "learning.verification" not in document.extensions


def test_signature_rules_decide_the_step_signature() -> None:
    document = project_run(_run(), CONTEXT, signature=SignatureRules(renamed={"list_stores": "lookup"}))

    assert document.step_signature == "lookup>query_stock>redacted_node"


def test_an_unknown_finish_reason_is_an_unknown_status() -> None:
    run = AgentRun(question="", steps=(), final_answer=None, finish_reason="exploded")

    document = project_run(run, CONTEXT)

    assert document.status == "unknown"
    assert document.termination_reason == "unknown"
    assert document.step_signature == "no_steps"


def test_the_publisher_judges_with_the_reference_and_publishes_the_document() -> None:
    domain = _InventoryJudge()
    publisher = _Publisher()

    async def reference(run: AgentRun) -> Mapping[str, float]:
        return {"units": 1200.0}

    publication = asyncio.run(
        RunPublisher(
            publisher,
            judge=OutcomeLadder(domain),
            assessment_publisher=_AssessmentPublisher(),
            reference_builder=reference,
        ).publish_after_turn(_run(), CONTEXT)
    )

    assert publication is not None
    assert publication.assessment_ids == ("assessment-1",)
    assert publication.document.execution_context["verified_success"] is True
    assert domain.references == [{"units": 1200.0}]
    assert publisher.documents == [publication.document]


def test_the_publisher_skips_a_run_without_a_final_answer() -> None:
    publisher = _Publisher()

    publication = asyncio.run(RunPublisher(publisher).publish_after_turn(_run(answer=None), CONTEXT))

    assert publication is None
    assert publisher.documents == []


def test_a_slow_reference_times_out_and_the_judge_falls_back() -> None:
    domain = _InventoryJudge()

    async def slow_reference(run: AgentRun) -> Mapping[str, float]:
        await asyncio.sleep(5)
        return {}

    publication = asyncio.run(
        RunPublisher(
            _Publisher(), judge=OutcomeLadder(domain), reference_builder=slow_reference, reference_timeout_s=0.01
        ).publish_after_turn(_run(), CONTEXT)
    )

    assert publication is not None
    assert domain.references == [None]


def test_meaning_findings_are_computed_after_the_turn_and_can_fail_the_run() -> None:
    class _Judge:
        def read(self, state: Mapping[str, str], questions: Sequence[MeaningQuestion]) -> Mapping[str, float]:
            return {"contradiction": 0.99}

    publication = asyncio.run(
        RunPublisher(
            _Publisher(), judge=OutcomeLadder(_InventoryJudge()), meaning=MeaningCheck(_Judge(), reads=1)
        ).publish_after_turn(_run(), CONTEXT)
    )

    assert publication is not None
    assert publication.document.extensions["learning.verification"]["outcome"] == "failed"


def test_a_meaning_check_that_times_out_is_skipped() -> None:
    class _SlowJudge:
        def read(self, state: Mapping[str, str], questions: Sequence[MeaningQuestion]) -> Mapping[str, float]:
            import time

            time.sleep(0.2)
            return {"contradiction": 0.99}

    publication = asyncio.run(
        RunPublisher(
            _Publisher(),
            judge=OutcomeLadder(_InventoryJudge()),
            meaning=MeaningCheck(_SlowJudge(), reads=1),
            meaning_timeout_s=0.01,
        ).publish_after_turn(_run(), CONTEXT)
    )

    assert publication is not None
    assert publication.document.extensions["learning.verification"]["outcome"] == "verified"


def test_a_failing_assessment_publisher_still_publishes_the_document() -> None:
    publisher = _Publisher()

    publication = asyncio.run(
        RunPublisher(
            publisher, judge=OutcomeLadder(_InventoryJudge()), assessment_publisher=_AssessmentPublisher(OSError("x"))
        ).publish_after_turn(_run(), CONTEXT)
    )

    assert publication is not None
    assert publication.assessment_ids == ()
    assert len(publisher.documents) == 1


def test_a_failing_document_publisher_never_raises_into_the_turn() -> None:
    publication = asyncio.run(
        RunPublisher(_Publisher(ConnectionError("store down"))).publish_after_turn(_run(), CONTEXT)
    )

    assert publication is None


def test_a_penguiflow_trajectory_becomes_a_run_with_its_rendered_output() -> None:
    trajectory = Trajectory(
        query="How many units?",
        final_answer={"text": "1,200"},
        finish_reason=None,
        steps=[
            TrajectoryStep(action=PlannerAction(next_node="query_stock", args={"store": "North"}), failure={"x": 1}),
            TrajectoryStep(action=PlannerAction(next_node="render_table", args={"rows": [{"units": 1200}]})),
        ],
    )

    run = agent_run_from_trajectory(trajectory)

    assert run.steps[0].error == "failed"
    assert run.rendered[0].kind == "table"
    assert run.final_answer == "{'text': '1,200'}"
    assert run.finish_reason == "unknown"


@pytest.mark.parametrize("finish_reason", ["budget_exhausted", "no_path"])
def test_a_run_that_did_not_complete_keeps_its_terminal_reason(finish_reason: str) -> None:
    run = AgentRun(question="q", steps=(), final_answer=None, finish_reason=finish_reason)

    assert project_run(run, CONTEXT).termination_reason == finish_reason
