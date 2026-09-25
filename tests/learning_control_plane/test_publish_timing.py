from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from learning_control_plane.contracts.investigation import InvestigationTrajectoryV1, SourceTraceRef
from learning_control_plane.control_plane import LearningControlPlane, PromotionPolicy
from learning_control_plane.control_plane.worker import OfflineEvaluationWorker
from learning_control_plane.evaluation import LocalEvaluationBackend
from learning_control_plane.evaluation.verification import (
    InvestigationVerification,
    SafeStepEvidence,
    VerificationCheck,
    score_final_answer,
)
from learning_control_plane.integrations.generic import RunContext, RunPublisher
from learning_control_plane.integrations.penguiflow import PlannerTraceReadiness, TurnStash
from learning_control_plane.providers.assessment_publisher import (
    MlflowAssessmentPublisher,
    MlflowTraceReadiness,
    PendingAssessmentQueue,
)
from penguiflow.planner.models import PlannerAction
from penguiflow.planner.trajectory import Trajectory, TrajectoryStep


def _document(trace_id: str = "trace-1", investigation_id: str = "investigation-1") -> InvestigationTrajectoryV1:
    criteria = {
        name: VerificationCheck(name, "passed")
        for name in (
            "factual_numerical_correctness",
            "scope_correctness",
            "evidence_grounding",
            "completeness",
            "interpretation_correctness",
        )
    }
    verification = InvestigationVerification(
        step_evidence=(
            SafeStepEvidence(0, "query_stock", result_checks=(VerificationCheck("tool_execution", "passed"),)),
        ),
        final_answer=score_final_answer(criteria),
    )
    return InvestigationTrajectoryV1(
        investigation_id=investigation_id,
        source_trace_ref=SourceTraceRef("local", "exp-1", trace_id, "sha256:inventory-v1"),
        agent_ref="inventory_agent",
        provider_ref="plain-python",
        scope_ref="tenant:demo",
        started_at=datetime(2026, 9, 1, tzinfo=UTC),
        status="completed",
        execution_fingerprint="sha256:inventory-v1",
        request={"has_text": True},
        steps=({"node": "query_stock"},),
        redaction_profile="safe:v1",
        step_signature="query_stock",
        execution_context={"verified_success": True},
        assessment_refs=(verification.final_answer.assessment_ref,) if verification.final_answer else (),
        extensions={"learning.verification": verification.record()},
    )


@dataclass
class _Info:
    state: Any


class _SlowTraceMlflow:
    """A trace that is missing for `missing_polls` polls, then in progress for one, then closed."""

    AssessmentSource = None

    def __init__(self, missing_polls: int, *, feedback_fails: bool = False) -> None:
        self._states = iter(["missing"] * missing_polls + ["IN_PROGRESS", "OK"])
        self.polls = 0
        self.feedback: list[dict[str, Any]] = []
        self._feedback_fails = feedback_fails

    def get_trace(self, trace_id: str) -> Any:
        self.polls += 1
        state = next(self._states, "OK")
        if state == "missing":
            raise RuntimeError(f"Trace with ID {trace_id} not found")
        return SimpleNamespace(info=_Info(state=SimpleNamespace(value=state)))

    def log_feedback(self, **kwargs: Any) -> Any:
        if self._feedback_fails:
            raise RuntimeError("NOT_FOUND: Trace with ID trace-1 not found")
        self.feedback.append(kwargs)
        return SimpleNamespace(assessment_id=f"assessment-{len(self.feedback)}")


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def test_readiness_waits_until_the_trace_appears_and_is_closed() -> None:
    mlflow = _SlowTraceMlflow(missing_polls=3)
    clock = _Clock()

    closed = MlflowTraceReadiness(mlflow_module=mlflow, sleep_fn=clock.sleep, clock=clock).wait_closed("trace-1", 10)

    assert closed
    assert mlflow.polls == 5


def test_readiness_gives_up_after_its_timeout() -> None:
    mlflow = _SlowTraceMlflow(missing_polls=1_000)
    clock = _Clock()

    closed = MlflowTraceReadiness(
        mlflow_module=mlflow, poll_interval_s=1.0, sleep_fn=clock.sleep, clock=clock
    ).wait_closed("trace-1", 3)

    assert not closed
    assert mlflow.polls == 4


def test_a_trace_that_never_closes_queues_the_assessment_and_the_worker_publishes_it_later(tmp_path: Path) -> None:
    queue = PendingAssessmentQueue(tmp_path / "pending.jsonl")
    clock = _Clock()
    early = _SlowTraceMlflow(missing_polls=1_000)
    publisher = MlflowAssessmentPublisher(
        mlflow_module=early,
        readiness=MlflowTraceReadiness(mlflow_module=early, sleep_fn=clock.sleep, clock=clock),
        readiness_timeout_s=2,
        pending_queue=queue,
    )

    assert publisher.publish(_document()) == ()
    assert early.feedback == []
    assert [document.investigation_id for document in queue.pending()] == ["investigation-1"]

    later = _SlowTraceMlflow(missing_polls=0)
    drain = queue.drain(MlflowAssessmentPublisher(mlflow_module=later).publish_now)

    assert drain.published == ("investigation-1",)
    assert queue.pending() == ()
    assert later.feedback


def test_retries_that_never_find_the_trace_queue_instead_of_raising(tmp_path: Path) -> None:
    queue = PendingAssessmentQueue(tmp_path / "pending.jsonl")
    publisher = MlflowAssessmentPublisher(
        mlflow_module=_SlowTraceMlflow(0, feedback_fails=True),
        trace_availability_delays=(0.0,),
        sleep_fn=lambda _: None,
        pending_queue=queue,
    )

    assert publisher.publish(_document()) == ()
    assert len(queue.pending()) == 1


def test_without_a_queue_a_missing_trace_still_raises_after_the_retries() -> None:
    publisher = MlflowAssessmentPublisher(
        mlflow_module=_SlowTraceMlflow(0, feedback_fails=True),
        trace_availability_delays=(0.0,),
        sleep_fn=lambda _: None,
    )

    with pytest.raises(RuntimeError, match="not found"):
        publisher.publish(_document())


def test_a_readiness_timeout_without_a_queue_falls_back_to_the_retries() -> None:
    clock = _Clock()
    mlflow = _SlowTraceMlflow(missing_polls=1_000)
    publisher = MlflowAssessmentPublisher(
        mlflow_module=mlflow,
        readiness=MlflowTraceReadiness(mlflow_module=mlflow, sleep_fn=clock.sleep, clock=clock),
        readiness_timeout_s=1,
    )

    assert publisher.publish(_document())


def test_a_document_is_queued_once_and_a_failing_drain_keeps_it(tmp_path: Path) -> None:
    queue = PendingAssessmentQueue(tmp_path / "nested" / "pending.jsonl")
    queue.put(_document())
    queue.put(_document())
    queue.put(_document("trace-2", "investigation-2"))

    def publish(document: InvestigationTrajectoryV1) -> tuple[str, ...]:
        if document.investigation_id == "investigation-2":
            raise RuntimeError("trace still missing")
        return ("assessment",)

    drain = queue.drain(publish)

    assert drain.published == ("investigation-1",)
    assert drain.still_pending == ("investigation-2",)
    assert [document.investigation_id for document in queue.pending()] == ["investigation-2"]


def test_an_empty_queue_drains_to_nothing(tmp_path: Path) -> None:
    drain = PendingAssessmentQueue(tmp_path / "missing.jsonl").drain(lambda document: ())

    assert drain.published == ()
    assert drain.still_pending == ()


def test_the_worker_drains_pending_assessments_on_its_next_pass(tmp_path: Path) -> None:
    queue = PendingAssessmentQueue(tmp_path / "pending.jsonl")
    queue.put(_document())
    mlflow = _SlowTraceMlflow(0)
    plane = LearningControlPlane(
        policy=PromotionPolicy(policy_version="v1", primary_metric="quality"),
        evaluation_backend=LocalEvaluationBackend(),
    )
    worker = OfflineEvaluationWorker(
        plane,
        run_one=lambda case, variant: None,
        metric=lambda case, output: {},
        pending_assessments=queue,
        assessment_publisher=MlflowAssessmentPublisher(mlflow_module=mlflow),
    )

    run = asyncio.run(worker.run_pending())

    assert run.published_pending_assessments == ("investigation-1",)
    assert mlflow.feedback


def test_the_worker_needs_both_the_queue_and_a_publisher(tmp_path: Path) -> None:
    plane = LearningControlPlane(
        policy=PromotionPolicy(policy_version="v1", primary_metric="quality"),
        evaluation_backend=LocalEvaluationBackend(),
    )

    with pytest.raises(ValueError, match="both the queue and an assessment publisher"):
        OfflineEvaluationWorker(
            plane,
            run_one=lambda case, variant: None,
            metric=lambda case, output: {},
            pending_assessments=PendingAssessmentQueue(tmp_path / "q.jsonl"),
        )


CONTEXT = RunContext(
    source_trace_ref=SourceTraceRef("local", "exp-1", "mlflow-trace-1", "sha256:inventory-v1"),
    agent_ref="inventory_agent",
    scope_ref="tenant:demo",
    execution_fingerprint="sha256:inventory-v1",
    started_at=datetime(2026, 9, 1, tzinfo=UTC),
    allowed_node_names=frozenset({"query_stock"}),
)


class _Publisher:
    def __init__(self, error: Exception | None = None) -> None:
        self.documents: list[InvestigationTrajectoryV1] = []
        self._error = error

    def publish(self, document: InvestigationTrajectoryV1) -> str:
        if self._error:
            raise self._error
        self.documents.append(document)
        return document.digest()


def _trajectory(trace_id: str | None = "native-1") -> Trajectory:
    return Trajectory(
        query="How many units?",
        tool_context={"trace_id": trace_id} if trace_id else {},
        final_answer="draft answer",
        finish_reason="answer_complete",
        steps=[TrajectoryStep(action=PlannerAction(next_node="query_stock", args={}), observation={"units": 5})],
    )


def test_a_held_trajectory_is_published_with_the_answer_the_user_received() -> None:
    publisher = _Publisher()
    captured: list[Any] = []

    class _Recording(RunPublisher):
        async def publish_after_turn(self, run: Any, context: RunContext, **kwargs: Any) -> Any:
            captured.append(run)
            return await super().publish_after_turn(run, context, **kwargs)

    stash = TurnStash(_Recording(publisher), lambda trajectory: CONTEXT)

    assert stash.hold(_trajectory())
    publication = asyncio.run(stash.release("native-1", "final answer: 5 units."))

    assert publication is not None
    assert captured[0].final_answer == "final answer: 5 units."
    assert len(publisher.documents) == 1
    assert asyncio.run(stash.release("native-1", "again")) is None


def test_a_trajectory_without_a_trace_id_is_not_held() -> None:
    stash = TurnStash(RunPublisher(_Publisher()), lambda trajectory: CONTEXT)

    assert not stash.hold(_trajectory(trace_id=None))


def test_a_publisher_failure_never_reaches_the_turn() -> None:
    def broken_context(trajectory: Trajectory) -> RunContext:
        raise RuntimeError("experiment not configured")

    stash = TurnStash(RunPublisher(_Publisher()), broken_context)
    stash.hold(_trajectory())

    assert asyncio.run(stash.release("native-1", "answer.")) is None


def test_a_run_without_a_context_is_not_published() -> None:
    publisher = _Publisher()
    stash = TurnStash(RunPublisher(publisher), lambda trajectory: None)
    stash.hold(_trajectory())

    assert asyncio.run(stash.release("native-1", "answer.")) is None
    assert publisher.documents == []


def test_the_stash_drops_the_oldest_turn_when_full_and_forgets_discarded_turns() -> None:
    publisher = _Publisher()
    stash = TurnStash(RunPublisher(publisher), lambda trajectory: CONTEXT, max_held=1)
    stash.hold(_trajectory("native-1"))
    stash.hold(_trajectory("native-2"))
    stash.discard("native-2")

    assert asyncio.run(stash.release("native-1", "answer.")) is None
    assert asyncio.run(stash.release("native-2", "answer.")) is None


def test_a_stash_must_hold_at_least_one_turn() -> None:
    with pytest.raises(ValueError, match="max_held"):
        TurnStash(RunPublisher(_Publisher()), lambda trajectory: CONTEXT, max_held=0)


@pytest.fixture
def planner_loop() -> Iterator[asyncio.AbstractEventLoop]:
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join()
    loop.close()


class _Planner:
    def __init__(self, persisted: bool) -> None:
        self.persisted = persisted
        self.waited: list[str] = []

    async def wait_for_trace_persistence(self, trace_id: str, *, timeout_s: float) -> bool:
        self.waited.append(trace_id)
        return self.persisted


class _Ready:
    def __init__(self) -> None:
        self.calls: Sequence[str] = []

    def wait_closed(self, trace_id: str, timeout_s: float) -> bool:
        self.calls = [*self.calls, trace_id]
        return True


def test_planner_readiness_waits_for_the_planners_own_persistence_then_mlflow(
    planner_loop: asyncio.AbstractEventLoop,
) -> None:
    planner = _Planner(persisted=True)
    mlflow_ready = _Ready()

    ready = PlannerTraceReadiness(
        planner, planner_loop, then=mlflow_ready, native_trace_id=lambda trace_id: f"native-{trace_id}"
    ).wait_closed("trace-1", 2.0)

    assert ready
    assert planner.waited == ["native-trace-1"]
    assert mlflow_ready.calls == ["trace-1"]


def test_planner_readiness_is_false_when_the_planner_did_not_persist(
    planner_loop: asyncio.AbstractEventLoop,
) -> None:
    assert not PlannerTraceReadiness(_Planner(persisted=False), planner_loop).wait_closed("trace-1", 1.0)


def test_planner_readiness_without_a_next_check_is_done_once_persisted(
    planner_loop: asyncio.AbstractEventLoop,
) -> None:
    assert PlannerTraceReadiness(_Planner(persisted=True), planner_loop).wait_closed("trace-1", 1.0)


def test_planner_readiness_treats_a_broken_wait_as_not_ready(planner_loop: asyncio.AbstractEventLoop) -> None:
    class _Broken:
        async def wait_for_trace_persistence(self, trace_id: str, *, timeout_s: float) -> bool:
            raise RuntimeError("store down")

    assert not PlannerTraceReadiness(_Broken(), planner_loop).wait_closed("trace-1", 1.0)


def test_real_mlflow_readiness_sees_a_closed_trace_and_not_an_unknown_one(tmp_path: Path) -> None:
    mlflow = pytest.importorskip("mlflow")
    previous_tracking_uri = mlflow.get_tracking_uri()
    try:
        mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
        experiment_id = mlflow.create_experiment("lcp-readiness-test", artifact_location=(tmp_path / "a").as_uri())
        from mlflow.tracing.destination import MlflowExperimentLocation

        # An explicit destination, not set_experiment, so no active experiment leaks into other tests.
        with mlflow.start_span(name="agent.turn", trace_destination=MlflowExperimentLocation(experiment_id)):
            trace_id = mlflow.get_active_trace_id()
        mlflow.flush_trace_async_logging()
        readiness = MlflowTraceReadiness(poll_interval_s=0.05)

        assert readiness.wait_closed(trace_id, 5.0)
        assert not readiness.wait_closed("tr-does-not-exist", 0.1)
    finally:
        mlflow.set_tracking_uri(previous_tracking_uri)
