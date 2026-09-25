from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from learning_control_plane.contracts.investigation import InvestigationTrajectoryV1, SourceTraceRef
from learning_control_plane.evaluation.verification import (
    InvestigationVerification,
    SafeStepEvidence,
    VerificationCheck,
    score_final_answer,
)
from learning_control_plane.mining.investigation_mining import InvestigationSelection, MlflowInvestigationReader
from learning_control_plane.providers.assessment_publisher import MlflowAssessmentPublisher
from learning_control_plane.providers.investigation_publisher import MlflowAttachmentPublisher
from learning_control_plane.providers.verdict_revision import (
    RevisionReceipt,
    revise_verdict,
    revised_investigation,
)

_CRITERIA = (
    "factual_numerical_correctness",
    "scope_correctness",
    "evidence_grounding",
    "completeness",
    "interpretation_correctness",
)
_REVISED_AT = datetime(2026, 9, 25, 9, 0, tzinfo=UTC)


def _verification(*, passed: bool) -> InvestigationVerification:
    status = "passed" if passed else "failed"
    assessment = score_final_answer(
        {criterion: VerificationCheck(criterion, status, (f"answer_{status}",)) for criterion in _CRITERIA},
        hard_failure_codes=() if passed else ("primary_result_incorrect",),
    )
    return InvestigationVerification(
        step_evidence=(
            SafeStepEvidence(
                step_index=0,
                node_name="lookup_stock",
                argument_facts={"filter_count": 1},
                result_checks=(VerificationCheck("tool_execution", "passed"),),
            ),
        ),
        final_answer=assessment,
    )


def _published_investigation(*, experiment_id: str = "experiment-7", passed: bool = False) -> InvestigationTrajectoryV1:
    verification = _verification(passed=passed)
    assert verification.final_answer is not None
    return InvestigationTrajectoryV1(
        investigation_id="investigation-a1",
        source_trace_ref=SourceTraceRef(
            tracking_store_ref="mlflow-test",
            experiment_id=experiment_id,
            mlflow_trace_id="source-trace-a1",
            deployment_ref="sha256:inventory-agent-v1",
        ),
        agent_ref="inventory_agent",
        provider_ref="plain-python:v1",
        scope_ref="tenant:demo",
        started_at=datetime(2026, 9, 20, 12, 0, tzinfo=UTC),
        status="completed",
        execution_fingerprint="sha256:execution-v1",
        request={"has_text": True, "input_part_count": 1},
        steps=(
            {"index": 0, "node": "lookup_stock", "status": "completed", "verified": True, "result_checks": []},
            {"index": 1, "node": "summarize", "status": "completed"},
        ),
        redaction_profile="safe:v1",
        step_signature="lookup_stock>summarize",
        execution_context={"step_count": 2, "failed_step_count": 0, "verified_success": verification.verified_success},
        assessment_refs=(verification.final_answer.assessment_ref,),
        extensions={"learning.verification": verification.record()},
    )


@dataclass
class _TraceInfo:
    trace_id: str
    tags: dict[str, str]


@dataclass
class _Span:
    outputs: dict[str, Any] = field(default_factory=dict)

    def set_outputs(self, outputs: dict[str, Any]) -> None:
        self.outputs = outputs


@dataclass
class _TraceData:
    spans: list[_Span]


@dataclass
class _Trace:
    info: _TraceInfo
    data: _TraceData


class _SpanContext:
    def __init__(self, mlflow: _FakeMlflow) -> None:
        self._mlflow = mlflow
        self.span = _Span()

    def __enter__(self) -> _Span:
        self._mlflow.open_span = self.span
        return self.span

    def __exit__(self, *_: object) -> None:
        self._mlflow.open_span = None


@dataclass
class _Attachment:
    content_type: str
    content_bytes: bytes


class _FakeMlflow:
    """An in-memory MLflow: tag-filtered trace search, retagging, attachments and feedback."""

    def __init__(self) -> None:
        self.traces: list[_Trace] = []
        self.attachments: dict[str, bytes] = {}
        self.feedback: list[dict[str, Any]] = []
        self.tag_calls: list[tuple[str, str, str]] = []
        self.open_span: _Span | None = None
        self.fail_on_tag: str | None = None

    def search_traces(self, *, locations: Sequence[str], filter_string: str, return_type: str) -> list[_Trace]:
        del locations, return_type
        match = re.fullmatch(r'tags\.`(?P<name>[^`]+)` = "(?P<value>[^"]*)"', filter_string)
        assert match is not None, filter_string
        return [trace for trace in self.traces if trace.info.tags.get(match["name"]) == match["value"]]

    def set_trace_tag(self, trace_id: str, key: str, value: str) -> None:
        if key == self.fail_on_tag:
            raise ConnectionError("simulated crash while retagging")
        self.tag_calls.append((trace_id, key, value))
        next(trace for trace in self.traces if trace.info.trace_id == trace_id).info.tags[key] = value

    def start_span(self, *, name: str, span_type: str, trace_destination: object) -> _SpanContext:
        del name, span_type, trace_destination
        return _SpanContext(self)

    def update_current_trace(self, *, tags: dict[str, str]) -> None:
        assert self.open_span is not None
        trace_id = f"trace-{len(self.traces)}"
        self.traces.append(_Trace(_TraceInfo(trace_id, dict(tags)), _TraceData([self.open_span])))

    def log_feedback(self, **feedback: Any) -> None:
        self.feedback.append(feedback)

    def attachment(self, *, content_type: str, content_bytes: bytes) -> str:
        attachment_id = f"attachment-{len(self.attachments)}"
        self.attachments[attachment_id] = content_bytes
        return f"attachment://{attachment_id}"

    def download(self, trace: _Trace, attachment_id: str) -> bytes:
        del trace
        return self.attachments[attachment_id]


def _publisher(mlflow: _FakeMlflow) -> MlflowAttachmentPublisher:
    return MlflowAttachmentPublisher(
        mlflow_module=mlflow,
        attachment_factory=mlflow.attachment,
        trace_destination_factory=lambda experiment_id: experiment_id,
    )


def _reader(mlflow: _FakeMlflow) -> MlflowInvestigationReader:
    return MlflowInvestigationReader(
        mlflow_module=mlflow,
        attachment_downloader=mlflow,
        attachment_reference_parser=lambda reference: {"attachment_id": reference.rsplit("/", 1)[1]},
    )


def _revise(
    mlflow: _FakeMlflow, original: InvestigationTrajectoryV1, verification: InvestigationVerification
) -> RevisionReceipt:
    return revise_verdict(
        original,
        verification,
        reason="judge_fix_scope_suffix",
        judge_version="judge.v7",
        trace_store=mlflow,
        publisher=_publisher(mlflow),
        assessment_publisher=MlflowAssessmentPublisher(mlflow_module=mlflow, trace_availability_delays=()),
        now=_REVISED_AT,
    )


def _status_of(mlflow: _FakeMlflow, investigation_id: str) -> str:
    [trace] = mlflow.search_traces(
        locations=[], filter_string=f'tags.`learning.investigation.id` = "{investigation_id}"', return_type="list"
    )
    return trace.info.tags["learning.investigation.status"]


def test_revised_document_carries_the_new_verdict_and_names_what_it_supersedes() -> None:
    original = _published_investigation(passed=False)

    revised = revised_investigation(
        original, _verification(passed=True), reason="judge_fix", judge_version="judge.v7", revised_at=_REVISED_AT
    )

    assert revised.investigation_id == "investigation-a1:rev1"
    assert revised.execution_context["verified_success"] is True
    assert revised.extensions["learning.verification"]["verified_success"] is True
    assert revised.assessment_refs != original.assessment_refs
    assert revised.revision is not None
    assert revised.revision["supersedes"] == "investigation-a1"
    assert revised.revision["supersedes_digest"] == original.digest()
    assert revised.revision["revised_at"] == "2026-09-25T09:00:00.000000Z"
    assert revised.query_index()["learning.investigation.supersedes"] == "investigation-a1"
    assert revised.source_trace_ref == original.source_trace_ref


def test_revised_document_rewrites_step_evidence_from_the_new_verification() -> None:
    revised = revised_investigation(
        _published_investigation(),
        _verification(passed=True),
        reason="judge_fix",
        judge_version="judge.v7",
        revised_at=_REVISED_AT,
    )

    assert revised.steps[0]["argument_facts"] == {"filter_count": 1}
    assert revised.steps[0]["result_checks"] == [{"check_id": "tool_execution", "status": "passed", "reason_codes": []}]
    assert "verified" not in revised.steps[1]


def test_an_unrevised_document_keeps_its_original_query_index() -> None:
    original = _published_investigation()

    assert original.revision is None
    assert "learning.investigation.supersedes" not in original.query_index()


def test_revision_rejects_evidence_for_steps_the_investigation_does_not_have() -> None:
    verification = InvestigationVerification(
        step_evidence=(SafeStepEvidence(step_index=5, node_name="lookup_stock"),),
    )

    with pytest.raises(ValueError, match="does not have"):
        revised_investigation(
            _published_investigation(), verification, reason="fix", judge_version="v1", revised_at=_REVISED_AT
        )


def test_revision_rejects_a_reason_that_is_prose_rather_than_a_code() -> None:
    with pytest.raises(ValueError, match="reason must be a safe identifier"):
        revised_investigation(
            _published_investigation(),
            _verification(passed=True),
            reason="the customer said it was wrong",
            judge_version="v1",
            revised_at=_REVISED_AT,
        )


def test_revision_rejects_a_naive_timestamp() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        revised_investigation(
            _published_investigation(),
            _verification(passed=True),
            reason="fix",
            judge_version="v1",
            revised_at=datetime(2026, 9, 25),
        )


def test_revise_verdict_publishes_the_revision_and_supersedes_the_old_trace() -> None:
    mlflow = _FakeMlflow()
    original = _published_investigation(passed=False)
    _publisher(mlflow).publish(original)

    receipt = _revise(mlflow, original, _verification(passed=True))

    assert receipt.outcome == "revised"
    assert receipt.investigation_id == "investigation-a1:rev1"
    assert _status_of(mlflow, "investigation-a1") == "superseded"
    assert _status_of(mlflow, "investigation-a1:rev1") == "completed"
    old_trace = mlflow.traces[0]
    assert old_trace.info.tags["learning.investigation.superseded_by"] == "investigation-a1:rev1"
    assert len(mlflow.traces) == 2
    assert len(receipt.assessment_ids) == len(mlflow.feedback) > 0
    assert {entry["trace_id"] for entry in mlflow.feedback} == {"source-trace-a1"}
    assert {entry["metadata"]["investigation_id"] for entry in mlflow.feedback} == {"investigation-a1:rev1"}


def test_the_reader_mines_only_the_revision_after_a_verdict_is_revised() -> None:
    mlflow = _FakeMlflow()
    original = _published_investigation(passed=False)
    _publisher(mlflow).publish(original)

    _revise(mlflow, original, _verification(passed=True))
    records = _reader(mlflow).load_records(InvestigationSelection(experiment_id="experiment-7"))

    assert len(records) == 1
    assert records[0].trace_id == "source-trace-a1"
    assert records[0].verified_success is True


def test_a_superseded_trace_is_never_loaded_even_on_its_own() -> None:
    mlflow = _FakeMlflow()
    original = _published_investigation(passed=True)
    _publisher(mlflow).publish(original)
    mlflow.set_trace_tag(mlflow.traces[0].info.trace_id, "learning.investigation.status", "superseded")

    records = _reader(mlflow).load_records(InvestigationSelection(experiment_id="experiment-7"))

    assert records == ()


def test_rerunning_a_finished_revision_changes_nothing() -> None:
    mlflow = _FakeMlflow()
    original = _published_investigation(passed=False)
    _publisher(mlflow).publish(original)
    first = _revise(mlflow, original, _verification(passed=True))
    tag_calls = list(mlflow.tag_calls)
    feedback_count = len(mlflow.feedback)

    second = _revise(mlflow, original, _verification(passed=True))

    assert second.outcome == "already_revised"
    assert second.digest == first.digest
    assert mlflow.tag_calls == tag_calls
    assert len(mlflow.feedback) == feedback_count
    assert len(mlflow.traces) == 2


def test_rerunning_after_a_crash_before_the_retag_finishes_the_revision() -> None:
    mlflow = _FakeMlflow()
    original = _published_investigation(passed=False)
    _publisher(mlflow).publish(original)
    mlflow.fail_on_tag = "learning.investigation.superseded_by"
    with pytest.raises(ConnectionError):
        _revise(mlflow, original, _verification(passed=True))
    assert _status_of(mlflow, "investigation-a1") == "completed"
    mlflow.fail_on_tag = None

    receipt = _revise(mlflow, original, _verification(passed=True))

    assert receipt.outcome == "finished_interrupted_revision"
    assert _status_of(mlflow, "investigation-a1") == "superseded"
    assert len(mlflow.traces) == 2


def test_rerunning_after_a_crash_between_the_two_retags_finishes_the_revision() -> None:
    mlflow = _FakeMlflow()
    original = _published_investigation(passed=False)
    _publisher(mlflow).publish(original)
    mlflow.fail_on_tag = "learning.investigation.status"
    with pytest.raises(ConnectionError):
        _revise(mlflow, original, _verification(passed=True))
    mlflow.fail_on_tag = None

    receipt = _revise(mlflow, original, _verification(passed=True))

    assert receipt.outcome == "finished_interrupted_revision"
    assert _status_of(mlflow, "investigation-a1") == "superseded"


def test_the_reader_keeps_only_the_revision_when_a_crash_left_both_completed() -> None:
    mlflow = _FakeMlflow()
    original = _published_investigation(passed=False)
    _publisher(mlflow).publish(original)
    mlflow.fail_on_tag = "learning.investigation.superseded_by"
    with pytest.raises(ConnectionError):
        _revise(mlflow, original, _verification(passed=True))

    records = _reader(mlflow).load_records(InvestigationSelection(experiment_id="experiment-7"))

    assert len(records) == 1
    assert records[0].verified_success is True


def test_revising_a_revision_supersedes_it_and_keeps_the_root_lineage() -> None:
    mlflow = _FakeMlflow()
    original = _published_investigation(passed=False)
    _publisher(mlflow).publish(original)
    _revise(mlflow, original, _verification(passed=True))
    first_revision = revised_investigation(
        original,
        _verification(passed=True),
        reason="judge_fix_scope_suffix",
        judge_version="judge.v7",
        revised_at=_REVISED_AT,
    )

    receipt = _revise(mlflow, first_revision, _verification(passed=False))
    records = _reader(mlflow).load_records(InvestigationSelection(experiment_id="experiment-7"))

    assert receipt.investigation_id == "investigation-a1:rev2"
    assert _status_of(mlflow, "investigation-a1:rev1") == "superseded"
    assert [record.verified_success for record in records] == [False]
    [latest] = mlflow.search_traces(
        locations=[], filter_string='tags.`learning.investigation.id` = "investigation-a1:rev2"', return_type="list"
    )
    assert latest.info.tags["learning.investigation.supersedes"] == "investigation-a1:rev1"


def test_the_reader_drops_every_superseded_link_of_a_chain_left_completed() -> None:
    mlflow = _FakeMlflow()
    original = _published_investigation(passed=False)
    first = revised_investigation(
        original, _verification(passed=True), reason="fix", judge_version="v1", revised_at=_REVISED_AT
    )
    second = revised_investigation(
        first, _verification(passed=False), reason="fix", judge_version="v2", revised_at=_REVISED_AT
    )
    for document in (original, first, second):
        _publisher(mlflow).publish(document)

    records = _reader(mlflow).load_records(InvestigationSelection(experiment_id="experiment-7"))

    assert [record.investigation_digest for record in records] == [second.digest()]


def test_revising_a_superseded_document_again_with_different_inputs_is_refused() -> None:
    mlflow = _FakeMlflow()
    original = _published_investigation(passed=False)
    _publisher(mlflow).publish(original)
    _revise(mlflow, original, _verification(passed=True))

    with pytest.raises(ValueError, match="already holds a different revision"):
        revise_verdict(
            original,
            _verification(passed=True),
            reason="another_fix",
            judge_version="judge.v8",
            trace_store=mlflow,
            publisher=_publisher(mlflow),
            now=_REVISED_AT,
        )


def test_a_document_superseded_by_another_revision_is_refused() -> None:
    mlflow = _FakeMlflow()
    original = _published_investigation(passed=False)
    _publisher(mlflow).publish(original)
    mlflow.set_trace_tag(mlflow.traces[0].info.trace_id, "learning.investigation.superseded_by", "other:rev3")

    with pytest.raises(ValueError, match="already superseded by 'other:rev3'"):
        _revise(mlflow, original, _verification(passed=True))


def test_revising_an_investigation_that_was_never_published_is_refused() -> None:
    mlflow = _FakeMlflow()

    with pytest.raises(ValueError, match="no published trace"):
        _revise(mlflow, _published_investigation(), _verification(passed=True))
    assert mlflow.traces == []


def test_revising_from_a_document_that_differs_from_the_published_one_is_refused() -> None:
    mlflow = _FakeMlflow()
    original = _published_investigation(passed=False)
    _publisher(mlflow).publish(original)
    edited = replace(original, step_signature="lookup_stock")

    with pytest.raises(ValueError, match="does not match its published digest"):
        _revise(mlflow, edited, _verification(passed=True))
    assert len(mlflow.traces) == 1


def test_an_injected_trace_store_needs_a_publisher() -> None:
    with pytest.raises(ValueError, match="publisher is required"):
        revise_verdict(
            _published_investigation(),
            _verification(passed=True),
            reason="fix",
            judge_version="v1",
            trace_store=_FakeMlflow(),
        )


def test_real_mlflow_revision_supersedes_the_old_trace_and_the_reader_mines_the_revision(tmp_path: Path) -> None:
    mlflow = pytest.importorskip("mlflow")

    previous_tracking_uri = mlflow.get_tracking_uri()
    try:
        mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
        experiment_id = mlflow.create_experiment("lcp-revision-test", artifact_location=(tmp_path / "a").as_uri())
        original = _published_investigation(experiment_id=experiment_id, passed=False)
        MlflowAttachmentPublisher().publish(original)

        receipt = revise_verdict(
            original,
            _verification(passed=True),
            reason="judge_fix",
            judge_version="judge.v7",
            trace_store=mlflow,
            publisher=MlflowAttachmentPublisher(),
            now=_REVISED_AT,
        )
        rerun = revise_verdict(
            original,
            _verification(passed=True),
            reason="judge_fix",
            judge_version="judge.v7",
            trace_store=mlflow,
            publisher=MlflowAttachmentPublisher(),
        )
        records = MlflowInvestigationReader().load_records(InvestigationSelection(experiment_id=experiment_id))

        assert receipt.outcome == "revised"
        assert rerun.outcome == "already_revised"
        assert [record.verified_success for record in records] == [True]
        [old_trace] = mlflow.search_traces(
            locations=[experiment_id],
            filter_string='tags.`learning.investigation.id` = "investigation-a1"',
            return_type="list",
        )
        assert old_trace.info.tags["learning.investigation.status"] == "superseded"
        assert old_trace.info.tags["learning.investigation.superseded_by"] == "investigation-a1:rev1"
    finally:
        mlflow.set_tracking_uri(previous_tracking_uri)
