"""Correct a published verdict by superseding its investigation document, never by deleting it.

Investigation documents are write-once: the publisher refuses to publish an investigation ID
again with different bytes, and the mining reader trusts the verdict inside each "completed"
document. When a judge fix shows a published verdict was wrong, the correction is a new
document with its own ID (`<root id>:rev<n>`) that carries the new verification and a
`learning.revision` record naming the document it supersedes. The old document's trace is then
retagged `learning.investigation.status = "superseded"` with `superseded_by`, so the reader's
status filter stops loading it. Nothing is deleted.

The steps are ordered so a crash at any point can be finished by running the same revision again:
new assessments, then the new document, then `superseded_by`, then the old status. A re-run
recognises its own earlier revision by the revision key (a digest of the inputs) that the new
document carries in its tags. Assessments are logged at least once: a crash after logging them but
before the document is published logs them again on the re-run, each copy naming the same
investigation ID in its metadata.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from ..contracts.investigation import REVISION_EXTENSION, InvestigationTrajectoryV1
from ..evaluation.verification import InvestigationVerification
from .assessment_publisher import InvestigationAssessmentPublisher
from .investigation_publisher import INVESTIGATION_DIGEST_TAG, INVESTIGATION_ID_TAG, InvestigationPublisher

INVESTIGATION_STATUS_TAG = "learning.investigation.status"
SUPERSEDED_STATUS = "superseded"
SUPERSEDED_BY_TAG = "learning.investigation.superseded_by"
REVISION_KEY_TAG = "learning.investigation.revision_key"
_VERIFICATION_EXTENSION = "learning.verification"
# Step fields the projector copies from verification evidence; a revision rewrites all of them.
_STEP_EVIDENCE_FIELDS = ("argument_facts", "decision_reason_codes", "result_checks", "verified")
_SAFE_CODE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")

RevisionOutcome = Literal["revised", "finished_interrupted_revision", "already_revised"]


class TraceTagStore(Protocol):
    """The MLflow operations a revision needs: find an investigation's trace and retag it."""

    def search_traces(self, *, locations: Sequence[str], filter_string: str, return_type: str) -> Sequence[Any]:
        """Return the traces matching one tag filter."""
        ...

    def set_trace_tag(self, trace_id: str, key: str, value: str) -> None:
        """Set one tag on an existing trace."""
        ...


@dataclass(frozen=True, slots=True)
class RevisionReceipt:
    """What one revision call did: which document now carries the verdict, and how it got there."""

    superseded_investigation_id: str
    investigation_id: str
    digest: str
    outcome: RevisionOutcome
    assessment_ids: tuple[str, ...] = ()


def revised_investigation(
    original: InvestigationTrajectoryV1,
    verification: InvestigationVerification,
    *,
    reason: str,
    judge_version: str,
    revised_at: datetime,
) -> InvestigationTrajectoryV1:
    """Return the document that supersedes `original` with a corrected verification."""

    _require_safe_code(reason, "reason")
    _require_safe_code(judge_version, "judge_version")
    if revised_at.tzinfo is None:
        raise ValueError("revised_at must be timezone-aware")

    previous = original.revision
    if previous is None:
        root_id = original.investigation_id
        number = 1
    else:
        root_id = str(previous["root_investigation_id"])
        number = int(previous["revision"]) + 1
    assessment_refs: tuple[str, ...] = ()
    if verification.final_answer is not None:
        assessment_refs = (verification.final_answer.assessment_ref,)

    revision = {
        "supersedes": original.investigation_id,
        "supersedes_digest": original.digest(),
        "root_investigation_id": root_id,
        "revision": number,
        "reason": reason,
        "judge_version": judge_version,
        "revised_at": revised_at.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z"),
        "revision_key": revision_key(original, verification, reason=reason, judge_version=judge_version),
    }
    return replace(
        original,
        investigation_id=f"{root_id}:rev{number}",
        steps=_steps_with_evidence(original.steps, verification),
        execution_context={**original.execution_context, "verified_success": verification.verified_success},
        assessment_refs=assessment_refs,
        extensions={
            **original.extensions,
            _VERIFICATION_EXTENSION: verification.record(),
            REVISION_EXTENSION: revision,
        },
    )


def revision_key(
    original: InvestigationTrajectoryV1,
    verification: InvestigationVerification,
    *,
    reason: str,
    judge_version: str,
) -> str:
    """Return a digest of a revision's inputs, so a re-run can recognise its own earlier revision."""

    inputs = {
        "supersedes_digest": original.digest(),
        "verification": verification.record(),
        "reason": reason,
        "judge_version": judge_version,
    }
    encoded = json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def revise_verdict(
    original: InvestigationTrajectoryV1,
    verification: InvestigationVerification,
    *,
    reason: str,
    judge_version: str,
    trace_store: TraceTagStore | None = None,
    publisher: InvestigationPublisher | None = None,
    assessment_publisher: InvestigationAssessmentPublisher | None = None,
    now: datetime | None = None,
) -> RevisionReceipt:
    """Publish a corrected verdict for `original` and mark its trace superseded; safe to re-run.

    With no `trace_store`, the installed MLflow is used for tags, documents and assessments. An
    injected `trace_store` needs an injected `publisher`; assessments are then published only when
    an `assessment_publisher` is given.
    """

    if trace_store is None:
        trace_store, publisher, assessment_publisher = _mlflow_dependencies(publisher, assessment_publisher)
    if publisher is None:
        raise ValueError("publisher is required with an injected trace_store")

    experiment_id = original.source_trace_ref.experiment_id
    old_trace = _only_trace(trace_store, experiment_id, original.investigation_id)
    if old_trace is None:
        raise ValueError(f"investigation {original.investigation_id!r} has no published trace to supersede")
    old_tags = _tags(old_trace)
    if old_tags.get(INVESTIGATION_DIGEST_TAG) != original.digest():
        raise ValueError(f"investigation {original.investigation_id!r} does not match its published digest")

    key = revision_key(original, verification, reason=reason, judge_version=judge_version)
    draft = revised_investigation(
        original,
        verification,
        reason=reason,
        judge_version=judge_version,
        revised_at=now or datetime.now(UTC),
    )
    superseded_by = old_tags.get(SUPERSEDED_BY_TAG)
    if superseded_by is not None and superseded_by != draft.investigation_id:
        raise ValueError(
            f"investigation {original.investigation_id!r} is already superseded by {superseded_by!r}; "
            "revise that document instead"
        )

    existing_revision = _only_trace(trace_store, experiment_id, draft.investigation_id)
    if existing_revision is not None:
        existing_tags = _tags(existing_revision)
        if existing_tags.get(REVISION_KEY_TAG) != key:
            raise ValueError(
                f"{draft.investigation_id!r} already holds a different revision of {original.investigation_id!r}; "
                "revise that document instead"
            )
        digest = existing_tags[INVESTIGATION_DIGEST_TAG]
        if old_tags.get(INVESTIGATION_STATUS_TAG) == SUPERSEDED_STATUS and superseded_by == draft.investigation_id:
            return RevisionReceipt(original.investigation_id, draft.investigation_id, digest, "already_revised")
        _mark_superseded(trace_store, old_trace, draft.investigation_id)
        return RevisionReceipt(
            original.investigation_id, draft.investigation_id, digest, "finished_interrupted_revision"
        )

    assessment_ids: tuple[str, ...] = ()
    if assessment_publisher is not None:
        assessment_ids = assessment_publisher.publish(draft)
    digest = publisher.publish(draft)
    _mark_superseded(trace_store, old_trace, draft.investigation_id)
    return RevisionReceipt(original.investigation_id, draft.investigation_id, digest, "revised", assessment_ids)


def _steps_with_evidence(
    steps: Sequence[Mapping[str, Any]], verification: InvestigationVerification
) -> list[dict[str, Any]]:
    """Rewrite each step's copied verification fields from the new evidence, as the projector does."""

    evidence_by_index = {evidence.step_index: evidence for evidence in verification.step_evidence}
    out_of_range = sorted(index for index in evidence_by_index if index >= len(steps))
    if out_of_range:
        raise ValueError(f"verification has evidence for steps the investigation does not have: {out_of_range}")

    revised_steps: list[dict[str, Any]] = []
    for index, step in enumerate(steps):
        revised = {name: value for name, value in step.items() if name not in _STEP_EVIDENCE_FIELDS}
        evidence = evidence_by_index.get(index)
        if evidence is not None and evidence.node_name == step.get("node"):
            revised.update(
                {
                    "argument_facts": dict(evidence.argument_facts),
                    "decision_reason_codes": list(evidence.decision_reason_codes),
                    "result_checks": [check.record() for check in evidence.result_checks],
                    "verified": evidence.verified,
                }
            )
        revised_steps.append(revised)
    return revised_steps


def _mark_superseded(trace_store: TraceTagStore, trace: Any, revised_id: str) -> None:
    """Point the old trace at its revision first, so a crash never leaves it superseded by nothing."""

    trace_id = _trace_id(trace)
    trace_store.set_trace_tag(trace_id, SUPERSEDED_BY_TAG, revised_id)
    trace_store.set_trace_tag(trace_id, INVESTIGATION_STATUS_TAG, SUPERSEDED_STATUS)


def _only_trace(trace_store: TraceTagStore, experiment_id: str, investigation_id: str) -> Any | None:
    traces = trace_store.search_traces(
        locations=[experiment_id],
        filter_string=f'tags.`{INVESTIGATION_ID_TAG}` = "{investigation_id}"',
        return_type="list",
    )
    if not traces:
        return None
    if len(traces) > 1:
        raise RuntimeError(f"multiple MLflow traces use investigation_id {investigation_id!r}")
    return traces[0]


def _tags(trace: Any) -> dict[str, str]:
    tags = getattr(getattr(trace, "info", None), "tags", None)
    if not isinstance(tags, Mapping):
        raise ValueError("MLflow investigation trace has no tag mapping")
    return {str(name): str(value) for name, value in tags.items()}


def _trace_id(trace: Any) -> str:
    info = getattr(trace, "info", None)
    trace_id = getattr(info, "trace_id", None) or getattr(info, "request_id", None)
    if not isinstance(trace_id, str) or not trace_id:
        raise ValueError("MLflow investigation trace has no trace ID")
    return trace_id


def _require_safe_code(value: str, field_name: str) -> None:
    # Documents are content-free, so the reason is a code ("judge_fix_scope_suffix"), never prose.
    if _SAFE_CODE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a safe identifier")


def _mlflow_dependencies(
    publisher: InvestigationPublisher | None,
    assessment_publisher: InvestigationAssessmentPublisher | None,
) -> tuple[TraceTagStore, InvestigationPublisher, InvestigationAssessmentPublisher]:
    try:
        import mlflow
    except ImportError as error:
        raise RuntimeError("MLflow 3.12.0 or newer is required to revise a published verdict") from error
    from .assessment_publisher import MlflowAssessmentPublisher
    from .investigation_publisher import MlflowAttachmentPublisher

    return mlflow, publisher or MlflowAttachmentPublisher(), assessment_publisher or MlflowAssessmentPublisher()


__all__ = [
    "INVESTIGATION_STATUS_TAG",
    "REVISION_KEY_TAG",
    "RevisionOutcome",
    "RevisionReceipt",
    "SUPERSEDED_BY_TAG",
    "SUPERSEDED_STATUS",
    "TraceTagStore",
    "revise_verdict",
    "revised_investigation",
    "revision_key",
]
