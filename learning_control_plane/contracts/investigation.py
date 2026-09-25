"""Portable, redacted investigation trajectory contract and canonical digest."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal

INVESTIGATION_TRAJECTORY_SCHEMA_VERSION = "investigation_trajectory.v1"
# Extension written on a document that corrects an earlier document's verdict (see verdict_revision).
REVISION_EXTENSION = "learning.revision"
InvestigationStatus = Literal["completed", "failed", "timed_out", "cancelled", "interrupted", "unknown"]
_VALID_STATUSES = frozenset({"completed", "failed", "timed_out", "cancelled", "interrupted", "unknown"})


def _non_empty(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


def _timestamp(value: datetime, field_name: str) -> str:
    if value.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _canonical_number(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("canonical JSON does not allow non-finite numbers")
    if value == 0:
        return "0"

    decimal_value = Decimal(str(value)).normalize()
    text = format(decimal_value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _canonical_json(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _canonical_number(value)
    if isinstance(value, Mapping):
        entries: list[str] = []
        for key in sorted(value):
            if not isinstance(key, str):
                raise ValueError("canonical JSON object keys must be strings")
            entries.append(f"{_canonical_json(key)}:{_canonical_json(value[key])}")
        return "{" + ",".join(entries) + "}"
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return "[" + ",".join(_canonical_json(item) for item in value) + "]"
    raise ValueError(f"canonical JSON does not support {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class SourceTraceRef:
    """The complete opaque locator for the native run behind an investigation."""

    tracking_store_ref: str
    experiment_id: str
    mlflow_trace_id: str
    deployment_ref: str
    native_trace_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tracking_store_ref", _non_empty(self.tracking_store_ref, "tracking_store_ref"))
        object.__setattr__(self, "experiment_id", _non_empty(self.experiment_id, "experiment_id"))
        object.__setattr__(self, "mlflow_trace_id", _non_empty(self.mlflow_trace_id, "mlflow_trace_id"))
        object.__setattr__(self, "deployment_ref", _non_empty(self.deployment_ref, "deployment_ref"))
        if self.native_trace_id is not None:
            object.__setattr__(self, "native_trace_id", _non_empty(self.native_trace_id, "native_trace_id"))

    def record(self) -> dict[str, str]:
        """Return the locator without requiring data from the native trace."""

        record = {
            "tracking_store_ref": self.tracking_store_ref,
            "experiment_id": self.experiment_id,
            "mlflow_trace_id": self.mlflow_trace_id,
            "deployment_ref": self.deployment_ref,
        }
        if self.native_trace_id is not None:
            record["native_trace_id"] = self.native_trace_id
        return record


@dataclass(frozen=True, slots=True)
class InvestigationTrajectoryV1:
    """One redacted, portable projection of one completed native agent run."""

    investigation_id: str
    source_trace_ref: SourceTraceRef
    agent_ref: str
    provider_ref: str
    scope_ref: str
    started_at: datetime
    status: InvestigationStatus
    execution_fingerprint: str
    request: Mapping[str, Any]
    steps: Sequence[Mapping[str, Any]]
    redaction_profile: str
    step_signature: str
    completed_at: datetime | None = None
    session_ref: str | None = None
    parent_investigation_ids: Sequence[str] = ()
    intent_descriptor: Mapping[str, Any] | None = None
    model_context: Mapping[str, Any] = field(default_factory=dict)
    execution_context: Mapping[str, Any] = field(default_factory=dict)
    input_parts: Sequence[Mapping[str, Any]] = ()
    artifact_refs: Sequence[Mapping[str, Any]] = ()
    source_refs: Sequence[Mapping[str, Any]] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)
    summary: Mapping[str, Any] | None = None
    control_state: Mapping[str, Any] | None = None
    resume_input: Mapping[str, Any] | None = None
    interventions: Sequence[Mapping[str, Any]] = ()
    async_results: Sequence[Mapping[str, Any]] = ()
    final_output: Mapping[str, Any] | None = None
    termination_reason: str | None = None
    events: Sequence[Mapping[str, Any]] = ()
    outcome_refs: Sequence[str] = ()
    assessment_refs: Sequence[str] = ()
    extensions: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = field(default=INVESTIGATION_TRAJECTORY_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "investigation_id", _non_empty(self.investigation_id, "investigation_id"))
        object.__setattr__(self, "agent_ref", _non_empty(self.agent_ref, "agent_ref"))
        object.__setattr__(self, "provider_ref", _non_empty(self.provider_ref, "provider_ref"))
        object.__setattr__(self, "scope_ref", _non_empty(self.scope_ref, "scope_ref"))
        object.__setattr__(
            self,
            "execution_fingerprint",
            _non_empty(self.execution_fingerprint, "execution_fingerprint"),
        )
        object.__setattr__(self, "redaction_profile", _non_empty(self.redaction_profile, "redaction_profile"))
        object.__setattr__(self, "step_signature", _non_empty(self.step_signature, "step_signature"))
        _timestamp(self.started_at, "started_at")
        if self.completed_at is not None:
            _timestamp(self.completed_at, "completed_at")
        if self.status not in _VALID_STATUSES:
            raise ValueError(f"unsupported investigation status: {self.status!r}")
        if self.session_ref is not None:
            object.__setattr__(self, "session_ref", _non_empty(self.session_ref, "session_ref"))

    def record(self) -> dict[str, Any]:
        """Return the complete JSON-safe contract record before canonical encoding."""

        record: dict[str, Any] = {
            "schema_version": self.schema_version,
            "investigation_id": self.investigation_id,
            "source_trace_ref": self.source_trace_ref.record(),
            "agent_ref": self.agent_ref,
            "provider_ref": self.provider_ref,
            "scope_ref": self.scope_ref,
            "started_at": _timestamp(self.started_at, "started_at"),
            "status": self.status,
            "execution_fingerprint": self.execution_fingerprint,
            "request": dict(self.request),
            "steps": list(self.steps),
            "redaction_profile": self.redaction_profile,
            "step_signature": self.step_signature,
            "model_context": dict(self.model_context),
            "execution_context": dict(self.execution_context),
            "input_parts": list(self.input_parts),
            "artifact_refs": list(self.artifact_refs),
            "source_refs": list(self.source_refs),
            "attributes": dict(self.attributes),
            "interventions": list(self.interventions),
            "async_results": list(self.async_results),
            "events": list(self.events),
            "outcome_refs": list(self.outcome_refs),
            "assessment_refs": list(self.assessment_refs),
            "extensions": dict(self.extensions),
        }
        optional_values = {
            "completed_at": _timestamp(self.completed_at, "completed_at") if self.completed_at else None,
            "session_ref": self.session_ref,
            "parent_investigation_ids": list(self.parent_investigation_ids),
            "intent_descriptor": dict(self.intent_descriptor) if self.intent_descriptor else None,
            "summary": dict(self.summary) if self.summary else None,
            "control_state": dict(self.control_state) if self.control_state else None,
            "resume_input": dict(self.resume_input) if self.resume_input else None,
            "final_output": dict(self.final_output) if self.final_output else None,
            "termination_reason": self.termination_reason,
        }
        record.update({key: value for key, value in optional_values.items() if value is not None})
        return record

    def canonical_bytes(self) -> bytes:
        """Return the contract's canonical UTF-8 JSON bytes without a byte-order mark."""

        return _canonical_json(self.record()).encode("utf-8")

    def digest(self) -> str:
        """Return the SHA-256 identity of the canonical document bytes."""

        return f"sha256:{hashlib.sha256(self.canonical_bytes()).hexdigest()}"

    @property
    def revision(self) -> Mapping[str, Any] | None:
        """Return the revision record when this document supersedes an earlier one."""

        revision = self.extensions.get(REVISION_EXTENSION)
        if isinstance(revision, Mapping) and isinstance(revision.get("supersedes"), str):
            return revision
        return None

    def query_index(self) -> dict[str, str]:
        """Return the small string-only index used to discover investigations."""

        intent_class = "unknown"
        if self.intent_descriptor is not None:
            candidate = self.intent_descriptor.get("class")
            if isinstance(candidate, str) and candidate.strip():
                intent_class = candidate.strip()
        index = {
            "learning.investigation.schema": self.schema_version,
            "learning.investigation.id": self.investigation_id,
            "learning.investigation.agent_ref": self.agent_ref,
            "learning.investigation.provider_ref": self.provider_ref,
            "learning.investigation.scope_ref": self.scope_ref,
            "learning.investigation.status": self.status,
            "learning.investigation.execution_fingerprint": self.execution_fingerprint,
            "learning.investigation.intent_class": intent_class,
            "learning.investigation.step_signature": self.step_signature,
            "learning.investigation.has_outcome": str(bool(self.outcome_refs)).lower(),
            "learning.investigation.has_assessment": str(bool(self.assessment_refs)).lower(),
        }
        if "verified_success" in self.execution_context:
            index["learning.investigation.verified_success"] = str(
                self.execution_context.get("verified_success") is True
            ).lower()
        # Only revisions carry these, so the index of every existing document is unchanged.
        if self.revision is not None:
            index["learning.investigation.supersedes"] = str(self.revision["supersedes"])
            index["learning.investigation.revision_key"] = str(self.revision.get("revision_key", ""))
        return index


def investigation_from_canonical_bytes(content_bytes: bytes) -> InvestigationTrajectoryV1:
    """Rebuild a document from its canonical bytes, refusing anything that is not exactly canonical."""

    try:
        decoded = content_bytes.decode("utf-8")
        payload = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("investigation attachment is not UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("investigation attachment must contain a JSON object")
    if payload.get("schema_version") != INVESTIGATION_TRAJECTORY_SCHEMA_VERSION:
        raise ValueError("investigation attachment has an unsupported schema version")

    document_payload = dict(payload)
    document_payload.pop("schema_version", None)
    source_trace_ref = document_payload.pop("source_trace_ref", None)
    if not isinstance(source_trace_ref, Mapping):
        raise ValueError("investigation attachment has no source_trace_ref")
    for timestamp_name in ("started_at", "completed_at"):
        value = document_payload.get(timestamp_name)
        if value is not None:
            document_payload[timestamp_name] = _parse_timestamp(value, timestamp_name)

    try:
        document = InvestigationTrajectoryV1(
            source_trace_ref=SourceTraceRef(**dict(source_trace_ref)),
            **document_payload,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("investigation attachment does not match InvestigationTrajectoryV1") from error
    if document.canonical_bytes() != content_bytes:
        raise ValueError("investigation attachment bytes are not canonical")
    return document


def _parse_timestamp(value: object, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"investigation attachment {field_name} must be a timestamp string")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"investigation attachment {field_name} is invalid") from error


__all__ = [
    "INVESTIGATION_TRAJECTORY_SCHEMA_VERSION",
    "REVISION_EXTENSION",
    "InvestigationStatus",
    "InvestigationTrajectoryV1",
    "SourceTraceRef",
    "investigation_from_canonical_bytes",
]
