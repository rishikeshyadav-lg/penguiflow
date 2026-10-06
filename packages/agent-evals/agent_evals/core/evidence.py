"""Redacted evidence records: who ran what, and a sink interface to send them to.

This module carries no destination-specific naming. A destination (MLflow, OpenTelemetry, a file)
turns an `EvidenceEvent` into its own tags and attributes; the learning control plane keeps its
`lcp.*` projections in its own package.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

_SENSITIVE_ATTRIBUTE_PARTS = frozenset(
    {
        "api_key",
        "authorization",
        "content",
        "cookie",
        "credential",
        "input",
        "message",
        "output",
        "password",
        "prompt",
        "secret",
        "token",
    }
)


def _require_non_empty(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


def redact_attributes(attributes: Mapping[str, Any]) -> dict[str, Any]:
    """Remove content and credential fields from evidence metadata."""

    redacted: dict[str, Any] = {}
    for raw_key, value in attributes.items():
        key = str(raw_key).strip()
        normalized_key = key.lower().replace("-", "_")

        if not key or any(part in normalized_key for part in _SENSITIVE_ATTRIBUTE_PARTS):
            continue

        try:
            json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        except (TypeError, ValueError):
            redacted[key] = str(value)
        else:
            redacted[key] = value
    return redacted


@dataclass(frozen=True, slots=True)
class EvidenceContext:
    """Identify the exact agent, deployment, and evaluation evidence belongs to."""

    agent_id: str
    deployment_digest: str
    trace_id: str | None = None
    evaluation_id: str | None = None
    candidate_id: str | None = None
    dataset_version: str | None = None
    metric_version: str | None = None
    policy_version: str | None = None
    scope_ref: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_id", _require_non_empty(self.agent_id, "agent_id"))
        object.__setattr__(self, "deployment_digest", _require_non_empty(self.deployment_digest, "deployment_digest"))


@dataclass(frozen=True, slots=True)
class EvidenceEvent:
    """Record one evaluation event and its redacted metadata."""

    event_type: str
    context: EvidenceContext
    attributes: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: f"ev_{uuid4().hex}")
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", _require_non_empty(self.event_type, "event_type"))
        object.__setattr__(self, "event_id", _require_non_empty(self.event_id, "event_id"))
        if self.occurred_at.tzinfo is None:
            raise ValueError("occurred_at must be timezone-aware")
        object.__setattr__(self, "attributes", redact_attributes(self.attributes))
        object.__setattr__(self, "metrics", self._validated_metrics())

    def _validated_metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for raw_name, raw_value in self.metrics.items():
            name = _require_non_empty(str(raw_name), "metric name")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(f"metric {name!r} must be finite")
            metrics[name] = value
        return metrics

    def record(self) -> dict[str, Any]:
        """Return the JSON-safe record stored by evidence systems."""

        record = asdict(self.context)
        record.update(
            {
                "event_id": self.event_id,
                "event_type": self.event_type,
                "occurred_at": self.occurred_at.isoformat(),
                "attributes": dict(self.attributes),
                "metrics": dict(self.metrics),
            }
        )
        return record


@runtime_checkable
class EvidenceSink(Protocol):
    """Best-effort destination for redacted evidence records."""

    def emit(self, event: EvidenceEvent) -> bool:
        """Store or publish an event, returning whether the attempt succeeded."""
        ...


__all__ = ["EvidenceContext", "EvidenceEvent", "EvidenceSink", "redact_attributes"]
