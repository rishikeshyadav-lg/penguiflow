"""Redacted evidence records and best-effort OpenTelemetry/MLflow publishers.

The neutral pieces -- `EvidenceContext`, the event shape and the `EvidenceSink` interface -- live in
`agent_evals`, so an evaluation can emit evidence without knowing about this package. What stays here
is what is specific to the learning control plane: the `lcp.*` tag and attribute names, and the sinks
that write them. Publishers never make an LCP decision and never raise into an agent workload.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any

from agent_evals.core.evidence import EvidenceContext, EvidenceSink, redact_attributes
from agent_evals.core.evidence import EvidenceEvent as NeutralEvidenceEvent

logger = logging.getLogger("learning_control_plane.evidence")

MLFLOW_LINEAGE_SCHEMA_VERSION = "v1"


def mlflow_tags(event: NeutralEvidenceEvent) -> dict[str, str]:
    """Return the versioned MLflow tags that identify this evidence."""

    tags = {
        "lcp.lineage_schema": MLFLOW_LINEAGE_SCHEMA_VERSION,
        "lcp.event_id": event.event_id,
        "lcp.event_type": event.event_type,
        "lcp.occurred_at": event.occurred_at.isoformat(),
    }
    for key, value in asdict(event.context).items():
        if value is not None:
            tags[f"lcp.{key}"] = str(value)
    return tags


def mlflow_metrics(event: NeutralEvidenceEvent) -> dict[str, float]:
    """Return metric values under the reserved MLflow metric namespace."""

    return {f"lcp.metric.{name}": value for name, value in event.metrics.items()}


def mlflow_artifact_path(event: NeutralEvidenceEvent) -> str:
    """Return the fixed artifact path for this complete evidence receipt."""

    return f"learning_control_plane/evidence/{MLFLOW_LINEAGE_SCHEMA_VERSION}/{event.event_id}.json"


def telemetry_attributes(event: NeutralEvidenceEvent) -> dict[str, str | bool | float | int]:
    """Return flat scalar attributes accepted by OpenTelemetry."""

    attributes: dict[str, str | bool | float | int] = {
        "lcp.event_id": event.event_id,
        "lcp.event_type": event.event_type,
        "lcp.occurred_at": event.occurred_at.isoformat(),
    }
    for key, value in asdict(event.context).items():
        if value is not None:
            attributes[f"lcp.{key}"] = str(value)
    for key, value in event.attributes.items():
        if isinstance(value, (str, bool, float, int)):
            attributes[f"lcp.attr.{key}"] = value
    return attributes


@dataclass(frozen=True, slots=True)
class EvidenceEvent(NeutralEvidenceEvent):
    """An evidence event that can also describe itself in the `lcp.*` naming used by MLflow and OTel.

    The fields and validation are the neutral event's. The sinks below call the module functions, so
    they accept a neutral event (one an evaluation emitted) as readily as this one.
    """

    def mlflow_tags(self) -> dict[str, str]:
        return mlflow_tags(self)

    def mlflow_metrics(self) -> dict[str, float]:
        return mlflow_metrics(self)

    def mlflow_artifact_path(self) -> str:
        return mlflow_artifact_path(self)

    def telemetry_attributes(self) -> dict[str, str | bool | float | int]:
        return telemetry_attributes(self)


class CompositeEvidenceSink:
    """Fan out evidence without allowing one unavailable backend to block another."""

    def __init__(self, sinks: Sequence[EvidenceSink]) -> None:
        self._sinks = tuple(sinks)

    def emit(self, event: NeutralEvidenceEvent) -> bool:
        delivered = False
        for sink in self._sinks:
            try:
                delivered = sink.emit(event) or delivered
            except Exception:
                logger.warning("Evidence sink failed", exc_info=True)
        return delivered


class OpenTelemetryEvidenceSink:
    """Emit metadata-only LCP spans when OpenTelemetry is configured.

    OpenTelemetry is imported lazily. If it is absent or an exporter fails, the
    caller receives ``False`` and can continue its scheduled/offline work.
    """

    def __init__(self, *, tracer: Any | None = None, span_prefix: str = "lcp.evidence") -> None:
        self._tracer = tracer
        self._span_prefix = span_prefix
        self._unavailable = False

    def _resolve_tracer(self) -> Any | None:
        if self._unavailable:
            return None
        if self._tracer is not None:
            return self._tracer
        try:
            from opentelemetry import trace
        except ImportError:
            self._unavailable = True
            logger.info("OpenTelemetry evidence sink disabled: opentelemetry-api is not installed")
            return None
        self._tracer = trace.get_tracer("learning_control_plane")
        return self._tracer

    def emit(self, event: NeutralEvidenceEvent) -> bool:
        tracer = self._resolve_tracer()
        if tracer is None:
            return False
        try:
            with tracer.start_as_current_span(f"{self._span_prefix}.{event.event_type}") as span:
                for key, value in telemetry_attributes(event).items():
                    span.set_attribute(key, value)
            return True
        except Exception:
            logger.warning("OpenTelemetry evidence emission failed", exc_info=True)
            return False


class MlflowEvidenceSink:
    """Persist redacted evidence records as MLflow tags and JSON artifacts.

    MLflow remains an evidence store: this sink creates no candidate, gate, or
    promotion decision. It accepts an injected module to make integration tests and
    host-specific MLflow configuration straightforward.
    """

    def __init__(self, *, mlflow_module: Any | None = None, run_name_prefix: str = "lcp-evidence") -> None:
        self._mlflow = mlflow_module
        self._run_name_prefix = run_name_prefix
        self._unavailable = False

    def _module(self) -> Any | None:
        if self._unavailable:
            return None
        if self._mlflow is not None:
            return self._mlflow
        try:
            import mlflow
        except ImportError:
            self._unavailable = True
            logger.info("MLflow evidence sink disabled: mlflow is not installed")
            return None
        self._mlflow = mlflow
        return mlflow

    def emit(self, event: NeutralEvidenceEvent) -> bool:
        mlflow = self._module()
        if mlflow is None:
            return False
        try:
            active_run = getattr(mlflow, "active_run", lambda: None)()
            run_context = nullcontext(active_run)
            if active_run is None:
                run_context = mlflow.start_run(run_name=f"{self._run_name_prefix}-{event.event_type}")
            with run_context:
                mlflow.set_tags(mlflow_tags(event))
                metrics = mlflow_metrics(event)
                if metrics and hasattr(mlflow, "log_metrics"):
                    mlflow.log_metrics(metrics)
                if hasattr(mlflow, "log_dict"):
                    mlflow.log_dict(event.record(), mlflow_artifact_path(event))
            return True
        except Exception:
            logger.warning("MLflow evidence emission failed", exc_info=True)
            return False


__all__ = [
    "CompositeEvidenceSink",
    "EvidenceContext",
    "EvidenceEvent",
    "EvidenceSink",
    "MLFLOW_LINEAGE_SCHEMA_VERSION",
    "MlflowEvidenceSink",
    "OpenTelemetryEvidenceSink",
    "mlflow_artifact_path",
    "mlflow_metrics",
    "mlflow_tags",
    "redact_attributes",
    "telemetry_attributes",
]
