"""Best-effort MLflow publication of safe investigation assessments.

An assessment is logged on the agent's own MLflow trace, which is still being exported when the
turn ends. Logging too early fails with "trace not found", and a verdict lost that way was simply
gone. Now the publisher first waits for the trace to be closed (`TraceReadiness`), keeps its short
retries as a fallback, and when the trace still is not there it puts the assessment in a
`PendingAssessmentQueue` that the offline worker drains on its next pass. A verdict is delayed,
never dropped.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import monotonic, sleep
from typing import Any, Protocol

from ..contracts.investigation import InvestigationTrajectoryV1, investigation_from_canonical_bytes

logger = logging.getLogger("learning_control_plane.assessment_publisher")


class InvestigationAssessmentPublisher(Protocol):
    """Publish safe assessment scores linked to an investigation's source trace."""

    def publish(self, document: InvestigationTrajectoryV1) -> tuple[str, ...]:
        """Return identifiers for assessment records accepted by the backend."""
        ...


class TraceReadiness(Protocol):
    """Say when an agent's trace has been closed and can take assessments."""

    def wait_closed(self, trace_id: str, timeout_s: float) -> bool:
        """Wait up to `timeout_s` for the trace to be closed; return whether it is."""
        ...


class MlflowTraceReadiness:
    """Poll MLflow until a trace exists and is no longer in progress."""

    def __init__(
        self,
        *,
        mlflow_module: Any | None = None,
        poll_interval_s: float = 0.25,
        sleep_fn: Callable[[float], None] = sleep,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._mlflow = mlflow_module
        self._poll_interval_s = poll_interval_s
        self._sleep = sleep_fn
        self._clock = clock

    def wait_closed(self, trace_id: str, timeout_s: float) -> bool:
        """Return True once the trace is closed, or False when `timeout_s` passes first."""

        mlflow = self._dependencies()
        deadline = self._clock() + timeout_s
        while True:
            if _trace_is_closed(mlflow, trace_id):
                return True
            if self._clock() >= deadline:
                return False
            self._sleep(self._poll_interval_s)

    def _dependencies(self) -> Any:
        if self._mlflow is None:
            try:
                import mlflow
            except ImportError as error:
                raise RuntimeError("MLflow is required to wait for a trace") from error
            self._mlflow = mlflow
        return self._mlflow


def _trace_is_closed(mlflow: Any, trace_id: str) -> bool:
    """Return whether MLflow has the trace and it is no longer in progress; a lookup error means not yet."""

    try:
        trace = mlflow.get_trace(trace_id)
    except Exception:  # noqa: BLE001 -- a trace still being exported is often reported as an error
        return False
    if trace is None:
        return False
    state = getattr(trace.info, "state", None) or getattr(trace.info, "status", None)
    return str(getattr(state, "value", state)) != "IN_PROGRESS"


@dataclass(frozen=True, slots=True)
class PendingDrain:
    """What one drain of the pending queue did."""

    published: tuple[str, ...]
    still_pending: tuple[str, ...]


class PendingAssessmentQueue:
    """A durable JSONL queue of investigations whose assessments could not be logged yet.

    Each line holds one document's canonical bytes, which are already redacted, so the queue holds
    nothing the document itself does not. A document is kept once, however often it is queued.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._lock = Lock()

    def put(self, document: InvestigationTrajectoryV1) -> None:
        """Queue a document's assessments for a later drain."""

        line = document.canonical_bytes().decode("utf-8")
        with self._lock:
            if line in self._lines():
                return
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as queue_file:
                queue_file.write(line + "\n")

    def pending(self) -> tuple[InvestigationTrajectoryV1, ...]:
        """Return the queued documents, oldest first."""

        with self._lock:
            return tuple(investigation_from_canonical_bytes(line.encode("utf-8")) for line in self._lines())

    def drain(self, publish: Callable[[InvestigationTrajectoryV1], tuple[str, ...]]) -> PendingDrain:
        """Publish each queued document; keep the ones whose publication still fails."""

        with self._lock:
            published: list[str] = []
            kept: list[str] = []
            for line in self._lines():
                document = investigation_from_canonical_bytes(line.encode("utf-8"))
                try:
                    publish(document)
                except Exception:  # noqa: BLE001 -- kept for the next drain
                    logger.info("pending assessment still cannot be published: %s", document.investigation_id)
                    kept.append(line)
                    continue
                published.append(document.investigation_id)
            self._rewrite(kept)
            still_pending = tuple(investigation_from_canonical_bytes(line.encode()).investigation_id for line in kept)
        return PendingDrain(published=tuple(published), still_pending=still_pending)

    def _lines(self) -> list[str]:
        if not self._path.exists():
            return []
        return [line for line in self._path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _rewrite(self, lines: list[str]) -> None:
        """Replace the queue file in one step, so a crash mid-drain never loses a queued document."""

        temporary = self._path.with_suffix(self._path.suffix + ".tmp")
        temporary.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
        os.replace(temporary, self._path)


class MlflowAssessmentPublisher:
    """Log deterministic final-answer feedback on the source MLflow trace."""

    def __init__(
        self,
        *,
        mlflow_module: Any | None = None,
        trace_availability_delays: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0),
        sleep_fn: Callable[[float], None] = sleep,
        readiness: TraceReadiness | None = None,
        readiness_timeout_s: float = 30.0,
        pending_queue: PendingAssessmentQueue | None = None,
    ) -> None:
        self._mlflow = mlflow_module
        self._trace_availability_delays = trace_availability_delays
        self._sleep = sleep_fn
        self._readiness = readiness
        self._readiness_timeout_s = readiness_timeout_s
        self._pending_queue = pending_queue

    def publish(self, document: InvestigationTrajectoryV1) -> tuple[str, ...]:
        """Publish only scores and reason codes; never publish answer content.

        With a readiness check, wait for the trace to close first. With a pending queue, a trace
        that did not close in time, or never became available, queues the assessment instead of
        raising; without one, the short retries are the fallback.
        """

        trace_id = document.source_trace_ref.mlflow_trace_id
        if self._readiness is not None and not self._readiness.wait_closed(trace_id, self._readiness_timeout_s):
            if self._pending_queue is not None:
                self._pending_queue.put(document)
                logger.warning("MLflow trace %s was not closed in time; the assessment is queued", trace_id)
                return ()
            logger.info("MLflow trace %s was not closed in time; falling back to retries", trace_id)
        try:
            return self.publish_now(document)
        except Exception as error:
            if self._pending_queue is None or not _trace_is_not_available(error):
                raise
            self._pending_queue.put(document)
            logger.warning("MLflow trace %s is still not available; the assessment is queued", trace_id)
            return ()

    def publish_now(self, document: InvestigationTrajectoryV1) -> tuple[str, ...]:
        """Publish with the short trace-availability retries only; raise when the trace never appears."""

        for delay in (*self._trace_availability_delays, None):
            try:
                return self._publish_once(document)
            except Exception as error:
                if delay is None or not _trace_is_not_available(error):
                    raise
                logger.info(
                    "MLflow trace is not available for assessment publication; retrying in %ss",
                    delay,
                )
                self._sleep(delay)

        raise RuntimeError("unreachable")

    def _publish_once(self, document: InvestigationTrajectoryV1) -> tuple[str, ...]:
        """Publish one complete assessment set after the source trace is available."""

        verification = document.extensions.get("learning.verification")
        if not isinstance(verification, dict):
            return ()
        final_answer = verification.get("final_answer")
        assessment_ref = verification.get("assessment_ref")
        if not isinstance(final_answer, dict) or not isinstance(assessment_ref, str):
            return ()

        mlflow, assessment_source = self._dependencies()
        metadata = {
            "assessment_ref": assessment_ref,
            "investigation_id": document.investigation_id,
            "rubric_version": str(final_answer.get("rubric_version", "unknown")),
        }
        published_ids = [
            self._log_feedback(
                mlflow,
                assessment_source,
                trace_id=document.source_trace_ref.mlflow_trace_id,
                name="lcp_final_answer_accuracy",
                value=float(final_answer.get("score", 0.0)),
                reason_codes=final_answer.get("hard_failure_codes"),
                metadata=metadata,
            )
        ]
        for criterion in final_answer.get("criteria", []):
            if not isinstance(criterion, dict) or criterion.get("score") is None:
                continue
            criterion_id = criterion.get("criterion_id")
            if not isinstance(criterion_id, str):
                continue
            published_ids.append(
                self._log_feedback(
                    mlflow,
                    assessment_source,
                    trace_id=document.source_trace_ref.mlflow_trace_id,
                    name=f"lcp_final_answer_accuracy_{criterion_id}",
                    value=float(criterion["score"]),
                    reason_codes=criterion.get("reason_codes"),
                    metadata=metadata,
                )
            )
        published_ids.append(
            self._log_feedback(
                mlflow,
                assessment_source,
                trace_id=document.source_trace_ref.mlflow_trace_id,
                name="lcp_verified_success",
                value=bool(verification.get("verified_success")),
                reason_codes=(),
                metadata=metadata,
            )
        )
        return tuple(published_ids)

    def _dependencies(self) -> tuple[Any, Any]:
        if self._mlflow is None:
            try:
                import mlflow
                from mlflow.entities import AssessmentSource
            except ImportError as error:
                raise RuntimeError("MLflow with trace assessments is required") from error
            self._mlflow = mlflow
            return mlflow, AssessmentSource(source_type="CODE", source_id="learning-control-plane")

        assessment_source_factory = getattr(self._mlflow, "AssessmentSource", None)
        if assessment_source_factory is None:
            return self._mlflow, None
        return self._mlflow, assessment_source_factory(source_type="CODE", source_id="learning-control-plane")

    @staticmethod
    def _log_feedback(
        mlflow: Any,
        source: Any,
        *,
        trace_id: str,
        name: str,
        value: float | bool,
        reason_codes: Any,
        metadata: dict[str, str],
    ) -> str:
        codes = [str(code) for code in reason_codes or ()]
        assessment_name = name.replace(".", "_")
        assessment = mlflow.log_feedback(
            trace_id=trace_id,
            name=assessment_name,
            value=value,
            source=source,
            rationale=",".join(codes) if codes else None,
            metadata=metadata,
        )
        assessment_id = getattr(assessment, "assessment_id", None)
        return str(assessment_id or f"{metadata['assessment_ref']}:{assessment_name}")


def _trace_is_not_available(error: Exception) -> bool:
    """Recognize MLflow's transient response before an async trace export finishes."""

    message = str(error).lower()
    return "trace with id" in message and "not found" in message


__all__ = [
    "InvestigationAssessmentPublisher",
    "MlflowAssessmentPublisher",
    "MlflowTraceReadiness",
    "PendingAssessmentQueue",
    "PendingDrain",
    "TraceReadiness",
]
