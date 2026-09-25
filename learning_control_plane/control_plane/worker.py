"""Manual/offline worker for persisted learning-control-plane evaluation jobs."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..evaluation.evaluation import Metric, RunOne
from .control_plane import LearningControlPlane

if TYPE_CHECKING:
    from ..judging.golden import GoldenReport
    from ..providers.assessment_publisher import MlflowAssessmentPublisher, PendingAssessmentQueue


@dataclass(frozen=True, slots=True)
class WorkerRun:
    """A summary of one bounded pass over durable draft evaluation jobs."""

    attempted_job_ids: tuple[str, ...]
    ready_for_review_job_ids: tuple[str, ...]
    rejected_job_ids: tuple[str, ...]
    failed_job_ids: tuple[str, ...]
    published_pending_assessments: tuple[str, ...] = ()
    still_pending_assessments: tuple[str, ...] = ()


class OfflineEvaluationWorker:
    """Execute persisted draft jobs without participating in agent request handling."""

    def __init__(
        self,
        control_plane: LearningControlPlane,
        *,
        run_one: RunOne,
        metric: Metric,
        golden_report: GoldenReport | None = None,
        pending_assessments: PendingAssessmentQueue | None = None,
        assessment_publisher: MlflowAssessmentPublisher | None = None,
    ) -> None:
        if (pending_assessments is None) != (assessment_publisher is None):
            raise ValueError("draining pending assessments needs both the queue and an assessment publisher")
        self._control_plane = control_plane
        self._run_one = run_one
        self._metric = metric
        self._golden_report = golden_report
        self._pending_assessments = pending_assessments
        self._assessment_publisher = assessment_publisher

    async def run_pending(self, *, max_jobs: int | None = None) -> WorkerRun:
        """Publish queued assessments, then evaluate a bounded, deterministic batch of draft jobs."""

        if max_jobs is not None and max_jobs < 1:
            raise ValueError("max_jobs must be at least 1")

        published: tuple[str, ...] = ()
        still_pending: tuple[str, ...] = ()
        if self._pending_assessments is not None and self._assessment_publisher is not None:
            drain = await asyncio.to_thread(self._pending_assessments.drain, self._assessment_publisher.publish_now)
            published, still_pending = drain.published, drain.still_pending

        pending_jobs = self._control_plane.list_jobs(state="draft")
        if max_jobs is not None:
            pending_jobs = pending_jobs[:max_jobs]

        ready_for_review: list[str] = []
        rejected: list[str] = []
        failed: list[str] = []
        for pending_job in pending_jobs:
            completed_job = await self._control_plane.run_job(
                pending_job.job_id,
                self._run_one,
                self._metric,
                golden_report=self._golden_report,
            )
            if completed_job.state == "ready_for_review":
                ready_for_review.append(completed_job.job_id)
            elif completed_job.state == "rejected":
                rejected.append(completed_job.job_id)
            else:
                failed.append(completed_job.job_id)

        return WorkerRun(
            attempted_job_ids=tuple(job.job_id for job in pending_jobs),
            ready_for_review_job_ids=tuple(ready_for_review),
            rejected_job_ids=tuple(rejected),
            failed_job_ids=tuple(failed),
            published_pending_assessments=published,
            still_pending_assessments=still_pending,
        )


__all__ = ["OfflineEvaluationWorker", "WorkerRun"]
