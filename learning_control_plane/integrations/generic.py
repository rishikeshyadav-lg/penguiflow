"""Plug any agent into the learning control plane, whatever framework (or none) it is built with.

An integration expresses each finished run as an `AgentRun` (see `judging.runs`) and gives the
trusted identity of that run as a `RunContext`. `project_run` turns the two into a redacted
`InvestigationTrajectoryV1`: node names outside the integration's allowlist are redacted, raw
arguments, results and answers never enter the document, and only the judge's safe verification
evidence is copied in.

`RunPublisher.publish_after_turn` is the whole post-turn path: it waits for nothing the user is
waiting on, computes the judge's reference and the meaning check with timeouts, judges, projects,
and publishes the assessment and the document. It never raises into the agent's turn.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import Any

from ..contracts.investigation import InvestigationStatus, InvestigationTrajectoryV1, SourceTraceRef
from ..evaluation.verification import InvestigationVerification
from ..judging.answer_text import shown_to_user
from ..judging.judge import OutcomeLadder, ReferenceBuilder, VerificationProjector
from ..judging.meaning import MeaningCheck
from ..judging.runs import AgentRun
from ..judging.signature import SignatureNormalizer
from ..providers.assessment_publisher import InvestigationAssessmentPublisher
from ..providers.investigation_publisher import InvestigationPublisher

logger = logging.getLogger("learning_control_plane.integrations.generic")

# Terminal reasons a run may report, and the portable investigation status each maps to.
_STATUS_BY_FINISH_REASON: dict[str, InvestigationStatus] = {
    "answer_complete": "completed",
    "budget_exhausted": "timed_out",
    "cancelled": "cancelled",
    "pause": "interrupted",
    "paused": "interrupted",
    "interrupted": "interrupted",
    "no_path": "failed",
}


@dataclass(frozen=True, slots=True)
class RunContext:
    """Trusted identity of one run, supplied by the integration rather than read from the run."""

    source_trace_ref: SourceTraceRef
    agent_ref: str
    scope_ref: str
    execution_fingerprint: str
    started_at: datetime
    provider_ref: str = "generic"
    redaction_profile: str = "generic-investigation-safe:v1"
    allowed_node_names: frozenset[str] = frozenset()
    intent_descriptor: Mapping[str, str] | None = None


@dataclass(frozen=True, slots=True)
class RunPublication:
    """What one post-turn publication produced."""

    document: InvestigationTrajectoryV1
    digest: str
    assessment_ids: tuple[str, ...] = ()


def project_run(
    run: AgentRun,
    context: RunContext,
    *,
    judge: VerificationProjector | Callable[[AgentRun], InvestigationVerification | None] | None = None,
    signature: SignatureNormalizer | None = None,
    completed_at: datetime | None = None,
    investigation_id: str | None = None,
) -> InvestigationTrajectoryV1:
    """Project one run into a redacted investigation document, with the judge's verification if given.

    A judge that raises is logged and treated as absent, so a judge bug never loses the document.
    Without a `signature` rule every projected step is part of the step signature.
    """

    verification = _judged(run, judge)
    evidence_by_index = (
        {evidence.step_index: evidence for evidence in verification.step_evidence} if verification is not None else {}
    )
    projected_steps: list[dict[str, Any]] = []
    for index, step in enumerate(run.steps):
        node = step.tool if step.tool in context.allowed_node_names else "redacted_node"
        projected_step: dict[str, Any] = {
            "index": index,
            "node": node,
            "status": "failed" if step.error else "completed",
            "has_observation": step.result is not None,
            "has_streams": step.streamed,
        }
        evidence = evidence_by_index.get(index)
        if evidence is not None and evidence.node_name == node:
            projected_step.update(
                {
                    "argument_facts": dict(evidence.argument_facts),
                    "decision_reason_codes": list(evidence.decision_reason_codes),
                    "result_checks": [check.record() for check in evidence.result_checks],
                    "verified": evidence.verified,
                }
            )
        projected_steps.append(projected_step)

    if signature is not None:
        signature_nodes = list(signature(projected_steps))
    else:
        signature_nodes = [str(step["node"]) for step in projected_steps]
    extensions: dict[str, Any] = {}
    assessment_refs: tuple[str, ...] = ()
    if verification is not None:
        extensions["learning.verification"] = verification.record()
        if verification.final_answer is not None:
            assessment_refs = (verification.final_answer.assessment_ref,)

    return InvestigationTrajectoryV1(
        investigation_id=investigation_id or _investigation_id(context.source_trace_ref),
        source_trace_ref=context.source_trace_ref,
        agent_ref=context.agent_ref,
        provider_ref=context.provider_ref,
        scope_ref=context.scope_ref,
        started_at=context.started_at,
        completed_at=completed_at,
        status=_STATUS_BY_FINISH_REASON.get(run.finish_reason, "unknown"),
        execution_fingerprint=context.execution_fingerprint,
        request={"has_text": bool(run.question), "input_part_count": run.input_part_count},
        steps=projected_steps,
        redaction_profile=context.redaction_profile,
        step_signature=">".join(signature_nodes) or "no_steps",
        intent_descriptor=context.intent_descriptor,
        execution_context={
            "step_count": len(projected_steps),
            "failed_step_count": sum(step["status"] == "failed" for step in projected_steps),
            "has_final_answer": run.final_answer is not None,
            "verified_success": verification.verified_success if verification is not None else False,
        },
        termination_reason=run.finish_reason if run.finish_reason in _STATUS_BY_FINISH_REASON else "unknown",
        assessment_refs=assessment_refs,
        extensions=extensions,
    )


class RunPublisher:
    """Judge and publish a run after its turn has its final answer; never fails the turn."""

    def __init__(
        self,
        publisher: InvestigationPublisher,
        *,
        judge: OutcomeLadder | None = None,
        assessment_publisher: InvestigationAssessmentPublisher | None = None,
        reference_builder: ReferenceBuilder | None = None,
        meaning: MeaningCheck | None = None,
        signature: SignatureNormalizer | None = None,
        reference_timeout_s: float = 20.0,
        meaning_timeout_s: float = 60.0,
    ) -> None:
        self._publisher = publisher
        self._judge = judge
        self._assessment_publisher = assessment_publisher
        self._reference_builder = reference_builder
        self._meaning = meaning
        self._signature = signature
        self._reference_timeout_s = reference_timeout_s
        self._meaning_timeout_s = meaning_timeout_s

    async def publish_after_turn(
        self,
        run: AgentRun,
        context: RunContext,
        *,
        completed_at: datetime | None = None,
    ) -> RunPublication | None:
        """Publish one finished run's investigation, or return None when it was skipped or failed."""

        if run.final_answer is None:
            # Judging before the answer exists scores an empty answer; the caller publishes too early.
            logger.warning("lcp_run_publish_skipped reason=no_final_answer")
            return None
        try:
            reference = await self._reference(run)
            findings = await self._meaning_findings(run)
            judge = self._judge
            document = project_run(
                run,
                context,
                judge=(lambda judged: judge.judge(judged, reference, meaning_findings=findings)) if judge else None,
                signature=self._signature,
                completed_at=completed_at,
            )
            assessment_ids: tuple[str, ...] = ()
            if self._assessment_publisher is not None:
                try:
                    assessment_ids = await asyncio.to_thread(self._assessment_publisher.publish, document)
                except Exception:  # noqa: BLE001 -- the document still carries the verdict
                    logger.warning("lcp_run_assessment_publish_failed", exc_info=True)
            digest = await asyncio.to_thread(self._publisher.publish, document)
        except Exception:  # noqa: BLE001 -- publishing is evidence collection, never part of the answer
            logger.warning("lcp_run_publish_failed", exc_info=True)
            return None
        return RunPublication(document=document, digest=digest, assessment_ids=assessment_ids)

    async def _reference(self, run: AgentRun) -> Any:
        if self._reference_builder is None:
            return None
        try:
            return await asyncio.wait_for(self._reference_builder(run), timeout=self._reference_timeout_s)
        except Exception as error:  # noqa: BLE001 -- without a reference the judge falls back to the agent's calls
            logger.warning("lcp_judge_reference_unavailable error_type=%s", type(error).__name__)
            return None

    async def _meaning_findings(self, run: AgentRun) -> tuple[str, ...]:
        if self._meaning is None:
            return ()
        try:
            findings = await asyncio.wait_for(
                asyncio.to_thread(self._meaning.check, run.question, shown_to_user(run), run.steps),
                timeout=self._meaning_timeout_s,
            )
        except Exception as error:  # noqa: BLE001 -- the meaning check can only fail runs, so skipping it is safe
            logger.warning("lcp_meaning_check_unavailable error_type=%s", type(error).__name__)
            return ()
        return findings.codes


def _judged(
    run: AgentRun,
    judge: VerificationProjector | Callable[[AgentRun], InvestigationVerification | None] | None,
) -> InvestigationVerification | None:
    if judge is None:
        return None
    try:
        return judge(run)
    except Exception:
        logger.warning("run verification failed; projecting without it", exc_info=True)
        return None


def _investigation_id(source_trace_ref: SourceTraceRef) -> str:
    identity = ":".join(
        (source_trace_ref.tracking_store_ref, source_trace_ref.experiment_id, source_trace_ref.mlflow_trace_id)
    )
    return f"investigation_{sha256(identity.encode()).hexdigest()[:24]}"


__all__ = ["RunContext", "RunPublication", "RunPublisher", "project_run"]
