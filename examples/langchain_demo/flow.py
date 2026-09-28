"""Run the learning-control-plane MVP against a LangChain agent, with no penguiflow in the path.

Mirrors `examples/learning_control_plane_local_demo/flow.py` step for step -- same control plane,
same mining and evaluation math, same gate -- swapping only the framework adapter and the agent
being learned from. That the two demos share everything except the adapter and the toy agent is
the point: proof that the control plane is genuinely framework-agnostic, not PenguiFlow-shaped.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from learning_control_plane.contracts.evidence import EvidenceContext
from learning_control_plane.control_plane import LearningControlPlane, PromotionPolicy
from learning_control_plane.control_plane.persistence import SQLiteControlPlaneRepository
from learning_control_plane.control_plane.worker import OfflineEvaluationWorker
from learning_control_plane.evaluation import (
    EvaluationCase,
    EvaluationDataset,
    EvaluationVariant,
    LocalEvaluationBackend,
)
from learning_control_plane.integrations.langchain.adapter import (
    LangChainFrameworkAdapter,
    LangChainInvestigationContext,
)
from learning_control_plane.mining import (
    CandidateMiner,
    TraceLearningRecord,
    reserve_later_held_out_cohort,
)

from .agent import run_campaign_agent

_QUERIES = [
    "how many clicks did campaign 112774 get",
    "how many clicks did campaign 112775 get",
    "how many clicks did campaign 112776 get",
    "how many clicks did campaign 112774 get",
    "how many clicks did campaign 112775 get",
    "how many clicks did campaign 112776 get",
    "how many clicks did campaign 112774 get",
]


async def run_demo(db_directory: Path) -> dict[str, str | list[str]]:
    """Execute the full learning loop against the toy LangChain agent and return its identifiers."""

    db_directory.mkdir(parents=True, exist_ok=True)
    control_plane_db = db_directory / "control-plane.db"
    if control_plane_db.exists():
        raise FileExistsError(f"demo directory must be empty: {db_directory}")

    context = EvidenceContext(agent_id="langchain-campaign-agent", deployment_digest="sha256:langchain-demo-v1")
    adapter = LangChainFrameworkAdapter(
        LangChainInvestigationContext(
            agent_ref="langchain-campaign-agent",
            scope_ref="tenant:acme",
            execution_fingerprint="sha256:langchain-demo-v1",
        )
    )

    records = []
    for index, query in enumerate(_QUERIES):
        native_run = run_campaign_agent(query)
        document = adapter.project(native_run, investigation_id=f"investigation_{index}")
        records.append(
            TraceLearningRecord(
                trace_id=f"trace-{index}",
                context=context,
                recorded_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=index),
                successful=document.status == "completed",
                pattern_key="hedged-click-count",
                safe_summary=str(native_run["input"]),
                investigation_digest=document.digest(),
                verified_success=bool(document.execution_context["verified_success"]),
            )
        )

    cohorts = reserve_later_held_out_cohort(tuple(records), held_out_count=2)
    candidate = (
        CandidateMiner(
            minimum_successes=3,
            drafter=lambda pattern: "State the click count plainly, with no hedging.",
        )
        .mine(cohorts.mining_records)[0]
        .candidate
    )

    dataset = EvaluationDataset(
        dataset_id="langchain-campaign-heldout",
        version="2026-01-01",
        cases=tuple(
            EvaluationCase(
                case_id=record.trace_id,
                inputs={"query": record.safe_summary},
                expected="plain",
                source_trace_id=record.trace_id,
                source_investigation_digest=record.investigation_digest,
            )
            for record in cohorts.held_out_records
        ),
    )
    policy = PromotionPolicy(
        policy_version="policy-v1",
        primary_metric="plainness",
        minimum_primary_improvement=0.5,
    )
    plane = LearningControlPlane(
        policy=policy,
        evaluation_backend=LocalEvaluationBackend(),
        repository=SQLiteControlPlaneRepository(control_plane_db),
    )
    plane.register_candidate(candidate, context)
    job = plane.create_job(
        candidate_id=candidate.candidate_id,
        evaluation_id="langchain-eval-1",
        context=context,
        dataset=dataset,
    )

    def run_one(case: EvaluationCase, variant: EvaluationVariant) -> str:
        native_run = run_campaign_agent(case.inputs["query"], guidance=variant.advisory_skill)
        return "hedged" if str(native_run["output"]).startswith("Well,") else "plain"

    def metric(case: EvaluationCase, output: str) -> dict[str, float]:
        return {"plainness": 1.0 if output == case.expected else 0.0}

    worker_result = await OfflineEvaluationWorker(plane, run_one=run_one, metric=metric).run_pending()
    if worker_result.ready_for_review_job_ids != (job.job_id,):
        raise RuntimeError("demo candidate did not pass the offline gate")
    plane.review_job(
        job.job_id,
        reviewer_id="local-reviewer",
        approved=True,
        reason="Held-out plainness improved without regressions.",
    )
    authorization = plane.authorize_delivery(
        job.job_id,
        scope_ref="tenant:acme",
        expires_at=datetime.now(UTC) + timedelta(days=1),
    )
    receipt = adapter.deliver(authorization, candidate)
    plane.record_activation_receipt(receipt)
    audit_record = plane.get_job_audit_record(job.job_id)
    if audit_record.job.decision is None:
        raise RuntimeError("demo job has no gate decision")

    return {
        "candidate_id": candidate.candidate_id,
        "job_id": job.job_id,
        "authorization_id": authorization.authorization_id,
        "receipt_id": receipt.receipt_id,
        "investigation_digests": list(audit_record.job.decision.investigation_digests),
        "control_plane_db": str(control_plane_db),
    }


def main() -> None:
    """Run the demo with an explicitly selected empty local directory."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--db-directory", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run_demo(args.db_directory)), indent=2))


if __name__ == "__main__":
    main()
