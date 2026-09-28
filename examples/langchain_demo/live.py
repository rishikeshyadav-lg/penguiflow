"""Run the learning-control-plane MVP against a REAL LangChain agent driven by a real model.

Opt-in and network-/credential-dependent, unlike `flow.py` (LLM-free): this is what actually
proves `LangChainFrameworkAdapter` against `langchain.agents.create_agent(...)`'s real output --
a live `{"messages": [...]}` run, not a hand-built dict. Needs a Databricks CLI profile with access
to a model serving endpoint (see `~/.databrickscfg`); defaults to the `penguiflow-oauth` profile
and the `databricks-claude-haiku-4-5` endpoint, both already used elsewhere in this repo.

    DATABRICKS_CONFIG_PROFILE=penguiflow-oauth \\
        uv run python -m examples.langchain_demo.live --db-directory /private/tmp/penguiflow-lcp-langchain-live

Whatever verdict the gate reaches is reported as-is: if the model already answers plainly without
guidance, the held-out improvement may not clear the policy's bar, and the demo reports that
rejection rather than tuning the prompt to force an approval.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from databricks_langchain import ChatDatabricks
from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage

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
    LangChainGuidanceInjector,
    LangChainInvestigationContext,
)
from learning_control_plane.mining import (
    CandidateMiner,
    TraceLearningRecord,
    reserve_later_held_out_cohort,
)

from .agent import _CAMPAIGN_CLICKS, _campaign_id_in, lookup_campaign_clicks

_MODEL_ENDPOINT = "databricks-claude-haiku-4-5"
_SYSTEM_PROMPT = "You answer questions about campaign click counts using the lookup_campaign_clicks tool."
_QUERIES = [f"how many clicks did campaign {campaign_id} get" for campaign_id in list(_CAMPAIGN_CLICKS) * 4][:12]
_PLAIN_ANSWER = re.compile(r"^\d+ clicks\.$")


def _build_agent() -> Any:
    """Build a real LangChain agent backed by a live Databricks-hosted model."""

    return create_agent(
        model=ChatDatabricks(endpoint=_MODEL_ENDPOINT), tools=[lookup_campaign_clicks], system_prompt=_SYSTEM_PROMPT
    )


def _run(agent: Any, query: str, *, guidance: str | None = None) -> dict[str, object]:
    """Run one real turn, injecting guidance through the adapter's own mechanism when given one."""

    messages: list[BaseMessage] = [HumanMessage(content=query)]
    if guidance:
        messages = LangChainGuidanceInjector(guidance=guidance, categories=()).apply(messages)
    return dict(agent.invoke({"messages": messages}))


async def run_demo(db_directory: Path) -> dict[str, object]:
    """Execute the full learning loop against a real LangChain agent and a real model."""

    db_directory.mkdir(parents=True, exist_ok=True)
    control_plane_db = db_directory / "control-plane.db"
    if control_plane_db.exists():
        raise FileExistsError(f"demo directory must be empty: {db_directory}")

    agent = _build_agent()
    context = EvidenceContext(agent_id="langchain-campaign-agent-live", deployment_digest="sha256:langchain-live-v1")
    adapter = LangChainFrameworkAdapter(
        LangChainInvestigationContext(
            agent_ref="langchain-campaign-agent-live",
            scope_ref="tenant:acme",
            execution_fingerprint="sha256:langchain-live-v1",
        )
    )

    records = []
    for index, query in enumerate(_QUERIES):
        native_run = _run(agent, query)
        document = adapter.project(native_run, investigation_id=f"investigation_{index}")
        records.append(
            TraceLearningRecord(
                trace_id=f"trace-{index}",
                context=context,
                recorded_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=index),
                successful=document.status == "completed",
                pattern_key="unstructured-click-count",
                safe_summary=query,
                investigation_digest=document.digest(),
                verified_success=bool(document.execution_context["verified_success"]),
            )
        )

    cohorts = reserve_later_held_out_cohort(tuple(records), held_out_count=6)
    mined = CandidateMiner(
        minimum_successes=3,
        drafter=lambda pattern: "Answer with only the number followed by ' clicks.', with nothing else.",
    ).mine(cohorts.mining_records)
    if not mined:
        raise RuntimeError("no repeated pattern was mined from the live baseline runs")
    candidate = mined[0].candidate

    dataset = EvaluationDataset(
        dataset_id="langchain-live-heldout",
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
    policy = PromotionPolicy(policy_version="policy-v1", primary_metric="plainness", minimum_primary_improvement=0.3)
    plane = LearningControlPlane(
        policy=policy,
        evaluation_backend=LocalEvaluationBackend(),
        repository=SQLiteControlPlaneRepository(control_plane_db),
    )
    plane.register_candidate(candidate, context)
    job = plane.create_job(
        candidate_id=candidate.candidate_id,
        evaluation_id="langchain-live-eval-1",
        context=context,
        dataset=dataset,
    )

    def run_one(case: EvaluationCase, variant: EvaluationVariant) -> str:
        query = str(case.inputs["query"])
        native_run = _run(agent, query, guidance=variant.advisory_skill)
        generic = adapter.to_generic_trajectory(native_run)
        campaign_id = _campaign_id_in(query)
        expected_clicks = _CAMPAIGN_CLICKS.get(campaign_id, 0) if campaign_id else 0
        used_the_tool = any(step.tool == "lookup_campaign_clicks" and not step.error for step in generic.steps)
        answer = (generic.final_answer or "").strip()
        if used_the_tool and _PLAIN_ANSWER.match(answer) and answer.startswith(f"{expected_clicks} "):
            return "plain"
        return "not_plain"

    def metric(case: EvaluationCase, output: str) -> dict[str, float]:
        return {"plainness": 1.0 if output == "plain" else 0.0}

    worker_result = await OfflineEvaluationWorker(plane, run_one=run_one, metric=metric).run_pending()
    gated = worker_result.ready_for_review_job_ids == (job.job_id,)
    decision = plane.get_job(job.job_id).decision

    result: dict[str, object] = {
        "candidate_id": candidate.candidate_id,
        "job_id": job.job_id,
        "gated": gated,
        "baseline_metrics": dict(decision.baseline_metrics) if decision is not None else None,
        "candidate_metrics": dict(decision.candidate_metrics) if decision is not None else None,
        "control_plane_db": str(control_plane_db),
    }
    if not gated:
        result["outcome"] = "rejected_by_gate"
        return result

    plane.review_job(
        job.job_id,
        reviewer_id="local-reviewer",
        approved=True,
        reason="Held-out plainness improved on a live model.",
    )
    authorization = plane.authorize_delivery(
        job.job_id,
        scope_ref="tenant:acme",
        expires_at=datetime.now(UTC) + timedelta(days=1),
    )
    receipt = adapter.deliver(authorization, candidate)
    plane.record_activation_receipt(receipt)

    sample_query = cohorts.held_out_records[0].safe_summary
    before_run = _run(agent, sample_query)
    after_run = _run(agent, sample_query, guidance=candidate.advisory_skill)
    result.update(
        {
            "outcome": "approved_and_delivered",
            "authorization_id": authorization.authorization_id,
            "receipt_id": receipt.receipt_id,
            "sample_query": sample_query,
            "answer_without_skill": adapter.to_generic_trajectory(before_run).final_answer,
            "answer_with_skill": adapter.to_generic_trajectory(after_run).final_answer,
        }
    )
    return result


def main() -> None:
    """Run the live demo with an explicitly selected empty local directory."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--db-directory", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run_demo(args.db_directory)), indent=2))


if __name__ == "__main__":
    main()
