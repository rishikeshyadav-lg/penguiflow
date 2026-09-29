"""Run the learning-control-plane MVP against a REAL PenguiFlow agent driven by a real model.

The PenguiFlow counterpart of `examples/langchain_demo/live.py`: the same mine -> evaluate -> gate
-> review -> authorize -> deliver loop, the same toy campaign-clicks domain, but driving a real
`penguiflow.planner.ReactPlanner` against a live Databricks-hosted model through
`PenguiFlowFrameworkAdapter` instead of the LangChain adapter -- proving the adapter that, until
now, had only ever seen hand-built `Trajectory` objects in tests and the offline demo.

Needs a Databricks CLI profile with access to a model serving endpoint (see `~/.databrickscfg`);
defaults to the `penguiflow-oauth` profile and the `databricks-claude-haiku-4-5` endpoint, both
already used by the LangChain live demo. `DatabricksProvider` does not read `~/.databrickscfg`
profiles itself, so auth is built explicitly from `databricks.sdk.core.Config` -- reading the
`DATABRICKS_CONFIG_PROFILE` environment variable rather than passing `profile=` directly, which
fails here because two profile names in this machine's `~/.databrickscfg` share one host and the
SDK's underlying `databricks auth token --host ...` call can't disambiguate by host alone.

    DATABRICKS_CONFIG_PROFILE=penguiflow-oauth \\
        uv run python -m examples.penguiflow_demo.live --db-directory /private/tmp/penguiflow-lcp-penguiflow-live

Whatever verdict the gate reaches is reported as-is, exactly as the LangChain live demo does.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from databricks.sdk.core import Config

from learning_control_plane.contracts.evidence import EvidenceContext
from learning_control_plane.contracts.investigation import SourceTraceRef
from learning_control_plane.control_plane import LearningControlPlane, PromotionPolicy
from learning_control_plane.control_plane.persistence import SQLiteControlPlaneRepository
from learning_control_plane.control_plane.worker import OfflineEvaluationWorker
from learning_control_plane.evaluation import (
    EvaluationCase,
    EvaluationDataset,
    EvaluationVariant,
    LocalEvaluationBackend,
)
from learning_control_plane.integrations.penguiflow.projector import (
    PenguiFlowFrameworkAdapter,
    PenguiFlowInvestigationContext,
    PenguiFlowInvestigationProjector,
)
from learning_control_plane.mining import (
    CandidateMiner,
    TraceLearningRecord,
    reserve_later_held_out_cohort,
)
from penguiflow.planner import ReactPlanner
from penguiflow.planner.trajectory import Trajectory
from penguiflow.skills.local_store import LocalSkillStore

from .agent import _CAMPAIGN_CLICKS, campaign_agent_catalog

_MODEL_ENDPOINT = "databricks/databricks-claude-haiku-4-5"
_SYSTEM_PROMPT = "You answer questions about campaign click counts using the lookup_campaign_clicks tool."
_QUERIES = [f"how many clicks did campaign {campaign_id} get" for campaign_id in list(_CAMPAIGN_CLICKS) * 4][:12]
_PLAIN_ANSWER = re.compile(r"^\d+ clicks\.$")


def _campaign_id_in(query: str) -> str | None:
    return next((word for word in query.split() if word.isdigit()), None)


def _databricks_llm_config() -> dict[str, Any]:
    """Build the native-LLM config dict for `ReactPlanner`, authenticating via the env-selected
    Databricks CLI profile rather than passing `profile=` directly (see module docstring).
    """

    cfg = Config()
    return {
        "model": _MODEL_ENDPOINT,
        "host": cfg.host,
        "token_provider": lambda: cfg.authenticate()["Authorization"].removeprefix("Bearer "),
    }


async def _run(query: str, *, guidance_hook: Any | None = None) -> Trajectory:
    """Run one real turn and return the captured `Trajectory`.

    `ReactPlanner` takes `llm_context_hooks` at construction time, so a fresh planner is built per
    call -- cheap, since construction does no network I/O -- letting each call choose whether the
    mined guidance hook is attached. `run()` itself returns a `PlannerFinish`/`PlannerPause`, not
    the `Trajectory`; the real trajectory only arrives via `on_trajectory_complete`.
    """

    captured: list[Trajectory] = []
    planner = ReactPlanner(
        llm=_databricks_llm_config(),
        use_native_llm=True,
        catalog=campaign_agent_catalog(),
        max_iters=4,
        system_prompt_extra=(
            f"{_SYSTEM_PROMPT} When the prompt's context includes an 'advisory_guidance' field, "
            "follow it exactly when composing your final answer."
        ),
        on_trajectory_complete=captured.append,
        llm_context_hooks=[guidance_hook] if guidance_hook is not None else None,
    )
    await planner.run(query)
    if not captured:
        raise RuntimeError(f"planner produced no trajectory for query: {query!r}")
    return captured[-1]


async def run_demo(db_directory: Path) -> dict[str, object]:
    """Execute the full learning loop against a real PenguiFlow agent and a real model."""

    db_directory.mkdir(parents=True, exist_ok=True)
    control_plane_db = db_directory / "control-plane.db"
    skills_db = db_directory / "skills.db"
    if control_plane_db.exists() or skills_db.exists():
        raise FileExistsError(f"demo directory must be empty: {db_directory}")

    context = EvidenceContext(agent_id="penguiflow-campaign-agent-live", deployment_digest="sha256:penguiflow-live-v1")
    adapter = PenguiFlowFrameworkAdapter(
        PenguiFlowInvestigationProjector(
            PenguiFlowInvestigationContext(
                source_trace_ref=SourceTraceRef(
                    tracking_store_ref="penguiflow-live",
                    experiment_id="penguiflow-live-experiment",
                    mlflow_trace_id="penguiflow-live-trace",
                    deployment_ref="sha256:penguiflow-live-v1",
                ),
                agent_ref="penguiflow-campaign-agent-live",
                scope_ref="tenant:acme",
                execution_fingerprint="sha256:penguiflow-live-v1",
                started_at=datetime.now(UTC),
            )
        ),
        LocalSkillStore(db_path=skills_db),
    )

    records = []
    for index, query in enumerate(_QUERIES):
        trajectory = await _run(query)
        document = adapter.project(trajectory, investigation_id=f"investigation_{index}")
        generic = adapter.to_generic_trajectory(trajectory)
        # `document.execution_context["verified_success"]` is always False here: PenguiFlow's
        # projector delegates that judgment to an optional `verification_projector` callback (the
        # full rubric-based verifier campaign wires up), which this toy demo doesn't build. A
        # simple, self-contained heuristic -- the tool ran without error and there's an answer --
        # is the right substitute for a demo, matching what the LangChain adapter's own
        # `verified_success` computes inline.
        used_the_tool_successfully = any(
            step.tool == "lookup_campaign_clicks" and not step.error for step in generic.steps
        )
        records.append(
            TraceLearningRecord(
                trace_id=f"trace-{index}",
                context=context,
                recorded_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=index),
                successful=document.status == "completed",
                pattern_key="unstructured-click-count",
                safe_summary=query,
                investigation_digest=document.digest(),
                verified_success=generic.final_answer is not None and used_the_tool_successfully,
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
        dataset_id="penguiflow-live-heldout",
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
        evaluation_id="penguiflow-live-eval-1",
        context=context,
        dataset=dataset,
    )

    async def run_one(case: EvaluationCase, variant: EvaluationVariant) -> str:
        query = str(case.inputs["query"])
        guidance_hook = (
            adapter.attach_guidance(guidance=variant.advisory_skill, categories=()) if variant.advisory_skill else None
        )
        trajectory = await _run(query, guidance_hook=guidance_hook)
        generic = adapter.to_generic_trajectory(trajectory)
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
    before_trajectory = await _run(sample_query)
    after_hook = adapter.attach_guidance(guidance=candidate.advisory_skill, categories=())
    after_trajectory = await _run(sample_query, guidance_hook=after_hook)
    result.update(
        {
            "outcome": "approved_and_delivered",
            "authorization_id": authorization.authorization_id,
            "receipt_id": receipt.receipt_id,
            "sample_query": sample_query,
            "answer_without_skill": adapter.to_generic_trajectory(before_trajectory).final_answer,
            "answer_with_skill": adapter.to_generic_trajectory(after_trajectory).final_answer,
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
