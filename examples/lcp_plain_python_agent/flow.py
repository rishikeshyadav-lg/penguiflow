"""Plug a plain-Python agent into the learning control plane, from judging its runs to human review.

The agent answers stock questions about a small in-memory inventory with three tools. It uses no
framework: each run is written down as an `AgentRun`. A fixed script decides its tool calls, so
the example is deterministic and needs no model, no MLflow and no network.

The runs cover the mistakes the judge kit exists for:
- a tool error the agent recovered from (a probe of a field that does not exist);
- an answer cut off mid-sentence (an agent error, never a success);
- a clarification (asking which of two matching stores was meant: handled correctly);
- a shortened list (the "... [N more items]" note is not an item).

The example judges each run with the outcome ladder and an inventory `DomainJudge`, publishes the
redacted investigations to a local store, mines the repeated verified pattern, drafts a candidate
skill, gates baseline against candidate by re-judging both with `VerificationMetric`, checks the
judge against a synthetic golden set first, and ends with the candidate in the review queue.

    uv run python examples/lcp_plain_python_agent/flow.py [--store-directory DIR]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from learning_control_plane.contracts.evidence import EvidenceContext
from learning_control_plane.contracts.investigation import SourceTraceRef
from learning_control_plane.control_plane import LearningControlPlane
from learning_control_plane.evaluation import (
    EvaluationCase,
    EvaluationDataset,
    EvaluationVariant,
    LocalEvaluationBackend,
)
from learning_control_plane.evaluation.verification import SafeStepEvidence, VerificationCheck
from learning_control_plane.integrations.generic import RunContext, RunPublisher
from learning_control_plane.judging import (
    OVERALL,
    RECOVERED_STEP_CODE,
    AgentRun,
    AgentStep,
    Expectation,
    MetricVocabulary,
    OutcomeLadder,
    RelativeTolerance,
    RubricJudgment,
    SignatureRules,
    VerificationMetric,
    judge_expectation,
    outcome_of,
    question_scope_check,
    recovered_step_indexes,
    shown_to_user,
    stated_values,
    step_failed,
    strip_shortened_list_note,
    verification_policy,
)
from learning_control_plane.judging.golden import GoldenLabel, run_golden_set
from learning_control_plane.mining import CandidateMiner, TracePattern
from learning_control_plane.providers.local_store import LocalInvestigationStore

# The in-memory inventory the agent's tools read, and the judge's reference reads independently.
STOCK: Mapping[str, Mapping[str, float]] = {
    "North Store": {"units": 1200, "revenue": 9600.0},
    "North Outlet": {"units": 450, "revenue": 3150.0},
    "South Store": {"units": 800, "revenue": 6400.0},
    "East Depot": {"units": 1500, "revenue": 12000.0},
    "West Depot": {"units": 300, "revenue": 2100.0},
}
SKUS: Mapping[str, Sequence[str]] = {"East Depot": [f"ED-{number}" for number in range(1, 11)]}
SKUS_SHOWN = 3
TOOLS = frozenset({"list_stores", "query_stock", "list_skus"})
DATA_TOOLS = frozenset({"query_stock", "list_skus"})
VOCABULARY = MetricVocabulary(aliases={"units": ("units sold", "units"), "revenue": ("revenue", "sales")})
SIGNATURE = SignatureRules(renamed={"list_stores": "lookup"})
UNITS_QUESTION = re.compile(r"^How many units did (?P<store>.+) sell\?$")
REVENUE_QUESTION = re.compile(r"^What revenue did (?P<store>.+) make\?$")
SKU_QUESTION = re.compile(r"^Which SKUs does (?P<store>.+) carry\?$")
VAGUE_QUESTION = re.compile(r"^How is (?P<prefix>\w+) doing\?$")
DEPLOYMENT = "sha256:inventory-agent-v1"
AGENT_CONTEXT = EvidenceContext(agent_id="inventory_agent", deployment_digest=DEPLOYMENT)


# The agent's tools: plain functions over the inventory.


def list_stores(prefix: str) -> dict[str, Any]:
    """Return the stores whose name starts with the prefix."""

    return {"values": [name for name in STOCK if name.startswith(prefix)]}


def query_stock(store: str, field: str) -> dict[str, Any]:
    """Return one stock figure for one store, or an error for a field that does not exist."""

    if field not in ("units", "revenue"):
        return {"error": f"unknown field: {field}"}
    if store not in STOCK:
        return {"rows": []}
    return {"rows": [{"store": store, field: STOCK[store][field]}]}


def list_skus(store: str) -> dict[str, Any]:
    """Return a store's SKUs; a long list keeps its first items and a note of how many were cut."""

    skus = list(SKUS.get(store, ()))
    if len(skus) <= SKUS_SHOWN:
        return {"skus": skus}
    return {"skus": [*skus[:SKUS_SHOWN], f"... [{len(skus) - SKUS_SHOWN} more items]"]}


TOOL_FUNCTIONS: Mapping[str, Callable[..., dict[str, Any]]] = {
    "list_stores": list_stores,
    "query_stock": query_stock,
    "list_skus": list_skus,
}


class InventoryAgent:
    """A scripted tool-calling agent; an advisory skill changes how it queries and how it answers."""

    def __init__(self, skill: str | None = None) -> None:
        self.skill = skill

    def run(self, question: str) -> AgentRun:
        """Answer one question and record the run."""

        steps: list[AgentStep] = []

        def call(tool: str, **args: Any) -> dict[str, Any]:
            result = TOOL_FUNCTIONS[tool](**args)
            steps.append(AgentStep(tool, args, result))
            return result

        if match := UNITS_QUESTION.match(question) or REVENUE_QUESTION.match(question):
            store = match["store"]
            field = "units" if question.startswith("How many units") else "revenue"
            if self.skill is None:
                # Without the skill the agent first guesses a field name that does not exist.
                call("query_stock", store=store, field=f"{field}_total")
            value = call("query_stock", store=store, field=field)["rows"][0][field]
            if field == "units":
                answer = f"{store} sold {value:,.0f} units."
            elif self.skill is None:
                # Without the skill a revenue answer runs out mid-sentence.
                answer = f"{store} made ${value:,.0f} in revenue, which compares with"
            else:
                answer = f"{store} made ${value:,.0f} in revenue."
        elif match := SKU_QUESTION.match(question):
            store = match["store"]
            skus = strip_shortened_list_note(call("list_skus", store=store)["skus"])
            answer = f"{store} carries {', '.join(str(sku) for sku in skus)} and more."
        elif match := VAGUE_QUESTION.match(question):
            names = call("list_stores", prefix=match["prefix"])["values"]
            answer = f"I found {' and '.join(names)}. Which one do you mean?"
        else:
            answer = "I can answer questions about units, revenue and SKUs."
        return AgentRun(question=question, steps=steps, final_answer=answer)


# The judge: the kit's outcome ladder with the inventory's domain knowledge.


class InventoryJudge:
    """The inventory's `DomainJudge`: safe step evidence, clarification candidates, and value checks."""

    def __init__(self) -> None:
        self._tolerance = RelativeTolerance(default=0.005)

    def step_evidence(self, run: AgentRun) -> Sequence[SafeStepEvidence]:
        """Return content-free checks for each tool step; a failure the agent moved past counts as recovered."""

        recovered = recovered_step_indexes(run.steps, is_data_step=lambda step: step.tool in DATA_TOOLS)
        evidence = []
        for index, step in enumerate(run.steps):
            if index in recovered:
                check = VerificationCheck("tool_execution", "passed", (RECOVERED_STEP_CODE,))
            elif step_failed(step):
                check = VerificationCheck("tool_execution", "failed", ("tool_returned_error",))
            else:
                check = VerificationCheck("tool_execution", "passed", ("tool_completed",))
            field = step.args.get("field")
            evidence.append(
                SafeStepEvidence(
                    step_index=index,
                    node_name=step.tool if step.tool in TOOLS else "redacted_node",
                    argument_facts={"field": field} if field in ("units", "revenue") else {},
                    result_checks=(check,),
                )
            )
        return evidence

    def clarification_candidates(self, run: AgentRun) -> Sequence[str]:
        """Return the store names a lookup found, when no figure was fetched."""

        if any(step.tool in DATA_TOOLS for step in run.steps):
            return ()
        return [name for step in run.steps if step.tool == "list_stores" for name in step.result["values"]]

    def service_unavailable(self, run: AgentRun) -> bool:
        """Return False: every inventory tool is local."""

        return False

    def judge(self, run: AgentRun, reference: Any) -> RubricJudgment | None:
        """Check the answer's figure or list against the judge's own reference."""

        shown = shown_to_user(run)
        if match := UNITS_QUESTION.match(run.question) or REVENUE_QUESTION.match(run.question):
            store = match["store"]
            metric = "units" if run.question.startswith("How many units") else "revenue"
            expectation = Expectation(rows={OVERALL: {metric: reference[store][metric]}}, source="reference")
            judgment = judge_expectation(
                expectation,
                stated_values(run.final_answer or "", VOCABULARY, run.rendered, implicit_metric=metric),
                shown_text=shown,
                tolerance=self._tolerance,
                required_metrics=[metric],
            )
            scope = question_scope_check(shown, question_terms=[store])
            return RubricJudgment(
                judgment.numerical, judgment.completeness, scope, judgment.grounding, judgment.hard_failures
            )
        if match := SKU_QUESTION.match(run.question):
            # The "... [N more items]" note in the tool result is not a SKU the answer must name.
            named_all = all(sku in shown for sku in reference)
            return RubricJudgment(
                numerical=VerificationCheck("factual_numerical_correctness", "not_applicable", ("no_values_asked",)),
                completeness=VerificationCheck(
                    "completeness",
                    "passed" if named_all else "failed",
                    ("all_listed_items_named",) if named_all else ("listed_items_missing",),
                ),
                scope=question_scope_check(shown, question_terms=[match["store"]]),
                grounding=VerificationCheck("evidence_grounding", "passed", ("items_match_tool_results",)),
            )
        return None


async def inventory_reference(run: AgentRun) -> Any:
    """Compute the judge's own answer from the inventory, independently of the agent's calls."""

    if match := SKU_QUESTION.match(run.question):
        return list(SKUS.get(match["store"], ()))[:SKUS_SHOWN]
    return {store: dict(figures) for store, figures in STOCK.items()}


LADDER = OutcomeLadder(InventoryJudge())


def judge_run(run: AgentRun, reference: Any) -> Any:
    """Judge one run, for the golden set and the gate."""

    return LADDER.judge(run, reference)


# The loop: judge and publish past runs, mine, gate, review.

HISTORY = (
    "How many units did North Store sell?",
    "How many units did South Store sell?",
    "How many units did East Depot sell?",
    "What revenue did West Depot make?",
    "How is North doing?",
    "Which SKUs does East Depot carry?",
)
HELD_OUT = (
    "How many units did West Depot sell?",
    "What revenue did North Store make?",
    "What revenue did South Store make?",
    "How is North doing?",
)
GOLDEN_LABELS = (
    ("How many units did North Store sell?", "verified"),
    ("What revenue did West Depot make?", "agent_error"),
    ("How is North doing?", "handled_correctly"),
    ("Which SKUs does East Depot carry?", "verified"),
)


def _context(index: int, question: str) -> RunContext:
    kind = "units" if "units" in question else "revenue" if "revenue" in question else "other"
    return RunContext(
        source_trace_ref=SourceTraceRef("local", "inventory-demo", f"run-{index}", DEPLOYMENT),
        agent_ref="inventory_agent",
        scope_ref="tenant:demo",
        execution_fingerprint=DEPLOYMENT,
        started_at=datetime(2026, 9, 1, tzinfo=UTC) + timedelta(minutes=index),
        provider_ref="plain-python",
        allowed_node_names=TOOLS,
        intent_descriptor={"class": kind},
    )


async def _outcome(run: AgentRun) -> str:
    return outcome_of(judge_run(run, await inventory_reference(run)))


def draft_skill(pattern: TracePattern) -> str:
    """Draft advice from a verified pattern; a real integration would use a constrained model drafter."""

    return "Query the exact field (units or revenue) directly, and state each figure in one complete sentence."


async def run_example(store_directory: Path) -> dict[str, Any]:
    """Run the whole loop and return a summary of what each stage produced."""

    store = LocalInvestigationStore(store_directory)
    publisher = RunPublisher(store, judge=LADDER, reference_builder=inventory_reference, signature=SIGNATURE)
    history_outcomes: dict[str, str] = {}
    for index, question in enumerate(HISTORY):
        publication = await publisher.publish_after_turn(InventoryAgent().run(question), _context(index, question))
        if publication is None:
            raise RuntimeError(f"run {index} was not published")
        history_outcomes[question] = publication.document.extensions["learning.verification"]["outcome"]

    records = store.load_records()
    mined = CandidateMiner(minimum_successes=3, drafter=draft_skill).mine(records)
    if not mined:
        raise RuntimeError("no repeated verified pattern was found")
    candidate = mined[0].candidate

    golden = await run_golden_set(
        [GoldenLabel(question, outcome) for question, outcome in GOLDEN_LABELS],  # type: ignore[arg-type]
        lambda label: InventoryAgent().run(label.case_key),
        judge_run,
        reference=inventory_reference,
    )

    plane = LearningControlPlane(
        policy=verification_policy("inventory-verification-v1", minimum_verified_improvement=0.2),
        evaluation_backend=LocalEvaluationBackend(),
    )
    plane.register_candidate(candidate, AGENT_CONTEXT)
    dataset = EvaluationDataset(
        dataset_id="inventory-heldout",
        version="v1",
        cases=tuple(
            EvaluationCase(case_id=f"held-out-{index}", inputs={"question": question})
            for index, question in enumerate(HELD_OUT)
        ),
    )
    job = plane.create_job(
        candidate_id=candidate.candidate_id, evaluation_id="inventory-eval-1", context=AGENT_CONTEXT, dataset=dataset
    )

    def run_one(case: EvaluationCase, variant: EvaluationVariant) -> AgentRun:
        return InventoryAgent(variant.advisory_skill).run(str(case.inputs["question"]))

    metric = VerificationMetric(judge_run, lambda case, output: output, inventory_reference)
    gated = await plane.run_job(job.job_id, run_one, metric, golden_report=golden)
    if gated.decision is None:
        raise RuntimeError("the gate made no decision")
    verified = next(s for s in gated.decision.metric_summaries if s.specification.name == "verified_success")
    held_out_outcomes = {}
    for pair in gated.evaluation.case_results if gated.evaluation else ():
        question = pair.baseline.output.question
        held_out_outcomes[question] = {
            "baseline": await _outcome(pair.baseline.output),
            "candidate": await _outcome(pair.candidate.output),
        }

    return {
        "history_outcomes": history_outcomes,
        "published_documents": len(store.documents()),
        "mined_pattern": mined[0].pattern.pattern_key,
        "candidate_id": candidate.candidate_id,
        "golden_set": {"agreement": golden.agreement, "total": len(golden.results), "passed": golden.passed},
        "gate": {
            "state": gated.state,
            "approved": gated.decision.approved,
            "reasons": list(gated.decision.reasons),
            "verified_success": {
                "baseline": verified.baseline_mean,
                "candidate": verified.candidate_mean,
                "excluded_case_ids": list(verified.excluded_case_ids),
            },
        },
        "review_queue": [item.candidate.candidate_id for item in plane.list_review_queue()],
        "held_out_outcomes": held_out_outcomes,
    }


def main() -> None:
    """Run the example in a given directory, or in a fresh temporary one."""

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store-directory", type=Path, help="Where to write the investigation documents")
    args = parser.parse_args()
    if args.store_directory is not None:
        summary = asyncio.run(run_example(args.store_directory))
    else:
        with tempfile.TemporaryDirectory(prefix="lcp-plain-python-") as directory:
            summary = asyncio.run(run_example(Path(directory)))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
