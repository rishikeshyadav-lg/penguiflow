# agent-evals

Agent-agnostic evaluation. Bring any agent (any framework, or a plain function), a fixed dataset and
a scorer; get repeatable runs, honest statistics and a report. This package has **no required
dependencies** and imports nothing from any agent framework.

In-house: install the pinned source archive of the owner's fork (the campaign repo pins it the same way). It is
not published to PyPI, and that is a decision.

```bash
uv pip install "agent-evals @ https://github.com/rishikeshyadav-lg/penguiflow/archive/<commit>.tar.gz#subdirectory=packages/agent-evals"
```

## A few lines to evaluate any agent

```python
import asyncio
from agent_evals import (DatasetManifest, EvaluationCase, EvaluationDataset, EvaluationVariant, ExactMatch,
                         GenericStep, GenericTrajectory, PredictionResult, RunRecord, ToolSelection,
                         build_report, report_markdown, run_suite)

def run_one(case, variant):                       # 1. call your agent, return a PredictionResult
    answer, calls = my_agent(case.inputs["question"])
    steps = [GenericStep(tool, args) for tool, args in calls]
    return PredictionResult(answer=answer, trajectory=GenericTrajectory(case.inputs["question"], steps, answer))

cases = [EvaluationCase("c1", {"question": "..."}, expected={"answer": "...", "tools": ["lookup"]})]  # 2. cases
dataset = EvaluationDataset("mine", "v1", cases)
manifest = DatasetManifest.from_dataset(dataset, suite="regression")

async def main():                                 # 3. run, score, report
    run = await run_suite(dataset, manifest, [EvaluationVariant("mine")], run_one, [answered, ToolSelection()],
                          metric_id="mine", metric_version="1")
    record = RunRecord.from_run(run, manifest, "mine", run_id="r1")
    print(report_markdown(build_report(run, manifest, "mine", record=record)))
```

Runnable versions: `examples/agent_evals_quickstart/` (a plain function, offline) and
`examples/agent_evals_live_langchain/` (a real LangChain agent on a model endpoint).

## What it gives you

Four layers, each with its own scorers, and a report that lists what could not be measured instead of hiding it:

- **Outcome:** exact, contains, regex and numeric scorers; state checks; weighted partial credit; pass@k and pass^k.
- **Trajectory:** tool selection, call order (exact, in order, any order), argument correctness, invariants
  (required, forbidden, allowed, capped, tracked), and an audit signal for success without the right tools.
- **Operational:** p50/p95/p99 latency and cost with intervals, step counts and bands, efficiency, loop detection.
- **Policy:** violations veto success; a flag names the cases.

Also: datasets on disk with frozen manifests, seeded disjoint splits, regression and capability suites, repeated
resumable runs, golden trajectories and shadow comparison, threshold calibration from a baseline, an optional
domain-judge seam with an agreement harness, and an optional MLflow log (the `mlflow` extra).

## What it is, and is not

- **Is:** an evaluation library with no required dependencies that imports nothing from any agent framework.
- **Is not:** an agent framework, a tracing system, a sandbox, or a learning loop. It detects and reports policy
  violations; it cannot prevent them, so enforce destructive-action rules in your agent's execution path. The
  learning control plane (`learning-control-plane`) is one consumer of this package.
- **Experimental:** plan adherence and multi-step coherence (judge-backed). Their agreement with human labels has
  not been measured, and a report that uses them says so.

## Status

`0.1.0`, in-house (not published to PyPI, by the owner's decision). Design, milestones and what is deliberately not built:
[`docs/evaluation-framework.md`](https://github.com/hurtener/penguiflow/blob/main/docs/evaluation-framework.md)
and [`docs/evaluation-roadmap.md`](https://github.com/hurtener/penguiflow/blob/main/docs/evaluation-roadmap.md).
How it is distributed, and the checklist kept in case that changes: `docs/agent-evals-publishing-checklist.md`.
