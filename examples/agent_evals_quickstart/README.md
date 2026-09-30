# agent_evals quickstart: evaluate a plain Python function

`flow.py` evaluates an ordinary function (`my_agent`) that calls two tools. The function knows nothing about
`agent_evals`; the file adds a runner, a dataset and scorers, about three dozen lines in all.

```bash
uv run python examples/agent_evals_quickstart/flow.py
```

It prints a Markdown report with the eight-metric scorecard. Metrics nothing produced (plan adherence, which
needs a judge) are listed as "not measured" with the reason, not left out.

What to change for your own agent:

1. Write `run_one(case, variant)` so it calls your agent and returns a `PredictionResult` (answer, and a
   `GenericTrajectory` of the tool calls if you have them).
2. Build an `EvaluationDataset` of `EvaluationCase`s and freeze it with `DatasetManifest.from_dataset`.
3. Pick scorers. `ExactMatch`, `ToolSelection`, `ArgumentCorrectness`, `PolicyVeto` are used here; the rest are
   in `docs/evaluation-framework.md`.

Test: `tests/agent_evals/test_quickstart_example.py` runs this file and checks the report.
