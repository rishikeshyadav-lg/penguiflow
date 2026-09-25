# Plugging an agent into the judge kit

Every step of the learning loop trusts the judge. Mining learns only from verified runs, and the
gate promotes a candidate only if it scores better when both arms are re-judged. The judge kit
(`learning_control_plane.judging`) holds the generic judging rules, so an integration supplies
only its domain parts. The runnable reference is `examples/lcp_plain_python_agent/`, a plain-Python
agent with no framework.

## 1. Describe each run as an `AgentRun`

```python
AgentRun(
    question="How many units did North Store sell?",
    steps=[AgentStep("query_stock", {"store": "North Store", "field": "units"}, {"rows": [...]})],
    final_answer="North Store sold 1,200 units.",
    rendered=[RenderedOutput("table", {"columns": [...], "rows": [...]})],
)
```

`AgentStep.error` is set when a call raised; a result mapping with an `"error"` key also counts as
a failed call. `rendered` holds the tables and reports shown beside the answer, because values
shown only there still count. PenguiFlow integrations use `agent_run_from_trajectory`.

## 2. Write a `DomainJudge`

The protocol has four methods:

| Method | Returns |
|---|---|
| `step_evidence(run)` | `SafeStepEvidence` per step: an allowlisted node name, safe argument facts, and a `tool_execution` check. Use `recovered_step_indexes` so a failure the agent moved past is `passed` with `tool_error_recovered`. |
| `clarification_candidates(run)` | The entity names a lookup found, when no data was fetched; the ladder checks whether the answer asks the user to choose between them. |
| `service_unavailable(run)` | True when a tool the question needs reported itself switched off. |
| `judge(run, reference)` | A `RubricJudgment` (numerical, completeness, scope, grounding, hard failures), or None when this kind of question cannot be checked. |

The kit's helpers do the generic work inside `judge`:

- `stated_values(answer, vocabulary, rendered, implicit_metric=...)` reads the values an answer
  states. `MetricVocabulary` gives the names your answers use for each metric.
- `Expectation` plus `judge_expectation(..., tolerance=...)` matches those values to the rows and
  metrics the question requires. Use `RelativeTolerance` or your own `Tolerance`.
- `question_scope_check` checks the answer names what the question named. Pass `suffix_patterns`
  for name suffixes answers drop, such as date ranges.
- `no_data_judgment` handles a scope your reference found empty.
- `states_unsupported_benchmark` fails comparisons with a "typical" level that no tool returned.

## 3. Build the ladder, and optionally a reference and a meaning check

```python
ladder = OutcomeLadder(
    InventoryJudge(),
    rubric=None,                      # the default five-criterion rubric
    extra_hard_failure_codes=(),      # your own hard-failure codes
    empty_lookups=EmptyLookupRule(...),  # recognise "not found" confirmed by empty lookups
)
verification = ladder.judge(run, reference)
```

The ladder decides in a fixed order: a missing answer fails; a truncated answer is an agent error;
a clarification, a confirmed "no data" and a switched-off service are handled correctly; then the
domain judgment; then meaning findings, which can only fail a run. The outcome is recorded on the
verification (`verification.outcome`).

- A `ReferenceBuilder` is an async callable `run -> reference`. It computes the judge's own answer
  independently of the agent's calls, so the verdict is not circular.
- A `MeaningCheck(judge, reads=3)` asks a `MeaningJudge` yes/no questions several times and
  averages the answers. `TypeSafeMeaningJudge.from_environment()` is available with the
  `lcp-typesafe` extra, only when `TYPESAFE_GOVERNANCE_ACKNOWLEDGED` and `TYPESAFE_API_KEY` are set.

## 4. Publish after the turn

```python
publisher = RunPublisher(
    MlflowAttachmentPublisher(),
    judge=ladder,
    assessment_publisher=MlflowAssessmentPublisher(
        readiness=MlflowTraceReadiness(), pending_queue=PendingAssessmentQueue("pending.jsonl")
    ),
    reference_builder=my_reference,
    meaning=MeaningCheck(TypeSafeMeaningJudge.from_environment()),
    signature=SignatureRules(left_out_prefixes=("render_",)),
)
await publisher.publish_after_turn(run, RunContext(...))
```

Call it only once the final answer exists. PenguiFlow agents use `TurnStash` to hold the
trajectory until then. The publisher runs the reference and the meaning check with timeouts and
never raises into the turn. The assessment waits for the trace to close. If the trace does not
close in time, the assessment is queued, and `OfflineEvaluationWorker(pending_assessments=...,
assessment_publisher=...)` publishes it on its next pass.

`SignatureRules` decides which steps make up the step signature a run is mined under. Leave out
failed and recovered calls and presentation calls, and give the same name to calls that do the
same job, so repeated work forms a repeated pattern.

## 5. Gate with the judge

```python
policy = verification_policy("my-policy-v1", minimum_verified_improvement=0.1)
metric = VerificationMetric(ladder.judge, to_run=lambda case, output: output, reference=my_reference)
job = await plane.run_job(job_id, run_one, metric, golden_report=report)
```

The policy requires more verified runs and no more hard failures. Both are measured only over pairs
judged on both arms, so a run handled correctly on one arm is excluded, not counted as a failure.
A policy that requires a golden set refuses to run without a passing `GoldenReport` (see
[the golden-set protocol](golden_set.md)).
