# The agent-agnostic evaluation framework (`agent-evals`)

Status: **E0–E11 done (offline)** (this contract, the package, the neutral core, the general comparison, repeated
resumable runs, public statistics, datasets and suites, and the outcome, trajectory and operational scorers, the policy layer, golden trajectories and shadow comparison, reports with the eight-metric scorecard, and the judge seam).
E12 onward is planned in `evaluation-roadmap.md`. Written 2026-09-29.

`agent-evals` lets you evaluate any agent, whatever framework it is written in: give it a dataset, a
function that runs the agent, and one or more scorers; it runs, compares variants, and reports with
statistics that account for the agent's own run-to-run noise. It depends on nothing. The learning control
plane (LCP) is one consumer of it and uses it to decide whether an advisory skill may ship.

## 1. Why this exists

The pieces of an evaluation framework already existed in three places that did not share a contract:

| Where | What it had | Why it was not enough |
|---|---|---|
| LCP `evaluation/` | dataset with digest, `RunOne`/`Metric`/`EvaluationBackend`, paired A/B result | a variant could only change a skill string; one metric; sequential; no repeats; statistics locked inside the promotion gate |
| campaign agent | statistics, threshold calibration, verdict wording, the arm runner, the judge, the eval app | tied to the delivery table and campaign rubric; pure maths copied rather than shared |
| `penguiflow/evals/` | JSONL datasets, metric helpers, `llm_judge`, reports | works only with PenguiFlow |

Three unimplemented designs describe parts of the answer: `DUAL_LOCAL_MLFLOW_EVALUATION_BACKENDS.md`
(run/metric/backend protocols), `FRAMEWORK_AGNOSTIC_LEARNING_CONTROL_PLANE.md` §5.7–5.10 (evaluation
package, objective contract, spec, evidence) and `RFC_TRACE_DERIVED_DATASETS_AND_EVALS.md` (dataset and
metric coupling). This document picks one vocabulary and says which parts are in scope.

## 2. What an agent must provide (the whole contract)

An agent takes part by supplying **one callable**:

```python
RunOne = Callable[[EvaluationCase, EvaluationVariant], Any | Awaitable[Any]]
```

It receives a case and a variant and returns *anything*. Scoring works on whatever it returns. To get
the full benefit (tool-call scoring, cost and latency statistics, comparable reports) it should return a
`PredictionResult` (implemented, E1):

| Field | Meaning |
|---|---|
| `status` | `ok`, `failed`, `cancelled` or `paused` |
| `answer` | the final answer text, or `None` |
| `trajectory` | a `GenericTrajectory` of the tool calls made, in order, or `None` |
| `latency_ms`, `cost_usd`, `llm_usage` | measured by the agent; the run backend measures wall-clock latency if absent |
| `error` | why it failed, when it did |
| `extra` | anything else worth keeping (route, trace ids, effective contexts) |

This is the minimal contract the campaign inventory (`FRAMEWORK-NEUTRAL-INVENTORY.md`) found a
non-PenguiFlow agent must meet: an answer, an ordered list of tool calls `{tool, args, result, status}`,
latency, LLM usage, and a way to apply a variant.

`GenericStep` and `GenericTrajectory` (implemented) are the run shape scorers read, so a scorer written
once works on any agent. Each framework supplies one translation function into it; the LCP's PenguiFlow,
LangChain and mock adapters already do.

## 3. Vocabulary, reconciled

| Concept | Name here | Comes from | State |
|---|---|---|---|
| One input and its expectation | `EvaluationCase(case_id, inputs, expected, source_trace_id, source_investigation_digest)` | existing code | implemented |
| A versioned, frozen set of cases | `EvaluationDataset` with `manifest_digest` | existing code | implemented |
| One agent configuration | `EvaluationVariant(variant_id, advisory_skill)`; gains `config` in E1 | existing code | implemented, extended in E1 |
| A paired baseline-vs-candidate run | `EvaluationRequest` | existing code | implemented (no skill rule; see §4) |
| A run over 1..N variants | a general request | new | E1 |
| The result of running an agent | `PredictionResult` | `DUAL_LOCAL_MLFLOW…` | E1 |
| The result of scoring | `ScoreResult(score, feedback, checks)`; a scorer may still return `float` or `Mapping[str, float]` | `DUAL_LOCAL_MLFLOW…` | E1 |
| The result of a whole run | `EvaluationResult` (per-variant case results; paired views on demand) | `DUAL_LOCAL_MLFLOW…` | E1 |
| Who and what evidence belongs to | `EvidenceContext`, `EvidenceEvent`, `EvidenceSink` | LCP contracts | implemented (moved) |
| Statistics | `agent_evals.statistics` | gate + campaign | E3 |
| Dataset files and manifest | loaders, manifest, coupling check | `RFC_TRACE_DERIVED…` | E4 |
| Scorers | built-in library | new | E5 |
| A domain judge | `DomainJudge` protocol | campaign seam | E6 |
| Run record and report | `RunRecord`, `report.md/json` | new | E7 |

From `FRAMEWORK_AGNOSTIC_LEARNING_CONTROL_PLANE.md` §5.7–5.10 this framework adopts, for now, only what
the code can honour: cases pinned by digest, one immutable objective per evaluation (a scorer set with a
version), and evidence that records failures explicitly. **Not adopted yet** (recorded so it is not
lost): `AgentEvaluationPackage` as one versioned unit, `ObjectiveContract`'s `pairing_key`/`required_slices`,
`EvaluationSpec` with a deployment-bundle ref and execution profiles, hiding gold from the runner
(trust-domain split), and `executor_attestation`.

## 4. Layering (implemented)

```
agent_evals                     depends on nothing
   ▲
learning_control_plane          depends on agent_evals (its only required dependency)
   ▲
adapters, campaign agent, apps
```

`agent_evals` owns: cases/datasets/variants/requests/results, the run backend, the generic step contract,
and the evidence context, event and sink *interface*. It imports no agent framework, no MLflow, and not the LCP
(checked by `tests/agent_evals/test_package_contract.py` in a fresh interpreter).

The LCP keeps what is specific to learning: mining, the control plane and gate, review, delivery, skill
semantics, the `FrameworkAdapter`s, and the MLflow/OpenTelemetry sinks with their `lcp.*` names. Two
consequences of the split worth knowing:

- **The skill rule is the LCP's.** "The baseline carries no skill and the candidate carries one" is a
  learning-loop rule, so `learning_control_plane.evaluation.EvaluationRequest` is a subclass that adds it.
  `agent_evals.EvaluationRequest` has no such rule.
- **`lcp.*` naming stays out of the neutral package.** `agent_evals.EvidenceEvent` is the neutral event. The
  LCP's `EvidenceEvent` subclass adds `mlflow_tags()`, `telemetry_attributes()` and friends, and the LCP's sinks
  call module functions of the same names, so they accept a neutral event as readily as the LCP's own.

## 5. Compatibility guarantees (tested)

Everything the LCP and its 25 dependent test and example files import keeps its path and names the very
same class objects, except `EvaluationRequest` and `EvidenceEvent`, which are subclasses (see above). The tests pin, as literals recorded
before the move: two dataset manifest digests, the paired numbers of a fixed run, and the exact events it
emits. A future change that alters a digest or an emitted event fails a test.

## 6. Planned behaviour, by milestone

Milestones E4 onward are laid out, with goals, method, what complete looks like and how each is verified, in
[`evaluation-roadmap.md`](evaluation-roadmap.md); the old E4–E9 entry below is superseded by it.

- **E1 generalise the core (done):** variants with arbitrary `config`; `ComparisonRequest` over 1..N variants;
  several scorers per run (a metric name produced twice fails the case visibly); `PredictionResult` and
  `ScoreResult`; paired views for any two variants. The old paired backend runs through the same engine.
- **E2 run mechanics (done):** `run_repeated(cases, variants, run_one, scorers, RunSettings(...), sink)`.
  - `RunSettings`: `repeats`, `concurrency`, `timeout_s`, `max_retries`, `retry_errors`. Only a
    `TransientError` raised by the runner is retried; a timeout or any other error is recorded, not retried.
  - `JsonlRowSink`: one row per (case, variant, repeat), flushed as it finishes. Its keys are a superset of the
    campaign's `runs.jsonl` (plus `case_id`, `metrics`, `score_details`). Resume skips rows already present;
    a recorded failure counts as done unless `retry_errors` is set. A half-written last line is dropped; a
    damaged line elsewhere, or rows that do not belong to this run, are errors.
  - Rows come back in dataset, variant, repeat order whatever the concurrency. Latency the runner reports is
    kept; otherwise wall-clock latency is measured. A row read back from the file has `result.output = None`
    (the live object is not stored); answer, tool calls, metrics and errors are.
- **E3 statistics (done, one part deferred):** `agent_evals.statistics`, `agent_evals.thresholds`,
  `agent_evals.calibration`.
  - The gate's case-clustered paired bootstrap moved out of `control_plane.py` unchanged (same seed
    derivation, same percentile method); the gate now calls `bootstrap_paired_intervals`. The interval and
    requirement types moved with it and the LCP re-exports the same objects. The prompt is the resampling
    unit because repeats of one prompt are not independent evidence.
  - Also public: `paired_bootstrap` (the plain mean-difference bootstrap used for calibration),
    `standard_error_from_interval`, `prompts_needed_to_clear`, `accuracy_improvement_detectability`, and the
    detection multiplier (1.959963985 + 0.8416212336 = 2.8016).
  - `propose_thresholds` derives a `PromotionThresholds` from a baseline run, with the measurements and one
    sentence per field; `baselines_from_run` builds its input from a `RepeatedRun`. `thresholds_version` names
    a decision by its values.
  - **Deliberate change:** in `PromotionThresholds` the measured fields (accuracy bar, latency and cost noise
    floors, candidate floor, margin) have no defaults; the campaign's 0.115 and the like were one agent's
    calibration. The rule defaults are unchanged. The campaign will supply its own defaults when it adopts the
    package.
  - **Deferred:** a generic `explain_verdict`. The campaign's is written in that agent's metric names and the
    reviewer's units (answers right per 100, seconds, dollars); a generic one needs a decision on how metrics
    declare their units and wording. Not started.
- **E4 datasets and suites (done):** `save_dataset` / `load_dataset` (JSON, JSONL, CSV; byte-identical on a
  save, load, save round trip); `DatasetManifest` (ids, digest, suite, expected metric; never case text);
  `run_suite` checks the digest and the metric before any case runs; `split_by_group` (lifted from the campaign's
  bank builder, reproduces `banks_v1` exactly); `suite_verdict`: a regression suite needs a hard pass rate, a
  capability suite reports partial credit with an interval and no pass or fail. A failed repeat counts as 0.
- **E5 outcome layer (done):** `ExactMatch`, `Contains`, `RegexMatch`, `NumericMatch` (with `Tolerance`),
  `StateCheck` (success judged on the world's state, not the agent's claim), `WeightedRubric` / `score_rubric`
  (partial credit; agrees with the LCP rubric on all 1,024 status combinations), `pass_at_k`, `pass_hat_k`
  (unbiased estimators) and `consistency_label`.
- **E6 trajectory layer (done):** `tool_selection_accuracy`, `sequence_matches` (exact, in_order, any_order),
  `ArgumentCorrectness` (syntactic and semantic levels, codes only), invariants (`Required`, `Forbidden`,
  `AllowedTools`, `MaxCalls`, `Tracked`) and `selection_gap`, an audit signal and not a verdict.
- **E7 operational layer (done):** `operational_summary` (p50/p95/p99 latency and cost with case-clustered
  intervals, cost per task, mean steps), `StepCount`, `ExecutionEfficiency`, expected step bands
  (`step_band_from_baseline`, `WithinStepBand`; under the band alarms like over), `find_loop` / `LoopGuard` /
  `NoLoop` (the same action, or a short cycle, three times back to back), `regression_status` (ok / alert /
  block). A run's latency is the runner's reported figure, else wall-clock measured by the executor.
  **Not built, on purpose:** a per-step `latency_ms` on `GenericStep` (no adapter's native run records step
  timing, so nothing could fill it) and extra `PromotionThresholds` fields for a p95 bound (nothing consumes them
  until the E10 profiles).
- **E8 policy layer (done):** `PolicyCheck` (forbidden tools, an allow-list, an effect registry with a permitted
  scope, recorded denials; codes name kind, tool and step, never arguments), `PolicyVeto` (a violation zeroes the
  success metric and keeps the unvetoed score beside it), `policy_flag`. It detects and reports; it cannot
  prevent, and enforcement belongs in the agent's execution path. The matching gate rule is a written proposal
  only: `evaluation-policy-gate-proposal.md`.
- **E9 golden trajectories, diff, shadow (done):** `GoldenTrajectory` (frozen run under one digest, tied to an
  environment reference; approvals are history, not content), `refresh_golden` (refused without an approval that
  names exactly this replacement), `diff_runs` (answer, tool sequence, changed argument *names*, cost and latency
  deltas, guardrails, policy findings; a golden trajectory can be the reference), `shadow_compare` (a candidate
  run in `dry_run` mode over recorded production inputs, reporting how often an answer-only comparison would
  have been fooled). `dry_run` is a request: the agent must honour it; nothing here sandboxes tools or replays them.
- **E10 reports and the eight-metric scorecard (done):** `RunRecord` (settings, bundle, dataset and metric
  versions, digest, verdict-grade flag); `build_scorecard` with the article's eight entries (success rate, tool
  selection, argument correctness, plan adherence, execution efficiency, cost per task, p95 latency, policy
  violations), each with a 95% range and the runs used and excluded; a metric that could not be computed is
  listed as "not measured" with the reason, never omitted; the success-versus-selection audit line; failed runs
  listed by case and error *type* (messages only on request). `report_json` and `report_markdown` give every
  agent the same structure. Threshold profiles: `EXAMPLE_PROFILES` (`ci_gate`, `production_slo`; the article's
  numbers, labelled as proposals) and `derived_profile` (from `propose_thresholds`); `evaluate_profile` holds a
  run to one. `log_report_to_mlflow` (optional, `agent-evals[mlflow]`) logs the scorecard's own numbers, so a
  report read back from MLflow equals the local one.
  Still open from E3: a generic `explain_verdict`. Plan adherence stays "not measured" until the E11 judge.
- **E11 domain judges and judge validation (done offline):** `DomainJudge` and `JudgeClient` protocols (no
  provider assumed), `JudgeScorer` (outcome labels to scores; an unnamed label is an error), and the agreement
  harness (`validate_judge`, `agreement_report`: agreement, per-outcome counts, a confusion table, every
  disagreement listed, a judge that raises counted as a disagreement). `plan_adherence` and
  `multi_step_coherence` are judge-backed scorers, **experimental**: the judge sees the announced plan, tool names,
  argument names and failure status, never values, results or the question unless asked; a scorecard that uses one
  says how often the judge agreed with labelled trajectories, or that this has not been measured.
  **Not done:** no plan-adherence agreement has been measured, because no labelled trajectories for it exist (the
  campaign's 83 golden labels grade its answer judge, not plan adherence); a live judge run would only be a smoke
  check. **Campaign wrapper not added:** the campaign consumes the LCP through a pin that does not include
  `agent-evals` yet. Its wrapper is a callable `(case, output) -> JudgeVerdict` that builds the `GenericTrajectory`,
  calls `project_campaign_verification` and maps the result with its existing outcome codes; it needs no import of
  `agent_evals` (the protocols are structural), so it can be added with the next pin bump.
- **E12–E13:** see `evaluation-roadmap.md`.

## 7. Out of scope for now

Rewriting `penguiflow/evals/` (a later milestone can adapt it as one implementation of this protocol);
tool replay and sandboxed execution profiles; executor attestation; MLflow-managed datasets; publishing to
PyPI. The package name is a working name: `agent-eval` and `agent-evaluation` are taken on PyPI.
