# Evaluation roadmap: `agent-evals` and the four-layer model

Status: E0–E10 done and E11 done offline, E12 done and deployed (see `evaluation-framework.md`); E13 planned below. The E4–E10 sections below are kept as written; deviations are listed in `evaluation-framework.md` §6. Written 2026-09-29 from the article
"AI Agent Evaluation: Outcome, Trajectory, and Operational Metrics in Production Systems" (Level Up Coding, Aug 2026).

## Context
The article is a Medium piece (Level Up Coding, Aug 2026) arguing that agent evaluation
needs four layers, not one success number: **outcome**, **trajectory**, **operational**, and a
**policy-compliance** constraint that cuts across all three. It closes with an 8-metric production
dashboard: Task Success Rate, Tool Selection Accuracy, Argument Correctness, Plan Adherence, Execution
Efficiency, cost per task, p95 latency, policy-violation flag. It also asks for pass@k / pass^k, two suites
(regression near 100%, capability with partial credit), scoring invariants rather than exact paths, golden
trajectories, shadow traffic that diffs the trajectory (not only the answer), and the loop "sample production
traces, freeze failures, fix, keep as a regression guard".

**Where we are (2026-09-29).** The package `agent-evals` (in `~/Desktop/penguiflow-plug-and-play`, branch
`feat/lcp-framework-agnostic`, all commits local, none pushed) has E0–E3 done: neutral core, N-variant
comparison, repeated resumable runs, public statistics and threshold calibration. Against the article it
covers about 2 of the 8 dashboard metrics (success rate, cost per task) plus the production-trace loop (the
LCP). Latency is captured but only as means. Everything about the trajectory, efficiency, policy and
golden/shadow comparison is missing.

**Intended outcome.** A framework where any agent, by supplying a runner, a dataset and optionally a few
checks, gets the article's full scorecard, with statistics that account for the agent's own noise, and
where each metric is verified to give the same answer for PenguiFlow, LangChain and a plain-Python agent.

**What we deliberately do not copy.** The article's fixed thresholds (85% CI gate, p95 under 4 s, block at
+15% cost) are, by its own words, the author's starting proposals. Ours are derived from a calibration of the
agent being judged; the article's numbers ship only as a labelled example profile. Runtime enforcement of
destructive actions ("a line in the prompt is not a safeguard") belongs in the agent's execution path; we
detect and report violations, and document that.

## How the earlier plan maps
Earlier E4–E9 are folded in, not dropped:

| Earlier | Now |
|---|---|
| E4 datasets | E4 (adds the regression / capability suite tag) |
| E5 scorers | split into E5 outcome, E6 trajectory, E7 operational, E8 policy |
| E6 domain-judge seam | E11 |
| E7 reports, MLflow | E10 (adds the 8-metric scorecard) |
| E8 runner service | E12 |
| E9 three-agent proof | E13 (re-scored against the article) |
| new | E9 golden trajectories, trajectory diff and shadow comparison |

## Coverage matrix (article item → milestone)
| Article item | Today | Milestone |
|---|---|---|
| Task success from real state, not the agent's claim | scorer-defined | E5 |
| pass@k, pass^k, consistency | not computed (`_consistency` in campaign) | E5 |
| Partial credit; regression vs capability suites | rubric in LCP; suites not modelled | E4, E5 |
| Tool Selection Accuracy | absent | E6 |
| Argument Correctness (syntactic + semantic) | campaign-only | E6 |
| Plan Adherence, Multi-step Coherence | absent | E11 (LLM judge; article leaves it open) |
| Invariants: required / forbidden / tracked events | mechanism only | E6 |
| Success-vs-tool-selection gap as an audit signal | absent | E6, E10 |
| Execution Efficiency, expected step band | counts only | E7 |
| Loop guard (hash last K actions) | absent | E7 (offline detector), runtime stop stays in the agent |
| p95 latency, cost per task | means only | E7 |
| Policy-violation flag | slot in LCP only | E8 |
| Golden trajectories | none | E9 |
| Shadow traffic with trajectory diff | none | E9 |
| Production-trace loop | the LCP | done; E13 re-proves it |
| 8-metric dashboard | 2 of 8 | E10 |

## Reuse (found by exploration, so we lift rather than rewrite)
Paths: **AE** = `packages/agent-evals/agent_evals/`, **LCP** = `packages/learning-control-plane/learning_control_plane/`,
**CPA** = `~/Desktop/acp-merge-dev/src/campaign_performance_agent/`.
- `GenericStep`/`GenericTrajectory` (AE/steps.py), `PredictionResult`, `ScoreResult` (its `checks` mapping fits
  invariants), `normalize_scores` (AE/prediction.py), `RepeatedRun.case_means` (AE/execution.py:184).
- `_percentile` (AE/statistics.py:218; private, make public for p95).
- `VerificationCheck`, `SafeStepEvidence`, `InvestigationVerification.policy_compliance` (LCP/evaluation/verification.py:57, :91, :323).
- `step_signature` (LCP/integrations/penguiflow/projector.py:177) for sequence fingerprints; `project_trajectory`
  (:331) for step counts.
- `PromotionPolicy` target/protected groups (LCP/control_plane/control_plane.py:84–111): already a regression guard.
- CPA `_execution_policy_check` (learning_control_plane_evaluation_scoring.py:2965: allowed / required / maximum
  calls by tool), `_tool_error_rate` (:3626), `_argument_mismatch_codes` (:2772, pattern only),
  `_consistency` (learning_control_plane_baseline_calibration.py:640), `_recovered_step_indexes`
  (learning_control_plane_verification.py:286).
- Design notes: `docs/proposals/TRACE_DATASET_EVAL_OPTIMIZATION_MVP_ARCHITECTURE.md:481–513` names
  `allowed_tools_only`, `workflow_order_pass`, `budget_pass`, `terminal_pass`;
  `RFC_TRACE_DERIVED_DATASETS_AND_EVALS.md:773` sketches tool-call fingerprinting for replay.

Known gap: no adapter gives per-step latency or tokens (only the campaign's own tool-call type has latency).

## Working rules (all milestones)
1. `agent_evals` stays dependency-free and never imports an agent framework; import-purity test stays green.
2. Every existing test stays green with zero edits; each new scorer has a negative-path test.
3. Logic lifted from the campaign is copied generically with **bit-compare against the old code first** (record
   the old output before moving, as done in E3); the campaign adopts the package later (a pin bump, not here).
4. Additive only: new optional fields, new modules; no rename of a public name.
5. No push, publish, deploy or live model spend without the owner's go; state cost before any paid step.
6. Verification counts by parsing progress lines or summing `--collect-only -qq` per file, never counting dots.
7. First step of the build: write this plan as `docs/evaluation-roadmap.md` and link it from
   `docs/evaluation-framework.md` §6 (which currently lists E4–E9 in the old shape).

---

## Milestones

### E4 — Datasets and suites
- **Goal.** Datasets any agent can load, freeze and keep disjoint, each declared a **regression** or
  **capability** suite, without customer text in the repo.
- **How.** Loader/saver for JSONL, CSV, JSON; manifest file (ids, digest, `suite`, optional metric id/version,
  text-free mode); seeded per-group disjoint split builder generalised from campaign's `build_question_banks.py`;
  dataset-to-metric coupling check from `RFC_TRACE_DERIVED` (run fails fast on a metric-version mismatch);
  suite semantics: regression suites carry a hard pass threshold and fail on any case below it, capability
  suites report partial-credit means and intervals. Reuse `EvaluationDataset.manifest_digest`.
- **Complete looks like.** A dataset round-trips byte-identically; a metric-version mismatch fails before any
  run; a split proves its halves disjoint; a run declares its suite type and the report says which rule applied.
- **Verify.** Round-trip and negative tests (mismatch, overlap, duplicate id, missing suite); same seed and
  categories reproduce `banks_v1.1`'s id sets (offline manual check against the local manifest); the two suite
  types give different verdicts on one synthetic result set.

### E5 — Outcome layer
- **Goal.** Outcome scoring that measures the real result and its repeatability, not one number.
- **How.** Built-in scorers: exact, contains, regex, numeric tolerance (mechanism from campaign
  `decision_tolerance`); a `StateCheck` protocol so success is graded on the environment's state (a database
  row, a file) and not the agent's text; a weighted partial-credit aggregator (lift `score_final_answer`'s
  mechanism, defaults stay in the LCP as data); `pass_at_k`, `pass_hat_k` (pass^k), and the per-case
  consistency label (`consistently_correct` / `intermittent` / `consistently_incorrect`, lifted from
  `_consistency`) computed from a `RepeatedRun`, using the unbiased estimators C(c,k)/C(n,k) and
  1 − C(n−c,k)/C(n,k).
- **Complete looks like.** One scorer set works on any output; pass@k and pass^k come from any repeated run; a
  `StateCheck` example fails an agent that says "done" without the state change.
- **Verify.** The article's own example: 75% per-attempt success gives pass^3 = 0.421875 (about 42%); edge cases
  (k > n, n = c, c = 0, k = 1 equals the plain rate); consistency labels bit-compare with `_consistency` on the
  local `baseline-v1` data (offline, not committed); a `StateCheck` test where the text claims success and the
  state disagrees.

### E6 — Trajectory layer
- **Goal.** Score how the agent got there: tools chosen, arguments given, and invariants held, without
  demanding one exact path.
- **How.** New `trajectory` module over `GenericTrajectory`:
  - `tool_selection_accuracy` = matched calls ÷ max(expected, actual), the article's IoU-style formula;
  - `argument_correctness` at two levels, syntactic (schema/type) and semantic (value ranges), reporting
    mismatch **codes**, not raw values (pattern from `_argument_mismatch_codes`);
  - sequence modes `exact`, `in_order`, `any_order` against a reference (fingerprint via the `step_signature` idea);
  - `Invariant` = required / forbidden / tracked-non-gating events, with an optional `when` condition ("an
    authorization must precede any write"); lifted generically from `_execution_policy_check`
    (allowed / required / maximum calls by tool);
  - a `selection_gap` field: task success rate minus tool selection accuracy, reported as an **audit signal, not a
    verdict**, as the article insists.
- **Complete looks like.** The same scorer, unchanged, scores PenguiFlow, LangChain and mock trajectories; an
  invariant set can express "authorize before write" and "diagnostic steps in any order"; a run can be right
  on outcome and low on selection and the report says so.
- **Verify.** Formula tests (expected `[a,b]`, actual `[a,b,c]` → 2/3; actual `[a]` → 1/2; disjoint → 0); the three
  order modes on one reference; codes-only output leaks no argument values; bit-compare the lifted policy check
  against `_execution_policy_check` on recorded campaign trajectories (offline); cross-adapter test reusing the
  four native runs from `test_provider_conformance.py`.

### E7 — Operational layer
- **Goal.** Turn accuracy into a product decision: steps, cost, latency as first-class, with sensible bands.
- **How.**
  - Make `_percentile` public; add p50/p95/p99 for latency and cost (per variant, with bootstrap intervals).
  - Step metrics: `step_count`, `failed_step_count`, `execution_efficiency` = optimal ÷ actual steps where an
    optimum is known, otherwise position within an **expected band** for the task class (band derived from the
    baseline calibration, e.g. the observed 10th–90th percentile per category). A run under the band is flagged as
    an alarm like one over it, because fewer steps can mean a skipped verification.
  - Loop detector: hash the last K actions (tool + normalised args); flag a run when the same hash repeats 3 times.
    Ship it as an offline scorer and a small helper a runner can call to stop early; the hard stop itself stays
    in the agent.
  - Optional `PromotionThresholds` fields for a p95 bound and a cost alert tier (default off, so no gate changes).
  - Additive `latency_ms` on `GenericStep`, filled only by adapters whose native run has it (PenguiFlow: check; LangChain: no).
- **Complete looks like.** A report shows p95 latency and cost per task next to success; a three-step run in a
  seven-to-eleven-step class is flagged; a repeated-call loop is caught offline; no existing gate decision changes.
- **Verify.** Percentile and band tests on synthetic data; loop test (identical calls ×3 flagged, same tool with
  different args not); the gate fixture reproduces the E3 golden intervals unchanged (exact floats); `GenericStep`
  golden digests unchanged; the adapters still pass their conformance tests.

### E8 — Policy layer
- **Goal.** A cross-cutting compliance check that can veto a run whatever its other scores are.
- **How.** A `PolicyCheck` (built on `VerificationCheck`) with forbidden tools and forbidden side-effect classes via
  a small effect-registry interface (pattern from `_tools_with_unsafe_side_effects`), denial-status detection
  (`blocked` / `denied` / `policy_denied`), and **hard-fail semantics**: any violation marks the run failed, is
  excluded from "success", and raises the report's `policy_violation` flag. Propose (not apply) a matching LCP gate
  rule "any violation in a candidate's runs blocks approval", since it changes gate behaviour.
- **Complete looks like.** A run with a perfect answer and one forbidden call is reported as a violation, not a
  success; the metric is visible in every report; the docs state plainly that enforcement is the agent's job.
- **Verify.** Negative tests for each violation class; a correct-outcome run with a violation flips the flag; lifted
  logic bit-compares with `_legacy_policy_check` and `_execution_policy_check`; LCP suite unchanged (the gate rule
  is a written proposal until the owner approves it).

### E9 — Golden trajectories, trajectory diff, shadow comparison
- **Goal.** Compare a candidate by its decisions, on frozen and on production-shaped inputs, without side effects
  taking hold.
- **How.** `GoldenTrajectory`: frozen full run (inputs, tool observations, decision sequence, outcome, environment
  reference) with a digest; a re-approval record so a refresh is a deliberate act (approver, reason, old and new
  digest), because a careless refresh turns a bug into the new "correct". `TrajectoryDiff` between any two runs or
  variants: tool sequence, arguments, cost, latency, triggered guardrails and policy flags, not only answer
  agreement. A **shadow comparison** helper: run a candidate over recorded production inputs with the variant
  config marked `dry_run`, then diff against the recorded production trajectory. We build the diff and report; the
  agent must honour dry-run itself (tool replay and sandboxing stay out of scope, §7 of the framework doc).
- **Complete looks like.** A shadow report that a final-answer-only comparison would call "identical" shows the
  candidate took a different tool path or cost more; a golden refresh without an approval record is refused.
- **Verify.** Synthetic pair with the same answer and different tools produces a non-empty diff; digest changes on
  any path change; refresh-without-approval test; diff of two recorded PenguiFlow runs from the conformance set
  agrees with a hand-written expected diff.

### E10 — Reports and the eight-metric scorecard
- **Goal.** Every run ends in a report that says what was and was not measured, one metric per layer.
- **How.** `RunRecord` (settings, bundle, dataset and metric versions, digests, verdict-grade flag);
  `report.json` and `report.md` with the article's eight entries (success rate, tool selection, argument
  correctness, plan adherence when a judge exists, execution efficiency, cost per task, p95 latency, policy flag),
  per suite, with E3 intervals, the `selection_gap` audit line, and every failure and exclusion listed explicitly;
  two threshold profiles, `ci_gate` and `production_slo`, the article's numbers shipped only as a labelled
  **example** profile beside profiles derived by `propose_thresholds`; optional MLflow backend behind
  `agent-evals[mlflow]` per `DUAL_LOCAL_MLFLOW_EVALUATION_BACKENDS.md`.
- **Complete looks like.** Same report structure for three different agents; a metric that could not be computed
  is shown as "not measured" with the reason, never omitted; MLflow aggregates equal the local backend's.
- **Verify.** Snapshot tests with no customer text; an MLflow equivalence test on a local SQLite store; a report
  test where plan adherence is absent shows "not measured".

### E11 — Domain-judge seam and judge validation (includes the LLM trajectory judge)
- **Goal.** Any agent can plug in its own judge, prove it is trustworthy, and use one for plan adherence and
  coherence, the part the article itself calls unresolved.
- **How.** `DomainJudge` protocol and a `JudgeClient` protocol (no provider dependency); campaign wraps its judge
  without moving logic; a generic golden-set agreement harness (labels, judge, agreement table, disagreements)
  from `check_judge_golden_set.py` / `rejudge_stored_cases.agreement_table`; plan adherence and multi-step coherence
  as judge-backed scorers **marked experimental** until their agreement is measured on labelled trajectories.
- **Complete looks like.** The campaign golden set still agrees 74/83 through the generic harness; a toy judge is
  validated by it; the two judge-backed metrics carry their measured agreement in every report.
- **Verify.** Run the harness on local labels and compare 74/83; synthetic-label unit tests. Any live judge run is
  a paid step: cost stated first, owner's go required.

### E12 — Agent-agnostic runner service (the eval app)
- **Goal.** The eval app runs any registered agent and reports the scorecard, not only the campaign agent.
- **How.** Stage registry and per-agent argument schema replacing the hard-coded script path; an
  `AgentUnderTest` registration (runner factory, case-inputs builder, settings provider, judge); Databricks
  specifics behind a `WorkspaceStore` interface; `lcp-eval` keeps its behaviour through an adapter.
- **Complete looks like.** The app runs a smoke for the campaign agent and for a second agent by configuration only;
  campaign `smoke-2` still gives 2 of 2 verified.
- **Verify.** The 13 existing eval-app tests unchanged plus new ones; **redeploying needs the owner's go**, then
  `smoke-2` (about $0.83) compared with the 2026-09-29 result.

### E13 — Three-agent proof, article re-score, packaging readiness
- **Goal.** Demonstrate the claim and re-measure the gap to the article.
- **How.** One evaluation definition run on (1) the campaign agent, reusing verdict-grade evidence (no new spend),
  (2) the PenguiFlow toy agent, (3) the LangChain toy agent (live, under $1 total), plus a deliberately
  misbehaving mock (timeouts, malformed output, exceptions, a loop, a forbidden call). Onboarding test: a fourth
  agent, a plain Python function, evaluated with a few dozen lines of glue. Fix stale docs
  (`evaluation/scoring.md` says intervals are "the next milestone"; `architecture.md` overstates what MLflow stores
  and that both arms share one bundle). README quickstart. Publishing checklist; **no publishing without the owner's go.**
- **Complete looks like.** The coverage matrix above re-scored with every row "covered" or explicitly "out of scope,
  with reason"; the four reports share one structure; the misbehaving mock produces a violation flag, a loop flag
  and no dropped row.
- **Verify.** Run all four; diff report structures; import-purity check; docs-versus-code review.

---

## Ordering and first build pass
Dependencies: E4 → E5 → E6 → E7 → E8 (each adds a layer to the scorecard), E9 needs E6–E8, E10 needs E5–E8,
E11–E13 last. **Recommended first pass: E4–E7** (datasets, outcome, trajectory, operational). It is offline, spends
nothing, and takes the scorecard from 2 to about 6 of 8 metrics. Stop for review after E7, as after E3.
E8–E10 second pass; E11–E13 third (E11 and E12 involve paid or deployed steps needing the owner's go).

## What "complete" means for the whole plan
1. The scorecard has all eight metrics for a new agent supplying only a runner, a dataset and optional checks; a
   metric that cannot be measured is shown as not measured, with the reason.
2. Each metric gives the same answer for PenguiFlow, LangChain and plain-Python trajectories.
3. Lifted logic reproduces the campaign's old numbers exactly (bit-compare recorded before each move).
4. The article's pass^k example, tool-selection formula and loop rule are tests, not prose.
5. The three real agents plus the hostile mock produce comparable reports.
6. Docs state only what the code does, including what is deliberately not built (runtime enforcement, sandboxing).

## Open items for the owner (none blocks starting E4)
- The LCP gate rule "a policy violation blocks approval" (E8) changes gate behaviour; written as a proposal until approved.
- The deferred generic `explain_verdict` from E3 needs a decision on how metrics declare units and wording (fits E10).
- Whether to push the local commits, and when to bump the campaign's pin to include `agent-evals`.
- The calibration decision (0.092 vs 0.115) is separate and still open.


---

## Result: article coverage re-scored (2026-09-30, end of E13)

"Covered" means built, tested and, where a number was recorded from older code, reproducing it.

| Article item | Now | Note |
|---|---|---|
| Task success from real state | covered | scorers plus `StateCheck` |
| pass@k, pass^k, consistency | covered | functions in `repeatability`; **not one of the eight scorecard entries** |
| Partial credit; regression vs capability suites | covered | `WeightedRubric`; `suite_verdict` |
| Tool selection, argument correctness | covered | argument specs are per tool, supplied by the user |
| Plan adherence, coherence | partly | judge-backed and experimental; agreement with labels **not measured** (no labelled trajectories exist) |
| Invariants (required, forbidden, tracked) | covered | |
| Success-versus-selection audit signal | covered | on the scorecard when both are measured |
| Execution efficiency, step band | partly | scorers exist; efficiency shows only when a scorer produces it, and the band is not a scorecard entry |
| Loop guard | partly | offline detector and a runner helper; the hard stop stays in the agent, by design |
| p95 latency, cost per task | covered | cost needs the runner to report it |
| Policy-violation flag | covered | detects and reports; enforcement is out of scope, by design |
| Golden trajectories | covered | re-approval enforced |
| Shadow traffic with trajectory diff | covered as a comparison | `dry_run` must be honoured by the agent; not yet run on real production traffic |
| Production-trace loop | covered | the LCP, unchanged |
| Eight-metric dashboard | eight entries always; how many are *measured* depends on what an agent supplies | see below |

### What the proofs measured
| Run | Measured of 8 | Not measured, and why |
|---|---|---|
| Campaign agent, `baseline-final` (existing evidence, no new spend) | 3: success 0.839 (0.720 to 0.946), cost $0.576, p95 latency 126.6 s | no trajectory scorers or policy check were run on it |
| Live LangChain agent (nano endpoint, 6 cases, 4,254 tokens) | 5: success, tool selection, argument correctness, p95 (2.5 s), policy | cost (runner reports tokens, not dollars), efficiency (no scorer), plan adherence (no judge) |
| Plain Python function (quickstart, 32 lines of glue) | 5 | as above |
| Recorded PenguiFlow, LangChain (two shapes) and mock runs | identical scores on identical paths | |
| Hostile mock (timeout, malformed output, exception, loop, forbidden call) | no run dropped; each failure listed by kind; flag raised | |

One finding from the campaign run: its **p95 latency is 127 s against a 55 s mean**, with a range to 274 s. The mean
hides the tail that users feel.

### Deviations from the E13 plan
- The "PenguiFlow toy agent" is a recorded native run converted by the LCP's PenguiFlow adapter, not a live agent
  (no LiteLLM in this environment).
- The LangChain run used a Databricks nano endpoint through `databricks_langchain`.
- Stale docs fixed: `evaluation/scoring.md` (intervals were called "the next milestone") and `architecture.md`
  (MLflow's role, and the same-bundle guarantee, which is the host runner's job and is not checked).
- Nothing is published; see `agent-evals-publishing-checklist.md`.

### Decisions taken after the build (2026-09-30)
- The local commits were pushed to the fork and the campaign's pin was bumped to `e1be846`, so the campaign can now
  import `agent_evals`.
- Not publishing to PyPI or open-sourcing: the package stays in-house (`agent-evals-publishing-checklist.md` now
  describes in-house distribution and keeps the old checklist only in case that changes).
- The "policy violation blocks approval" gate rule is declined; the policy check still detects and reports.
- The workspace calibration (bar 0.092) replaced the laptop one (0.115) in the campaign gate; see the campaign
  decision record, P15.
