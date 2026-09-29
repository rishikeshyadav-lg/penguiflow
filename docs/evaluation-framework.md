# The agent-agnostic evaluation framework (`agent-evals`)

Status: **E0 done** (this contract, the package, the neutral core moved out of the learning control
plane). E1–E9 are planned and marked as such. Written 2026-09-29.

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
`PredictionResult` (planned, E1):

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

- **E1 generalise the core:** variants with arbitrary `config`; a request over 1..N variants; several scorers
  per run; `PredictionResult`; paired views for any two variants.
- **E2 run mechanics:** `repeats`, `concurrency`, `timeout_s`, retries; a JSONL row per (case, variant,
  repeat) whose keys are a superset of the campaign's `runs.jsonl`; resume that skips finished rows; results
  independent of concurrency; every expected row has a result or an explicit failure. Silent dropping is a
  failed evaluation.
- **E3 statistics:** the case-clustered paired bootstrap (the prompt is the resampling unit because repeats of
  one prompt are not independent evidence), run-to-run noise, the minimum-detectable-effect multiplier
  (1.96 + 0.8416), prompts needed to clear a bar, threshold proposal, verdict wording.
- **E4–E9:** datasets and manifests, scorers, the domain-judge seam and golden-set validation, reports and
  an optional MLflow backend, an agent-agnostic runner service, and a three-agent proof.

## 7. Out of scope for now

Rewriting `penguiflow/evals/` (a later milestone can adapt it as one implementation of this protocol);
tool replay and sandboxed execution profiles; executor attestation; MLflow-managed datasets; publishing to
PyPI. The package name is a working name: `agent-eval` and `agent-evaluation` are taken on PyPI.
