# Learning Control Plane

The Learning Control Plane (LCP) turns evidence from production agent usage into
small, governed runtime assets. Its first asset type is an **advisory skill**: text
the agent may use or ignore. It never changes agent code, permissions, tool policy,
or the live request path.

The LCP is intentionally framework-neutral. PenguiFlow is the first planned
provider, not a dependency of the control-plane core.

## MVP boundary

The MVP is an offline workflow:

1. Read immutable trace and outcome evidence.
2. Create a candidate advisory skill.
3. Run baseline and candidate versions against the same held-out cases.
4. Apply deterministic improvement and regression gates.
5. Send a passing candidate to a human reviewer.
6. Authorize scoped delivery and record an activation receipt.

If the LCP, its scheduler, or its evidence store is unavailable, production agents
continue to serve requests with their last valid configuration.

## Package map

| Path | Responsibility |
|---|---|
| `contracts/evidence.py` | Evidence records and optional telemetry sinks. |
| `contracts/investigation.py` | Portable investigation document, canonical JSON bytes, digest, and discovery index. |
| `control_plane/control_plane.py` | Offline candidate registry, job lifecycle, and deterministic gate. |
| `control_plane/persistence.py`, `control_plane/worker.py` | SQLite control-plane state and offline evaluation workers. |
| `evaluation/evaluation.py` | Standalone baseline-versus-advisory-skill local evaluator. |
| `evaluation/verification.py` | Safe verification and final-answer scoring. |
| `mining/investigation_mining.py` | Read-only verified MLflow attachment reader for safe candidate mining. |
| `mining/mining.py`, `mining/skill_drafting.py` | Safe pattern mining and the validated advisory-skill drafter. |
| `providers/investigation_publisher.py` | Idempotent MLflow trace-attachment publisher for investigation documents. |
| `providers/assessment_publisher.py` | MLflow assessment publisher, trace readiness, and the pending-assessment queue. |
| `judging/` | The judge kit: outcomes, answer reading, expectations, scope, no-data answers, the outcome ladder, the meaning check, frozen classification, the golden-set runner, and the gate's re-judging metric. |
| `integrations/generic.py` | Framework-neutral run projection and after-the-turn publishing (`RunContext`, `project_run`, `RunPublisher`). |
| `integrations/penguiflow/projector.py` | PenguiFlow's redaction-first investigation projector (an adapter over `project_run`) and advisory-skill adapter. |
| `integrations/penguiflow/publishing.py` | `TurnStash` and `PlannerTraceReadiness`: publish when the turn ends, after the trace is persisted. |
| `providers/verdict_revision.py` | Supersede a published verdict with a corrected one; nothing is deleted. |
| `providers/local_store.py` | A write-once local directory store for investigations, for integrations without MLflow. |
| `providers/typesafe_meaning.py` | TypeSafe (Jev) as a meaning judge behind its governance gate (`lcp-typesafe` extra). |
| `docs/architecture.md` | Boundaries and evidence flow for the MVP. |
| `docs/control_plane/control_plane.md` | The MVP decision workflow and its safety boundary. |
| `docs/evaluation/` | Evaluator contract (`evaluation.md`), metric directions and gates (`scoring.md`), verification rubric (`verification.md`). |
| `docs/investigations/` | Trajectory contract, publisher, dual export, lineage, and mining (`investigation_*.md`). |
| `docs/integrations/penguiflow/` | PenguiFlow projection contract, adapter, and the Planner V2 held-out inputs. |
| `docs/providers/mlflow_lineage.md` | MLflow tag, metric, and artifact-path convention. |
| `docs/mining_skill_drafting.md` | Drafting input, output, validation, and local demo contract. |
| `docs/judging/` | Integrator guide, the judge mistakes behind each kit piece, and the golden-set protocol. |
| `FULL_LOOP_RUN.md` | Start-to-finish local MVP setup, execution, and inspection guide. |

Start with [the full local run](FULL_LOOP_RUN.md) before connecting a real agent.

## Directory layout

- `contracts/`: portable evidence and investigation data contracts.
- `control_plane/`: candidate lifecycle, SQLite persistence, and offline workers.
- `evaluation/`: baseline-versus-candidate execution and verification rubrics.
- `mining/`: MLflow investigation reading, safe pattern mining, and skill drafting.
- `judging/`: the framework-neutral judge kit.
- `providers/`: MLflow attachment and assessment publishers, a local store, verdict revision, and the TypeSafe meaning judge.
- `integrations/`: the generic run projector and publisher, and the optional PenguiFlow adapter.
- `docs/`: design notes grouped by control-plane area.

Import from these folders or from the package root (`from learning_control_plane import ...`).
