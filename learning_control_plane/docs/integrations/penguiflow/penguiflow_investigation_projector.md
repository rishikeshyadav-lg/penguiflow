# PenguiFlow investigation projector

`PenguiFlowInvestigationProjector` turns a completed PenguiFlow `Trajectory`
into an `InvestigationTrajectoryV1`. It is deliberately a one-way projection,
not a trajectory serializer.

The integration supplies `PenguiFlowInvestigationContext` when it starts a run.
That context carries the trusted source-trace locator, identity, scope,
deployment fingerprint, start time, and the exact set of safe node names.

## What the document keeps

- source trace reference and run identity supplied by the integration;
- terminal status and a known terminal-reason label;
- whether a text request and binary input parts were present;
- per-step index, allowlisted node name, success/failure state, and presence
  flags for observations and streams; and
- aggregate step counts and whether a final answer existed.

## What it never reads into the document

The projector excludes query text, LLM context, tool context, action arguments,
thoughts, observations, stream chunks, errors, failure payloads, artifacts,
sources, metadata, summaries, resume input, steering input, and final answers.

Unknown node names become `redacted_node`. This is intentional: a string that
looks like a tool name can still be customer content. A framework integration
must explicitly allowlist its configured node names to preserve them.

## A thin adapter over the generic projector

The projection rules are framework-neutral and live in
`integrations/generic.py` (`project_run`). The PenguiFlow projector:

1. expands each parallel step into one step per branch (`expand_parallel_steps`),
   so the redaction allowlist, the verifier and the step signature see the real
   tool calls;
2. turns the trajectory into an `AgentRun` (`agent_run_from_trajectory`);
3. calls `project_run` with the context's `verification_projector` (which still
   receives the expanded PenguiFlow `Trajectory`) and `signature_normalizer`.

The projected document is byte-identical to the one the projector produced
before this split; a locked-digest test keeps it that way. `SignatureNormalizer`
now lives in the judge kit (`judging.SignatureNormalizer`, with `SignatureRules`
as a ready-made one) and is re-exported here. `VerificationProjector` here is the
trajectory-typed protocol (`TrajectoryVerificationProjector`); the
framework-neutral one in the judge kit takes an `AgentRun`.

## Publishing

`PenguiFlowInvestigationPublicationHook` runs this projection before it calls an
`InvestigationPublisher`, from a daemon thread. Publication failures are logged
and cannot delay or fail an agent request. The existing
`PenguiFlowTracePublicationHook` remains available for metadata-only evidence.

A planner reports a completed trajectory before its turn has the final answer
the user receives. To judge that answer, hold the trajectory in a `TurnStash`
(`hold(trajectory)`), and at the end of the turn call `release(trace_id,
final_answer)`. It judges and publishes once through a `RunPublisher`, and it
never raises into the turn. `PlannerTraceReadiness` lets the assessment
publisher wait for the planner's own trace persistence before it polls MLflow.
