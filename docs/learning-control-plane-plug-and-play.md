# Plugging a new agent framework into the Learning Control Plane

The Learning Control Plane (LCP) mines an agent's traces for repeated failure or
weakness patterns, drafts an advisory skill, evaluates it against held-out cases,
gates promotion, records human review, and delivers it back to the live agent.
None of that -- the control plane, the evidence contracts, the evaluation math,
the mining and drafting -- knows anything about which agent framework produced
the traces. Only one seam does: turning a framework's own run shape into the
shapes the control plane and its verifier read. That seam is the
`FrameworkAdapter` Protocol.

This doc is for someone plugging in a framework that is not PenguiFlow or
LangChain (say, Google's ADK, or an in-house agent loop). It says exactly what
to implement, using the two existing adapters as templates.

## The contract

`learning_control_plane/integrations/protocol.py` defines it as a
`@runtime_checkable` `Protocol` -- four methods, all of which take your
framework's own native run object, whatever shape that is:

```python
class FrameworkAdapter(Protocol):
    def to_generic_trajectory(self, native_run: Any) -> GenericTrajectory: ...
    def project(self, native_run: Any, *, completed_at=None, investigation_id=None) -> InvestigationTrajectoryV1: ...
    def attach_guidance(self, *, guidance: str, categories: tuple[str, ...]) -> Any: ...
    def deliver(self, authorization: DeliveryAuthorization, candidate: AdvisorySkillCandidate, *, now=None) -> ActivationReceipt: ...
```

Adding a framework never requires a change to the control plane, the evidence
contracts, or the evaluation harness -- that is the non-negotiable requirement
in `docs/proposals/FRAMEWORK_AGNOSTIC_LEARNING_CONTROL_PLANE.md` §2, and the
conformance suite (below) is what proves it stays true.

### 1. `to_generic_trajectory` -- for the verifier

Translate one run into `GenericStep`/`GenericTrajectory`
(`learning_control_plane/contracts/steps.py`): the query, the tool calls (name,
args, observation, error), and the final answer, in a shape that owes nothing to
your framework's internals. A domain verifier is written once against this
shape and never needs to change when a new framework arrives --
`test_provider_conformance.py`'s `_toy_verifier` is a worked example of that
promise being kept.

### 2. `project` -- for mining

Produce an `InvestigationTrajectoryV1` (`contracts/investigation.py`), the
redacted, content-free document mining reads. This is usually a thin wrapper
around `to_generic_trajectory`, adding `SourceTraceRef`, `scope_ref`,
`execution_fingerprint`, and a `step_signature`.

### 3. `attach_guidance` -- for injecting a learned skill at runtime

Return *something* your framework's runtime can use to inject approved
guidance into a live turn. The protocol promises only that a caller gets an
object back -- it says nothing about how that object is invoked, because that
decision is always the framework's own. PenguiFlow's version
(`_GenericAdvisoryHook`) matches its `LLMContextHook` Protocol (async
`before_run`); LangChain's (`LangChainGuidanceInjector`) exposes a synchronous
`apply(messages, *, category=None)` meant to compose into a chain; the mock's
(`MockGuidanceHook`) is different again. Shape this to fit your framework's own
idiom, not the others'.

### 4. `deliver` -- for activation

Given an approved, unexpired `DeliveryAuthorization` and the
`AdvisorySkillCandidate` it authorizes, write the skill to wherever your
framework reads activated skills from, and return an `ActivationReceipt`.
Reject an expired or mismatched authorization by raising `ValueError` -- the
conformance suite checks this. The skill store itself can be anything; the
mock and LangChain adapters both use a five-line in-memory dict, proof that
`deliver` needs no particular backend.

## Starting point

Copy `learning_control_plane/integrations/mock/adapter.py`. It is the smallest
complete adapter in the repo, deliberately shaped nothing like PenguiFlow's
(different field names, no parallel-step concept, its own tiny skill store) --
it exists only to prove the protocol isn't secretly PenguiFlow-shaped, which
makes it the right template for a framework that also isn't PenguiFlow.

## Proving it

1. Add a case to `tests/learning_control_plane/test_provider_conformance.py`'s
   `CASES` dict (a factory returning `(adapter, native_run)`, same as
   `_mock_case` and `_langchain_case`). The suite's five test functions then run
   unchanged against your adapter: protocol conformance, the shared domain
   verifier, a valid investigation projection, category-matched guidance, and
   delivery accepting an active authorization while rejecting an expired one.
2. Optionally, build a small demo agent and run it through the real control
   plane end to end -- mine, evaluate, gate, review, authorize, deliver --
   following `examples/langchain_demo/flow.py` as a template. It mirrors
   `examples/learning_control_plane_local_demo/flow.py` line for line except
   for the adapter and the agent, which is the point: the control plane doesn't
   change; only the six lines that talk to your framework do.
