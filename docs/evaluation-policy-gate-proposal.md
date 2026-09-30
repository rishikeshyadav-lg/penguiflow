# Proposal: a policy violation blocks approval (declined)

Status: **declined by the owner on 2026-09-30, and not to be built.** The reason given: this is an internal tool, so
a hard automatic veto adds little; a person reviews every candidate anyway. The proposal is kept as the record of
what was considered. What stays: `PolicyCheck` and `PolicyVeto` still detect violations and stop a violating run
from counting as a success, and reports still show the policy flag. Nothing in the gate changed.
Written 2026-09-29 alongside E8 of `evaluation-roadmap.md`.

## What exists
`agent_evals.PolicyCheck` finds policy violations in a run and `PolicyVeto` stops a violating run from counting
as a success. So a candidate skill that makes the agent call a forbidden tool lowers its measured success rate.
That is a soft effect: a skill whose gain is large enough could still clear the accuracy bar with a few
violating runs among many good ones.

## The rule
A candidate is not approved if any of its evaluated runs raised the policy-violation flag
(`policy_flag(run, variant_id).raised`), whatever its accuracy, cost or latency results. The baseline's own
violations do not block a candidate, but they are reported, because a candidate that removes them is a benefit.

## Where it would go (sketch)
- `PromotionPolicy` (`packages/learning-control-plane/learning_control_plane/control_plane/control_plane.py`)
  gains one field, `block_on_policy_violation: bool = False`. The default keeps every existing decision as it is.
- The gate reads the candidate's `policy_compliance` metric from the paired results it already holds and, when
  the field is on and any complete pair has a candidate value below 1.0, adds the reason
  `policy_violation_in_candidate_runs` and rejects.
- `GateDecision` records the violating case ids (ids only, no text).

## Why it is not applied
- It is a new veto in the promotion path; the owner's rule so far is that accuracy decides and only huge cost or
  latency changes veto.
- The metric must exist in the evaluation results. Today only runs scored with `PolicyCheck` or `PolicyVeto`
  produce it, and the campaign does not yet.

## Verification if adopted
1. With the field off, the LCP suite and the gate fixture give byte-identical decisions (the E3 golden intervals).
2. With it on: a candidate with one violating run among 90 clean ones is rejected; the same candidate without the
   violation is approved; a baseline-only violation does not block.
3. A decision recorded under the new field gets a different policy version, so it never matches an older decision.
