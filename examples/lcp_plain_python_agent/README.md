# Plug a plain-Python agent into the learning control plane

This example shows that any agent, built with no framework at all, can go through the whole
learning control plane (LCP) loop. The agent answers stock questions about a small in-memory
inventory with three tools (`list_stores`, `query_stock`, `list_skus`). A fixed script decides its
tool calls, so the run is deterministic and needs no model, no MLflow and no network.

```bash
uv run python examples/lcp_plain_python_agent/flow.py
# or keep the published documents:
uv run python examples/lcp_plain_python_agent/flow.py --store-directory /tmp/lcp-inventory
```

## What it does

1. **Runs the agent** on six past questions and writes each run down as an `AgentRun`. The runs
   include a tool error the agent recovered from, an answer cut off mid-sentence, a question that
   matches two stores (the agent asks which one), and a SKU list shortened by a
   `"... [7 more items]"` note.
2. **Judges each run** with `OutcomeLadder` and a small `DomainJudge` for the inventory. The judge
   compares the stated figure with its own reference (read from the inventory, not from the
   agent's calls). The outcomes are `verified` for the correct answers,
   `agent_error` for the truncated one, and `handled_correctly` for the clarification.
3. **Publishes** each redacted investigation to a local directory with `RunPublisher` and
   `LocalInvestigationStore`.
4. **Mines** the repeated verified pattern (`units:query_stock`; the recovered probe is left out of
   the step signature) and drafts a candidate skill from a template.
5. **Checks the judge** against a synthetic golden set before any evaluation.
6. **Gates** the baseline against the candidate on held-out questions. `VerificationMetric`
   re-judges both arms, and `verification_policy` requires more verified runs and no more hard
   failures. The clarification is handled correctly on both arms, so it is left out of the
   comparison (`excluded_case_ids`).
7. **Ends in the review queue**: the candidate waits for a human decision.

The printed summary shows each stage's result. The candidate's `verified_success` rate rises from
0.33 to 1.0 over the judged held-out pairs.

## What to change for a real agent

- Build the `AgentRun` from your framework's record of the run (the PenguiFlow adapter does this in
  `agent_run_from_trajectory`).
- Replace `InventoryJudge` with your domain's judge. `MetricVocabulary`, `judge_expectation`,
  `question_scope_check` and `no_data_judgment` do the generic work.
- Replace `inventory_reference` with an independent query of your own data.
- Add a `MeaningCheck` (for example `TypeSafeMeaningJudge.from_environment()`) for mistakes value
  matching cannot see.
- Publish to MLflow with `MlflowAttachmentPublisher` and `MlflowAssessmentPublisher`, and use a
  real golden set of labelled runs.

See `learning_control_plane/docs/judging/` for the integrator guide.
