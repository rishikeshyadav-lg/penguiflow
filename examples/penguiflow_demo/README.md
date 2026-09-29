# PenguiFlow live learning-control-plane demo

The PenguiFlow counterpart of `examples/langchain_demo`: the same mine -> evaluate -> gate ->
deliver loop, the same toy campaign-clicks domain, but driving a real
`penguiflow.planner.ReactPlanner` agent against a live Databricks-hosted model through
`PenguiFlowFrameworkAdapter` (`learning_control_plane.integrations.penguiflow.projector`) --
proving that adapter against a real model for the first time; until now it had only ever seen
hand-built `Trajectory` objects in tests and the offline demo.

```bash
DATABRICKS_CONFIG_PROFILE=penguiflow-oauth \
  uv run python -m examples.penguiflow_demo.live --db-directory /private/tmp/penguiflow-lcp-penguiflow-live
```

The directory must not already contain `control-plane.db` or `skills.db`. This makes real, billed
model calls (roughly 26 turns on `databricks-claude-haiku-4-5`, well under $1 in total).

`DatabricksProvider` does not read `~/.databrickscfg` profiles itself, so auth is built explicitly
from `databricks.sdk.core.Config()`, reading the `DATABRICKS_CONFIG_PROFILE` environment variable
rather than passing `profile=` directly -- the latter fails whenever a `~/.databrickscfg` has two
profile names sharing one host, since the SDK's underlying `databricks auth token --host ...` call
can't disambiguate by host alone. See `live.py`'s module docstring for the full explanation.

The demo runs a toy campaign-analytics agent (`agent.py`, one PenguiFlow tool node,
`lookup_campaign_clicks`) several times, projects each run into the shared investigation contract,
mines a pattern from the model's natural hedging ("Campaign 112774 received 400 clicks." instead of
a plain count), evaluates the mined advisory skill against held-out queries by injecting it through
`attach_guidance`'s `LLMContextHook` (PenguiFlow's own native injection mechanism -- distinct from
LangChain's message-list `apply()`), gates it, records human approval, and delivers it -- printing a
real before/after answer pair from the live model.
