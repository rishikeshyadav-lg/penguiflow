# LangChain learning-control-plane demo

Runs the same mine -> evaluate -> gate -> deliver loop as
`examples/learning_control_plane_local_demo`, but against a LangChain agent instead of a
PenguiFlow one, through `learning_control_plane.integrations.langchain.LangChainFrameworkAdapter`.
No MLflow account, model API key, or PenguiFlow serving process is needed: `agent.py`'s toy agent
is LLM-free, and the control plane uses local SQLite storage and deterministic mock evaluation.

```bash
uv run python -m examples.langchain_demo.flow --db-directory /private/tmp/penguiflow-lcp-langchain-demo
```

The directory must not already contain `control-plane.db`. The demo runs a toy campaign-analytics
agent (`agent.py`, one real `langchain_core` tool) several times, projects each run into the shared
investigation contract, mines a pattern from the repeated hedging in its answers ("might have
received around..." instead of a plain count), evaluates the mined advisory skill against held-out
queries, gates it, records human approval, and delivers it through the LangChain adapter's own
skill store -- proving the same control plane instance serves both PenguiFlow and LangChain agents
without any control-plane code change.
