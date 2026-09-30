# agent_evals on a real LangChain agent (live)

`flow.py` evaluates a `langchain.agents.create_agent` agent with two tools on a Databricks model endpoint, using the
LCP's LangChain adapter to turn each native run into the framework-neutral trajectory that the scorers read.

```bash
DATABRICKS_CONFIG_PROFILE=<profile> uv run python examples/agent_evals_live_langchain/flow.py
# another endpoint:  AGENT_EVALS_ENDPOINT=<name> ...
```

It makes about six short agent runs (a few thousand model tokens on the default nano-tier endpoint) and prints the
scorecard. The runner reports token counts, not dollars, so cost per task shows as "not measured".

The same file works offline: `main(model=...)` takes any LangChain chat model that supports tool calling. The test
`tests/agent_evals/test_live_langchain_example.py` runs it with a scripted model, so it needs no credentials.
