"""Evaluate a real LangChain agent, live, on a Databricks model endpoint.

    DATABRICKS_CONFIG_PROFILE=<profile> uv run python examples/agent_evals_live_langchain/flow.py

The agent is `langchain.agents.create_agent` with two tools and a small model (`databricks-gpt-5-4-nano` by
default; set `AGENT_EVALS_ENDPOINT` to use another). The LCP's own LangChain adapter turns each native run into
a `GenericTrajectory`, and the same scorers used for any other agent score it. A run makes a handful of short
model calls: a few thousand tokens in all. The runner reports token counts, not dollars, because this example
does not know the endpoint's price, so the report lists cost per task as not measured.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from agent_evals import (
    ArgumentCorrectness,
    Contains,
    DatasetManifest,
    EvaluationCase,
    EvaluationDataset,
    EvaluationVariant,
    PolicyCheck,
    PolicyVeto,
    PredictionResult,
    RunRecord,
    RunSettings,
    ToolArguments,
    ToolSelection,
    build_report,
    report_markdown,
    run_suite,
)

POPULATION = {"paris": 2_100_000, "lyon": 520_000, "nice": 340_000, "lille": 230_000, "nantes": 320_000}
INSTRUCTIONS = (
    "Answer with the population of the city. Call lookup_population, then format_number on its result, "
    "and reply with only the formatted number. If the city is not known, reply with the word unknown."
)
TOOLS = ("lookup_population", "format_number")


def build_agent(model: Any) -> Any:
    """A LangChain agent over two tools."""

    from langchain.agents import create_agent
    from langchain_core.tools import tool

    @tool
    def lookup_population(city: str) -> int:
        """Look up the population of a city; -1 when the city is not known."""

        return POPULATION.get(city.strip().lower(), -1)

    @tool
    def format_number(value: int) -> str:
        """Format a whole number with thousands separators."""

        return f"{value:,}"

    return create_agent(model, [lookup_population, format_number], system_prompt=INSTRUCTIONS)


def runner_for(agent: Any):
    """A runner that invokes the agent and converts the native run with the LCP's LangChain adapter."""

    from learning_control_plane.integrations.langchain.adapter import (
        LangChainFrameworkAdapter,
        LangChainInvestigationContext,
    )

    adapter = LangChainFrameworkAdapter(
        LangChainInvestigationContext(
            agent_ref="live-langchain", scope_ref="tenant:example", execution_fingerprint="sha256:example"
        )
    )

    async def run_one(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
        started = time.perf_counter()
        native = await asyncio.to_thread(agent.invoke, {"messages": [("user", case.inputs["question"])]})
        trajectory = adapter.to_generic_trajectory(native)
        usage = {"input_tokens": 0, "output_tokens": 0}
        for message in native["messages"]:
            for key, value in (getattr(message, "usage_metadata", None) or {}).items():
                if key in usage:
                    usage[key] += value
        return PredictionResult(
            answer=trajectory.final_answer,
            trajectory=trajectory,
            latency_ms=(time.perf_counter() - started) * 1000,
            llm_usage=usage,
        )

    return run_one


def cases() -> list[EvaluationCase]:
    known = [
        EvaluationCase(
            f"pop-{city}",
            {"question": f"What is the population of {city.title()}?"},
            expected={"answer": f"{people:,}", "tools": list(TOOLS)},
        )
        for city, people in POPULATION.items()
    ]
    unknown = EvaluationCase(
        "pop-atlantis",
        {"question": "What is the population of Atlantis?"},
        expected={"answer": "unknown", "tools": [TOOLS[0]]},
    )
    return [*known, unknown]


def scorers() -> list[Any]:
    def answered(case: EvaluationCase, output: PredictionResult) -> dict[str, float]:
        return Contains(name="success")(EvaluationCase(case.case_id, {}, case.expected["answer"]), output)

    return [
        PolicyVeto(answered, PolicyCheck(allowed_tools=list(TOOLS)), "success"),
        ToolSelection(),
        ArgumentCorrectness({"lookup_population": ToolArguments(required=("city",), types={"city": str})}),
    ]


async def main(model: Any | None = None) -> tuple[str, Any]:
    """Run the evaluation and return the Markdown report and the run (for tokens and rows)."""

    if model is None:
        from databricks_langchain import ChatDatabricks

        model = ChatDatabricks(
            endpoint=os.environ.get("AGENT_EVALS_ENDPOINT", "databricks-gpt-5-4-nano"), temperature=0
        )
    dataset = EvaluationDataset("populations-live", "v1", cases())
    manifest = DatasetManifest.from_dataset(dataset, suite="capability")
    run = await run_suite(
        dataset, manifest, [EvaluationVariant("langchain")], runner_for(build_agent(model)), scorers(),
        metric_id="populations", metric_version="1", settings=RunSettings(repeats=1, concurrency=3, timeout_s=90),
    )  # fmt: skip
    record = RunRecord.from_run(run, manifest, "langchain", run_id="live-langchain", bundle={"framework": "langchain"})
    return report_markdown(build_report(run, manifest, "langchain", record=record)), run


if __name__ == "__main__":
    report, finished = asyncio.run(main())
    print(report)
    tokens = sum(
        row.result.llm_usage.get("input_tokens", 0) + row.result.llm_usage.get("output_tokens", 0)
        for row in finished.rows
    )
    print(f"model tokens used: {tokens}")
