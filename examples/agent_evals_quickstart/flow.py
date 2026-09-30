"""Evaluate a plain Python function as an agent, with a few dozen lines of glue.

    uv run python examples/agent_evals_quickstart/flow.py

The "agent" is an ordinary function that calls two tools. To evaluate it you supply three things: a runner
that turns one case into a `PredictionResult`, a dataset, and scorers. `agent_evals` does the rest: repeats,
statistics, the eight-metric scorecard, and a report.
"""

from __future__ import annotations

import asyncio

from agent_evals import (
    ArgumentCorrectness,
    DatasetManifest,
    EvaluationCase,
    EvaluationDataset,
    EvaluationVariant,
    ExactMatch,
    GenericStep,
    GenericTrajectory,
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

POPULATION = {"paris": 2_100_000, "lyon": 520_000, "nice": 340_000}


# ---- the agent under test: nothing here knows about agent_evals ---------------------------------------
def my_agent(question: str) -> tuple[str, list[tuple[str, dict]]]:
    """Answer 'population of <city>' by calling a lookup tool, then a formatting tool."""

    city = question.rsplit(" ", 1)[-1].lower()
    calls: list[tuple[str, dict]] = [("lookup_population", {"city": city})]
    people = POPULATION.get(city)
    calls.append(("format_number", {"value": people or 0}))
    return (f"{people:,}" if people else "unknown"), calls


# ---- the glue: a runner, a dataset, and scorers -----------------------------------------------------
def run_one(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
    answer, calls = my_agent(case.inputs["question"])
    steps = [GenericStep(tool, args) for tool, args in calls]
    return PredictionResult(answer=answer, trajectory=GenericTrajectory(case.inputs["question"], steps, answer))


def answered(case: EvaluationCase, output: PredictionResult) -> dict[str, float]:
    """Success: the answer matches the case's expected answer."""

    return ExactMatch(name="success")(EvaluationCase(case.case_id, {}, case.expected["answer"]), output)


CASES = [
    EvaluationCase(
        f"pop-{city}",
        {"question": f"population of {city}"},
        expected={"answer": f"{people:,}", "tools": ["lookup_population", "format_number"]},
    )
    for city, people in POPULATION.items()
]
SCORERS = [
    PolicyVeto(answered, PolicyCheck(forbidden_tools=["delete_record"]), "success"),
    ToolSelection(),
    ArgumentCorrectness({"lookup_population": ToolArguments(required=("city",), types={"city": str})}),
]


async def main() -> str:
    dataset = EvaluationDataset("populations", "v1", CASES)
    manifest = DatasetManifest.from_dataset(dataset, suite="regression")
    run = await run_suite(
        dataset, manifest, [EvaluationVariant("my_agent")], run_one, SCORERS,
        metric_id="populations", metric_version="1", settings=RunSettings(repeats=3),
    )  # fmt: skip
    record = RunRecord.from_run(run, manifest, "my_agent", run_id="quickstart")
    return report_markdown(build_report(run, manifest, "my_agent", record=record))


if __name__ == "__main__":
    print(asyncio.run(main()))
