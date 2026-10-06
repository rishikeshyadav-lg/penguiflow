"""Prove from a standalone install that this library depends on nothing and favours no agent framework.

    python -m agent_evals.selfcheck

The repository's own test suite proves the same things, but it needs the repository: its five-agent proof
reads cases from a sibling package and exercises PenguiFlow and LangChain adapters. Someone deciding whether
to adopt this library has only the installed package, so the two claims that matter to them are checked here,
with nothing but the standard library:

1. importing `agent_evals` loads no agent framework, no model client and no backend;
2. agents written in unrelated shapes produce the same scorecard from the same behaviour.

Exits non-zero if either fails, so it can run in a consumer's CI.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import textwrap
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from .datasets import DatasetManifest
from .evaluation import EvaluationCase, EvaluationDataset, EvaluationVariant
from .execution import RunSettings
from .outcome import ExactMatch
from .prediction import PredictionResult
from .report import RunRecord, build_report
from .steps import GenericStep, GenericTrajectory
from .suites import run_suite
from .trajectory import ToolSelection

# Importing the library must not drag in a framework, a model client or a backend.
FORBIDDEN_MODULES = (
    "langchain",
    "langchain_core",
    "learning_control_plane",
    "litellm",
    "mlflow",
    "openai",
    "penguiflow",
)

# Measured wall-clock time and spend differ between identical runs, so they are reported but not compared.
MEASURED_NOT_DERIVED = ("p95_latency", "cost_per_task")

POPULATION = {"paris": 2_100_000, "lyon": 520_000}
QUESTIONS = {city: f"population of {city}" for city in POPULATION}


def _answer_and_calls(question: str) -> tuple[str, list[tuple[str, dict[str, Any]]]]:
    """The one behaviour every agent shape below implements, however it is written."""

    city = question.rsplit(" ", 1)[-1].lower()
    people = POPULATION.get(city, 0)
    return f"{people:,}", [("lookup_population", {"city": city}), ("format_number", {"value": people})]


def _prediction(question: str) -> PredictionResult:
    answer, calls = _answer_and_calls(question)
    steps = [GenericStep(tool, args) for tool, args in calls]
    return PredictionResult(answer=answer, trajectory=GenericTrajectory(question, steps, answer))


def _plain_function(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
    """The shape the quickstart uses: an ordinary function."""

    return _prediction(case.inputs["question"])


async def _coroutine_function(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
    """An agent whose runner is async, as a served model client usually is."""

    await asyncio.sleep(0)
    return _prediction(case.inputs["question"])


class _CallableObject:
    """An agent held as object state, the shape a class-based framework produces."""

    def __init__(self, population: dict[str, int]) -> None:
        self._population = population

    def __call__(self, case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
        return _prediction(case.inputs["question"])


AGENT_SHAPES: dict[str, Callable[..., PredictionResult | Awaitable[PredictionResult]]] = {
    "plain function": _plain_function,
    "coroutine function": _coroutine_function,
    "callable object": _CallableObject(POPULATION),
}


def _success(case: EvaluationCase, output: PredictionResult) -> dict[str, float]:
    return ExactMatch(name="success")(EvaluationCase(case.case_id, {}, case.expected["answer"]), output)


def _cases() -> list[EvaluationCase]:
    return [
        EvaluationCase(
            f"pop-{city}",
            {"question": QUESTIONS[city]},
            expected={"answer": f"{people:,}", "tools": ["lookup_population", "format_number"]},
        )
        for city, people in POPULATION.items()
    ]


async def _scores_for(runner: Callable[..., Any]) -> dict[str, float | None]:
    """One agent shape's scorecard, reduced to the numbers a reader would compare."""

    dataset = EvaluationDataset("selfcheck", "v1", _cases())
    manifest = DatasetManifest.from_dataset(dataset, suite="regression")
    variant = EvaluationVariant("under test")
    run = await run_suite(
        dataset, manifest, [variant], runner, [_success, ToolSelection()],
        metric_id="selfcheck", metric_version="1", settings=RunSettings(repeats=2),
    )  # fmt: skip
    record = RunRecord.from_run(run, manifest, variant.variant_id, run_id="selfcheck")
    report = build_report(run, manifest, variant.variant_id, record=record)
    return {
        entry.key: entry.value
        for entry in report.scorecard.entries
        if entry.measured and entry.key not in MEASURED_NOT_DERIVED
    }


def check_imports_stay_clean() -> list[str]:
    """Import the package in a fresh interpreter and report any framework it pulled in."""

    program = textwrap.dedent(
        f"""
        import sys
        import agent_evals  # noqa: F401
        loaded = sorted({{name.split(".")[0] for name in sys.modules}} & set({FORBIDDEN_MODULES!r}))
        print(",".join(loaded))
        """
    )
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, check=True)
    return [name for name in result.stdout.strip().split(",") if name]


async def check_every_agent_shape_scores_the_same() -> dict[str, dict[str, float | None]]:
    """Score the same behaviour written three unrelated ways; the quality scores must agree."""

    return {name: await _scores_for(runner) for name, runner in AGENT_SHAPES.items()}


def _disagreements(scores: dict[str, dict[str, float | None]]) -> list[str]:
    first, *rest = scores.items()
    return [f"{name} scored {theirs} where {first[0]} scored {first[1]}" for name, theirs in rest if theirs != first[1]]


async def main(argv: Sequence[str] | None = None) -> int:
    """Run both checks, print what was proven, and return an exit code."""

    leaked = check_imports_stay_clean()
    if leaked:
        print(f"FAILED: importing agent_evals loaded {', '.join(leaked)}")
    else:
        print(f"ok: importing agent_evals loads none of {', '.join(FORBIDDEN_MODULES)}")

    scores = await check_every_agent_shape_scores_the_same()
    disagreements = _disagreements(scores)
    for line in disagreements:
        print(f"FAILED: {line}")
    if not disagreements:
        shapes = ", ".join(scores)
        print(f"ok: {len(scores)} agent shapes ({shapes}) scored identically")
        print("     (timing and spend are measured per run, so they are reported but not compared)")
        for metric, value in sorted(next(iter(scores.values())).items()):
            print(f"     {metric}: {'not measured' if value is None else round(value, 4)}")
    return 1 if leaked or disagreements else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
