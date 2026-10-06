"""Run every (case, variant, repeat) with bounded concurrency, and keep every row.

A run is a list of expected rows: one per case, variant and repeat. Each row ends up with a result or
an explicit failure; none is skipped. Rows are appended to a JSONL file as they finish, so a run that
is killed can resume and redo only what is missing. Results come back in a fixed order (dataset order,
then variant order, then repeat), whatever the concurrency was.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .evaluation import EvaluationCase, EvaluationVariant, RunOne, VariantCaseResult, _await_value
from .prediction import PredictionResult
from .runner import Scorers, _as_list, run_case_variant

logger = logging.getLogger("agent_evals.execution")

RowKey = tuple[str, str, int]  # (case_id, variant_id, repeat)


class TransientError(Exception):
    """Raised by a runner for a failure worth retrying, such as a rate limit or a dropped connection."""


@dataclass(frozen=True, slots=True)
class RunSettings:
    """How a run is executed. `repeats` runs each case that many times per variant."""

    repeats: int = 1
    concurrency: int = 1
    timeout_s: float | None = None
    max_retries: int = 0
    # Resume normally treats a recorded failure as done; set this to run failed rows again.
    retry_errors: bool = False

    def __post_init__(self) -> None:
        if self.repeats < 1:
            raise ValueError("repeats must be at least 1")
        if self.concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        if self.max_retries < 0:
            raise ValueError("max_retries must not be negative")
        if self.timeout_s is not None and not (math.isfinite(self.timeout_s) and self.timeout_s > 0):
            raise ValueError("timeout_s must be a positive number")


@dataclass(frozen=True, slots=True)
class RunRow:
    """One recorded run of one case with one variant. It is what the JSONL file holds."""

    case_id: str
    variant_id: str
    repeat: int
    result: VariantCaseResult
    answer: str | None = None
    tool_calls: Sequence[Mapping[str, Any]] = ()
    category: str | None = None
    pattern_key: str | None = None
    set_name: str | None = None
    native_trace_id: str | None = None

    @property
    def key(self) -> RowKey:
        return (self.case_id, self.variant_id, self.repeat)

    def record(self) -> dict[str, Any]:
        """The row as a JSON-ready dict.

        `variant_id` names the variant. Rows written before this key existed named it `arm`, after the
        first consumer's vocabulary; `from_record` still reads those, so old run files keep working.
        `category`, `pattern_key`, `set` and `native_trace_id` are optional labels a caller may leave None.
        """

        return {
            "answer": self.answer,
            "case_id": self.case_id,
            "category": self.category,
            "cost_usd": self.result.cost_usd,
            "error": self.result.error,
            "latency_ms": self.result.latency_ms,
            "llm_usage": dict(self.result.llm_usage),
            "metrics": dict(self.result.metrics),
            "native_trace_id": self.native_trace_id,
            "pattern_key": self.pattern_key,
            "repeat": self.repeat,
            "score_details": {name: dict(detail) for name, detail in self.result.score_details.items()},
            "set": self.set_name,
            "tool_calls": [dict(call) for call in self.tool_calls],
            "variant_id": self.variant_id,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> RunRow:
        """Read a row back, written by any version. The live output is not stored, so `result.output` is None."""

        variant_id = record["variant_id"] if "variant_id" in record else record["arm"]
        result = VariantCaseResult(
            variant_id=variant_id,
            metrics=record["metrics"],
            error=record["error"],
            score_details=record.get("score_details", {}),
            latency_ms=record["latency_ms"],
            cost_usd=record["cost_usd"],
            llm_usage=record["llm_usage"],
        )
        return cls(
            case_id=record["case_id"],
            variant_id=variant_id,
            repeat=record["repeat"],
            result=result,
            answer=record["answer"],
            tool_calls=record["tool_calls"],
            category=record["category"],
            pattern_key=record["pattern_key"],
            set_name=record["set"],
            native_trace_id=record["native_trace_id"],
        )


class JsonlRowSink:
    """Append finished rows to a JSONL file, one line per row, flushed as it is written."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> dict[RowKey, RunRow]:
        """Read every recorded row. A half-written last line, left by a killed run, is dropped."""

        if not self.path.exists():
            return {}
        lines = self.path.read_text(encoding="utf-8").split("\n")
        good_lines = [line for line in lines if line.strip()]
        rows: dict[RowKey, RunRow] = {}
        for number, line in enumerate(good_lines, start=1):
            try:
                row = RunRow.from_record(json.loads(line))
            except (ValueError, KeyError, TypeError) as error:
                if number == len(good_lines):
                    logger.warning("Dropping an unreadable last line in %s", self.path)
                    self._rewrite(good_lines[:-1])
                    break
                raise ValueError(f"{self.path} line {number} is not a valid run row") from error
            if row.key in rows:
                raise ValueError(f"{self.path} holds two rows for {row.key}")
            rows[row.key] = row
        return rows

    def append(self, row: RunRow) -> None:
        """Write one row to the end of the file."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row.record(), ensure_ascii=False, sort_keys=True, default=str) + "\n")
            handle.flush()

    def keep_only(self, rows: Sequence[RunRow]) -> None:
        """Rewrite the file so it holds exactly these rows."""

        lines = [json.dumps(row.record(), ensure_ascii=False, sort_keys=True, default=str) for row in rows]
        self._rewrite(lines)

    def _rewrite(self, lines: list[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


@dataclass(frozen=True, slots=True)
class RepeatedRun:
    """Every row of a run, in dataset, variant, repeat order."""

    settings: RunSettings
    rows: Sequence[RunRow]

    def rows_for(self, variant_id: str) -> tuple[RunRow, ...]:
        return tuple(row for row in self.rows if row.variant_id == variant_id)

    def failed_rows(self) -> tuple[RunRow, ...]:
        return tuple(row for row in self.rows if row.result.error is not None)

    def case_means(self, variant_id: str, metric: str) -> dict[str, float]:
        """Mean of one metric over each case's successful repeats. A case with none is left out."""

        values: dict[str, list[float]] = {}
        for row in self.rows_for(variant_id):
            if row.result.error is None and metric in row.result.metrics:
                values.setdefault(row.case_id, []).append(row.result.metrics[metric])
        return {case_id: sum(scores) / len(scores) for case_id, scores in values.items()}


def _tool_calls(output: Any) -> list[dict[str, Any]]:
    if not isinstance(output, PredictionResult) or output.trajectory is None:
        return []
    return [
        {
            "tool": step.tool,
            "args": dict(step.args),
            "result": step.observation,
            "status": "error" if step.error or step.failure else "ok",
        }
        for step in output.trajectory.steps
    ]


def _answer(output: Any) -> str | None:
    if isinstance(output, PredictionResult):
        return output.answer
    return output if isinstance(output, str) else None


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _row(case: EvaluationCase, variant: EvaluationVariant, repeat: int, result: VariantCaseResult) -> RunRow:
    output = result.output
    extra = output.extra if isinstance(output, PredictionResult) else {}
    return RunRow(
        case_id=case.case_id,
        variant_id=variant.variant_id,
        repeat=repeat,
        result=result,
        answer=_answer(output),
        tool_calls=_tool_calls(output),
        category=_text(case.inputs.get("category")),
        pattern_key=_text(case.inputs.get("pattern_key")),
        set_name=_text(case.inputs.get("set")),
        native_trace_id=_text(extra.get("native_trace_id")) or case.source_trace_id,
    )


async def _run_item(
    case: EvaluationCase,
    variant: EvaluationVariant,
    run_one: RunOne,
    scorers: list[Any],
    settings: RunSettings,
) -> VariantCaseResult:
    """Run one item, retrying only failures the runner called transient, each attempt under the timeout."""

    for attempt in range(settings.max_retries + 1):
        transient_failures: list[TransientError] = []

        async def run_and_note(case: EvaluationCase, variant: EvaluationVariant) -> Any:
            try:
                return await _await_value(run_one(case, variant))
            except TransientError as error:
                transient_failures.append(error)
                raise

        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                run_case_variant(case, variant, run_and_note, scorers), timeout=settings.timeout_s
            )
        except TimeoutError:
            result = VariantCaseResult(
                variant_id=variant.variant_id, error=f"TimeoutError: no result within {settings.timeout_s}s"
            )
        if result.latency_ms is None:
            result = replace(result, latency_ms=(time.perf_counter() - started) * 1000)
        if not transient_failures or attempt == settings.max_retries:
            return result
        logger.info("Retrying %s/%s after a transient failure", case.case_id, variant.variant_id)
    raise AssertionError("unreachable: the last attempt always returns")


async def run_repeated(
    cases: Sequence[EvaluationCase],
    variants: Sequence[EvaluationVariant],
    run_one: RunOne,
    scorers: Scorers,
    settings: RunSettings = RunSettings(),
    sink: JsonlRowSink | None = None,
) -> RepeatedRun:
    """Run every case with every variant `settings.repeats` times and return every row.

    With a sink, rows already recorded are kept and not run again, and new rows are appended as they
    finish. Cancelling the run leaves the finished rows in the sink, so a later call resumes.
    """

    scorer_list = _as_list(scorers)
    items = [(case, variant, repeat) for case in cases for variant in variants for repeat in range(settings.repeats)]
    expected_keys = {(case.case_id, variant.variant_id, repeat) for case, variant, repeat in items}

    recorded = sink.load() if sink is not None else {}
    unknown = sorted(set(recorded) - expected_keys)
    if unknown:
        raise ValueError(f"the sink holds rows that are not part of this run, for example {unknown[0]}")
    if settings.retry_errors and sink is not None:
        # Drop failed rows from the file too, so the new attempt does not sit beside the old one.
        recorded = {key: row for key, row in recorded.items() if row.result.error is None}
        sink.keep_only(list(recorded.values()))

    limit = asyncio.Semaphore(settings.concurrency)

    async def run_missing(case: EvaluationCase, variant: EvaluationVariant, repeat: int) -> RunRow:
        async with limit:
            result = await _run_item(case, variant, run_one, scorer_list, settings)
        row = _row(case, variant, repeat, result)
        if sink is not None:
            sink.append(row)
        return row

    async def row_for(case: EvaluationCase, variant: EvaluationVariant, repeat: int) -> RunRow:
        existing = recorded.get((case.case_id, variant.variant_id, repeat))
        return existing if existing is not None else await run_missing(case, variant, repeat)

    rows = await asyncio.gather(*(row_for(case, variant, repeat) for case, variant, repeat in items))
    return RepeatedRun(settings=settings, rows=tuple(rows))


__all__ = [
    "JsonlRowSink",
    "RepeatedRun",
    "RowKey",
    "RunRow",
    "RunSettings",
    "TransientError",
    "run_repeated",
]
