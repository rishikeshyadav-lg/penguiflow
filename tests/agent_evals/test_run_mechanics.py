"""Repeats, concurrency, timeouts, retries, and a run that can be killed and resumed."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from agent_evals import (
    EvaluationCase,
    EvaluationVariant,
    GenericStep,
    GenericTrajectory,
    JsonlRowSink,
    PredictionResult,
    RunRow,
    RunSettings,
    TransientError,
    run_repeated,
)

ROW_KEYS = {
    "answer",
    "variant_id",
    "category",
    "cost_usd",
    "latency_ms",
    "llm_usage",
    "native_trace_id",
    "pattern_key",
    "repeat",
    "set",
    "tool_calls",
}

CASES = [
    EvaluationCase("c1", {"query": "a", "category": "lookup", "set": "mining"}, expected=1, source_trace_id="t1"),
    EvaluationCase("c2", {"query": "bb"}, expected=2),
    EvaluationCase("c3", {"query": "ccc"}, expected=3),
]
VARIANTS = [EvaluationVariant("base"), EvaluationVariant("cand", advisory_skill="Be exact.")]


def score(case: EvaluationCase, output: PredictionResult) -> dict[str, float]:
    return {"correct": 1.0 if output.answer == str(case.expected) else 0.0}


def answering(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
    """Base answers every case right; cand gets c3 wrong."""

    wrong = variant.variant_id == "cand" and case.case_id == "c3"
    return PredictionResult(answer="wrong" if wrong else str(case.expected), latency_ms=10.0, cost_usd=0.01)


def canonical_keys(repeats: int) -> list[tuple[str, str, int]]:
    return [
        (case.case_id, variant.variant_id, repeat)
        for case in CASES
        for variant in VARIANTS
        for repeat in range(repeats)
    ]


def stable(rows: tuple[RunRow, ...] | list[RunRow]) -> list[dict]:
    """The recorded rows without wall-clock latency, which differs from run to run."""

    return [{**row.record(), "latency_ms": None} for row in rows]


async def test_every_expected_row_is_recorded_once_in_dataset_variant_repeat_order() -> None:
    run = await run_repeated(CASES, VARIANTS, answering, score, RunSettings(repeats=3))

    assert [row.key for row in run.rows] == canonical_keys(3)
    assert len(run.rows) == 3 * 2 * 3


async def test_results_are_the_same_at_concurrency_1_and_8_even_when_runs_finish_out_of_order() -> None:
    async def finishing_backwards(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
        # Earlier cases yield to the event loop more often, so they finish after later ones.
        for _ in range(4 - int(case.case_id[1:])):
            await asyncio.sleep(0)
        return answering(case, variant)

    settings = {"repeats": 3}
    serial = await run_repeated(CASES, VARIANTS, finishing_backwards, score, RunSettings(concurrency=1, **settings))
    parallel = await run_repeated(CASES, VARIANTS, finishing_backwards, score, RunSettings(concurrency=8, **settings))

    assert stable(serial.rows) == stable(parallel.rows)


async def test_concurrency_is_bounded_and_actually_used() -> None:
    in_flight = 0
    peak = 0

    async def counting(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        for _ in range(3):
            await asyncio.sleep(0)
        in_flight -= 1
        return answering(case, variant)

    await run_repeated(CASES, VARIANTS, counting, score, RunSettings(repeats=2, concurrency=3))

    assert peak == 3


async def test_a_run_that_is_cancelled_part_way_resumes_without_redoing_finished_rows(tmp_path: Path) -> None:
    sink = JsonlRowSink(tmp_path / "runs.jsonl")
    started: list[tuple[str, str]] = []
    reached_fourth = asyncio.Event()

    async def hangs_on_the_fourth(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
        started.append((case.case_id, variant.variant_id))
        if len(started) == 4:
            reached_fourth.set()
            await asyncio.Event().wait()  # never returns: the process is "killed" here
        return answering(case, variant)

    task = asyncio.create_task(run_repeated(CASES, VARIANTS, hangs_on_the_fourth, score, RunSettings(repeats=2), sink))
    await reached_fourth.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert [key for key in sink.load()] == canonical_keys(2)[:3]

    resumed_calls: list[tuple[str, str]] = []

    def recording(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
        resumed_calls.append((case.case_id, variant.variant_id))
        return answering(case, variant)

    resumed = await run_repeated(CASES, VARIANTS, recording, score, RunSettings(repeats=2), sink)
    uninterrupted = await run_repeated(CASES, VARIANTS, answering, score, RunSettings(repeats=2))

    assert len(resumed_calls) == 12 - 3
    assert stable(resumed.rows) == stable(uninterrupted.rows)


async def test_a_half_written_last_line_from_a_killed_run_is_dropped_and_redone(tmp_path: Path) -> None:
    sink = JsonlRowSink(tmp_path / "runs.jsonl")
    await run_repeated(CASES[:1], VARIANTS[:1], answering, score, RunSettings(repeats=2), sink)
    with sink.path.open("a", encoding="utf-8") as handle:
        handle.write('{"arm": "base", "case_id": "c2", "rep')  # the process died mid-write

    resumed = await run_repeated(CASES, VARIANTS[:1], answering, score, RunSettings(repeats=2), sink)

    assert [row.key for row in resumed.rows] == [
        (case.case_id, "base", repeat) for case in CASES for repeat in range(2)
    ]
    assert [key for key in sink.load()] != []
    assert len(sink.path.read_text().splitlines()) == 6


async def test_a_damaged_line_in_the_middle_of_the_file_is_an_error_not_something_skipped(tmp_path: Path) -> None:
    sink = JsonlRowSink(tmp_path / "runs.jsonl")
    await run_repeated(CASES, VARIANTS[:1], answering, score, RunSettings(), sink)
    lines = sink.path.read_text().splitlines()
    sink.path.write_text("\n".join([lines[0], "not json", lines[2]]) + "\n")

    with pytest.raises(ValueError, match="line 2 is not a valid run row"):
        sink.load()


async def test_a_sink_holding_rows_from_another_run_is_refused(tmp_path: Path) -> None:
    sink = JsonlRowSink(tmp_path / "runs.jsonl")
    await run_repeated(CASES, VARIANTS, answering, score, RunSettings(repeats=3), sink)

    with pytest.raises(ValueError, match="not part of this run"):
        await run_repeated(CASES, VARIANTS, answering, score, RunSettings(repeats=2), sink)


async def test_a_recorded_failure_counts_as_done_on_resume(tmp_path: Path) -> None:
    sink = JsonlRowSink(tmp_path / "runs.jsonl")
    calls = 0

    def fails_once(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
        nonlocal calls
        calls += 1
        raise RuntimeError("tool down")

    first = await run_repeated(CASES[:1], VARIANTS[:1], fails_once, score, sink=sink)
    second = await run_repeated(CASES[:1], VARIANTS[:1], answering, score, sink=sink)

    assert calls == 1
    assert first.failed_rows()[0].result.error == "RuntimeError: tool down"
    assert second.failed_rows()[0].result.error == "RuntimeError: tool down"


async def test_retry_errors_runs_failed_rows_again_and_leaves_one_row_per_key(tmp_path: Path) -> None:
    sink = JsonlRowSink(tmp_path / "runs.jsonl")

    def broken(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
        raise RuntimeError("tool down")

    await run_repeated(CASES, VARIANTS[:1], broken, score, sink=sink)
    fixed = await run_repeated(CASES, VARIANTS[:1], answering, score, RunSettings(retry_errors=True), sink)

    assert fixed.failed_rows() == ()
    assert len(sink.path.read_text().splitlines()) == 3
    assert len(sink.load()) == 3


async def test_a_run_that_exceeds_the_timeout_is_recorded_as_a_failure_and_the_others_finish() -> None:
    async def stalls_on_c2(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
        if case.case_id == "c2":
            await asyncio.Event().wait()
        return answering(case, variant)

    run = await run_repeated(CASES, VARIANTS[:1], stalls_on_c2, score, RunSettings(timeout_s=0.01, concurrency=3))

    assert [row.result.error for row in run.rows] == [None, "TimeoutError: no result within 0.01s", None]


async def test_a_transient_failure_is_retried_until_it_succeeds() -> None:
    calls = 0

    def rate_limited_twice(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
        nonlocal calls
        calls += 1
        if calls <= 2:
            raise TransientError("429")
        return answering(case, variant)

    run = await run_repeated(CASES[:1], VARIANTS[:1], rate_limited_twice, score, RunSettings(max_retries=2))

    assert calls == 3
    assert run.rows[0].result.error is None


async def test_a_transient_failure_that_outlasts_the_retries_is_recorded_as_a_failure() -> None:
    calls = 0

    def always_rate_limited(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
        nonlocal calls
        calls += 1
        raise TransientError("429")

    run = await run_repeated(CASES[:1], VARIANTS[:1], always_rate_limited, score, RunSettings(max_retries=1))

    assert calls == 2
    assert run.rows[0].result.error == "TransientError: 429"


async def test_an_ordinary_error_is_not_retried() -> None:
    calls = 0

    def broken(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
        nonlocal calls
        calls += 1
        raise RuntimeError("bad input")

    run = await run_repeated(CASES[:1], VARIANTS[:1], broken, score, RunSettings(max_retries=3))

    assert calls == 1
    assert run.rows[0].result.error == "RuntimeError: bad input"


async def test_latency_reported_by_the_runner_is_kept_and_otherwise_measured() -> None:
    def returns_a_plain_answer(case: EvaluationCase, variant: EvaluationVariant) -> str:
        return str(case.expected)

    def plain_score(case: EvaluationCase, output: str) -> dict[str, float]:
        return {"correct": 1.0}

    reported = await run_repeated(CASES[:1], VARIANTS[:1], answering, score)
    measured = await run_repeated(CASES[:1], VARIANTS[:1], returns_a_plain_answer, plain_score)

    assert reported.rows[0].result.latency_ms == 10.0
    assert reported.rows[0].result.cost_usd == 0.01
    assert measured.rows[0].result.latency_ms is not None and measured.rows[0].result.latency_ms >= 0
    assert measured.rows[0].answer == "1"


async def test_a_recorded_row_has_every_key_of_the_campaign_runs_file_and_reads_back_the_same(tmp_path: Path) -> None:
    def with_a_tool_call(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
        trajectory = GenericTrajectory(
            query="a",
            steps=[GenericStep("lookup", {"id": 1}, observation={"rows": 2}), GenericStep("send", error="denied")],
            final_answer="1",
        )
        return PredictionResult(
            answer="1", trajectory=trajectory, llm_usage={"input_tokens": 5}, extra={"native_trace_id": "native-9"}
        )

    sink = JsonlRowSink(tmp_path / "runs.jsonl")
    await run_repeated(CASES[:1], VARIANTS[:1], with_a_tool_call, score, sink=sink)

    written = json.loads(sink.path.read_text().splitlines()[0])
    assert ROW_KEYS <= set(written)
    assert written["variant_id"] == "base"
    assert written["category"] == "lookup"
    assert written["set"] == "mining"
    assert written["native_trace_id"] == "native-9"
    assert written["llm_usage"] == {"input_tokens": 5}
    assert written["tool_calls"] == [
        {"tool": "lookup", "args": {"id": 1}, "result": {"rows": 2}, "status": "ok"},
        {"tool": "send", "args": {}, "result": None, "status": "error"},
    ]
    reloaded = sink.load()[("c1", "base", 0)]
    assert reloaded.record() == written


async def test_the_source_trace_id_names_the_row_when_the_runner_gives_none() -> None:
    run = await run_repeated(CASES[:1], VARIANTS[:1], answering, score)

    assert run.rows[0].native_trace_id == "t1"


async def test_case_means_average_the_successful_repeats_of_each_case() -> None:
    outcomes = iter([1.0, 0.0, 1.0, 1.0, 0.0, 0.0])

    def scripted(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
        return PredictionResult(answer=str(case.expected) if next(outcomes) else "no")

    run = await run_repeated(CASES[:2], VARIANTS[:1], scripted, score, RunSettings(repeats=3))

    assert run.case_means("base", "correct") == {"c1": pytest.approx(2 / 3), "c2": pytest.approx(1 / 3)}


async def test_case_means_leave_out_a_case_whose_every_repeat_failed() -> None:
    def fails_on_c2(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
        if case.case_id == "c2":
            raise RuntimeError("no")
        return answering(case, variant)

    run = await run_repeated(CASES[:2], VARIANTS[:1], fails_on_c2, score, RunSettings(repeats=2))

    assert set(run.case_means("base", "correct")) == {"c1"}


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"repeats": 0}, "repeats must be at least 1"),
        ({"concurrency": 0}, "concurrency must be at least 1"),
        ({"max_retries": -1}, "max_retries must not be negative"),
        ({"timeout_s": 0}, "timeout_s must be a positive number"),
        ({"timeout_s": float("nan")}, "timeout_s must be a positive number"),
    ],
)
def test_settings_that_make_no_sense_are_refused(settings: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RunSettings(**settings)


def test_a_row_written_before_the_variant_key_was_renamed_still_reads() -> None:
    """Rows written when the variant was called `arm`, after the first consumer's vocabulary, must keep
    loading: there are recorded runs in that shape and re-running them is not free."""

    legacy = {
        "answer": "a",
        "arm": "base",
        "case_id": "c1",
        "category": None,
        "cost_usd": None,
        "error": None,
        "latency_ms": None,
        "llm_usage": {},
        "metrics": {"correct": 1.0},
        "native_trace_id": None,
        "pattern_key": None,
        "repeat": 0,
        "score_details": {},
        "set": None,
        "tool_calls": [],
    }

    row = RunRow.from_record(legacy)

    assert row.variant_id == "base"
    assert row.result.variant_id == "base"
    assert row.record()["variant_id"] == "base"
    assert "arm" not in row.record()
