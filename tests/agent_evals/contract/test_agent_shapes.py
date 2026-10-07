"""Two agent shapes the library claims to support but had no end-to-end test for.

The five-agent proof next door covers agents that call tools and run in this process. These two cover
the shapes it does not: an agent with no tools at all, and one that lives behind HTTP. Both go all the
way to a scorecard, because the interesting failures are in reporting rather than in running.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib import request as urlrequest
from urllib.error import HTTPError

import pytest

from agent_evals import (
    DatasetManifest,
    EvaluationCase,
    EvaluationDataset,
    EvaluationVariant,
    ExactMatch,
    GenericStep,
    GenericTrajectory,
    PredictionResult,
    RunRecord,
    RunSettings,
    ToolSelection,
    TransientError,
    build_report,
    run_suite,
)

CAPITALS = {"France": "Paris", "Peru": "Lima"}


def _success(case: EvaluationCase, output: PredictionResult) -> dict[str, float]:
    return ExactMatch(name="success")(EvaluationCase(case.case_id, {}, case.expected), output)


def _entries(run: Any, manifest: DatasetManifest, variant: str) -> dict[str, Any]:
    record = RunRecord.from_run(run, manifest, variant, run_id="r1")
    report = build_report(run, manifest, variant, record=record)
    return {entry.key: entry for entry in report.scorecard.entries}


def _dataset(cases: list[EvaluationCase]) -> tuple[EvaluationDataset, DatasetManifest]:
    dataset = EvaluationDataset("shapes", "v1", cases)
    return dataset, DatasetManifest.from_dataset(dataset, suite="capability")


def _score(runner: Any, cases: list[EvaluationCase], scorers: list[Any], repeats: int = 2) -> dict[str, Any]:
    dataset, manifest = _dataset(cases)
    run = asyncio.run(
        run_suite(dataset, manifest, [EvaluationVariant("v1")], runner, scorers,
                  metric_id="shapes", metric_version="1", settings=RunSettings(repeats=repeats))
    )  # fmt: skip
    return _entries(run, manifest, "v1")


# --- an agent with no tools -------------------------------------------------------------------------

QUESTIONS = [
    EvaluationCase("france", {"q": "capital of France?"}, expected="Paris"),
    # The agent does not know this one, so the run has a real failure in it rather than a clean sweep.
    EvaluationCase("chile", {"q": "capital of Chile?"}, expected="Santiago"),
]


def _chat_agent(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
    """A chat agent: an answer and nothing else. No trajectory, no tools, no cost reported."""

    country = case.inputs["q"].rsplit(" ", 1)[-1].rstrip("?")
    return PredictionResult(answer=CAPITALS.get(country, "I don't know"))


def test_an_agent_with_no_tools_is_scored_and_reported() -> None:
    """One answer right and one wrong, with no trajectory anywhere, still reaches a scorecard."""

    entries = _score(_chat_agent, QUESTIONS, [_success])

    assert entries["success_rate"].measured is True
    assert entries["success_rate"].value == pytest.approx(0.5)
    assert entries["success_rate"].runs_used == 4


def test_what_a_tool_less_agent_cannot_be_scored_on_says_why() -> None:
    """The gap that matters: a missing trajectory must read as "nobody checked", never as a zero."""

    entries = _score(_chat_agent, QUESTIONS, [_success])

    for key in ("tool_selection", "argument_correctness", "execution_efficiency", "policy_violation"):
        entry = entries[key]
        assert entry.measured is False, key
        assert entry.value is None, key
        assert entry.reason, f"{key} is not measured and must say why"


def test_a_trajectory_scorer_on_a_tool_less_agent_is_recorded_as_a_failure_with_its_reason() -> None:
    """Asking for tool selection when nothing declares the expected tools is a configuration mistake.

    It neither crashes the run nor quietly scores zero: the row keeps the reason, and the metric is
    left unmeasured, so the mistake is visible in the output rather than mistaken for a bad agent.
    """

    dataset, manifest = _dataset(QUESTIONS)
    run = asyncio.run(
        run_suite(dataset, manifest, [EvaluationVariant("v1")], _chat_agent, [_success, ToolSelection()],
                  metric_id="shapes", metric_version="1", settings=RunSettings(repeats=1))
    )  # fmt: skip

    assert len(run.failed_rows()) == len(QUESTIONS)
    reason = run.failed_rows()[0].result.error
    assert "expected['tools']" in reason
    assert run.failed_rows()[0].result.metrics == {}


# --- an agent behind HTTP ---------------------------------------------------------------------------


class _AgentHandler(BaseHTTPRequestHandler):
    """A stand-in for an agent served over HTTP: answers, or fails on demand."""

    fail_times = 0

    def do_POST(self) -> None:  # the name the base class dispatches on
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if _AgentHandler.fail_times > 0:
            _AgentHandler.fail_times -= 1
            self.send_error(503, "overloaded")
            return
        country = body["question"].rsplit(" ", 1)[-1].rstrip("?")
        payload = json.dumps(
            {"answer": CAPITALS.get(country, "I don't know"), "tools": [{"name": "lookup", "args": {"of": country}}]}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args: Any) -> None:
        """Keep the test output quiet."""


@pytest.fixture
def agent_url() -> Any:
    _AgentHandler.fail_times = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _AgentHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _remote_runner(url: str) -> Any:
    """The glue a caller writes for an agent that is not in this process."""

    def run_one(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
        payload = json.dumps({"question": case.inputs["q"]}).encode()
        post = urlrequest.Request(url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urlrequest.urlopen(post, timeout=10) as response:
                body = json.loads(response.read())
        except HTTPError as error:
            # 503 is the server asking us to come back, which is what TransientError means here.
            raise TransientError(f"the agent returned {error.code}") from error
        steps = [GenericStep(call["name"], call["args"]) for call in body["tools"]]
        return PredictionResult(
            answer=body["answer"],
            trajectory=GenericTrajectory(case.inputs["q"], steps, body["answer"]),
        )

    return run_one


HTTP_CASES = [
    EvaluationCase("france", {"q": "capital of France?"}, expected="Paris"),
]


def test_an_agent_behind_http_is_scored_like_any_other(agent_url: str) -> None:
    """Nothing in the library cares that the agent is a web server rather than a function."""

    entries = _score(
        _remote_runner(agent_url), HTTP_CASES, [_success, ToolSelection(expected_tools=lambda _: ["lookup"])]
    )

    assert entries["success_rate"].value == pytest.approx(1.0)
    assert entries["tool_selection"].measured is True
    assert entries["tool_selection"].value == pytest.approx(1.0)


def test_a_remote_agent_that_is_briefly_unavailable_is_retried(agent_url: str) -> None:
    """A 503 is worth retrying; the run should recover rather than record a failure."""

    _AgentHandler.fail_times = 1
    dataset, manifest = _dataset(HTTP_CASES)
    run = asyncio.run(
        run_suite(dataset, manifest, [EvaluationVariant("v1")], _remote_runner(agent_url), [_success],
                  metric_id="shapes", metric_version="1",
                  settings=RunSettings(repeats=1, max_retries=2, retry_errors=(TransientError,)))
    )  # fmt: skip

    assert run.failed_rows() == ()
    assert _entries(run, manifest, "v1")["success_rate"].value == pytest.approx(1.0)


def test_a_remote_agent_that_stays_down_is_recorded_as_a_failed_run(agent_url: str) -> None:
    """The error path: when retrying does not help, the row keeps the reason instead of vanishing."""

    _AgentHandler.fail_times = 99
    dataset, manifest = _dataset(HTTP_CASES)
    run = asyncio.run(
        run_suite(dataset, manifest, [EvaluationVariant("v1")], _remote_runner(agent_url), [_success],
                  metric_id="shapes", metric_version="1",
                  settings=RunSettings(repeats=1, max_retries=1, retry_errors=(TransientError,)))
    )  # fmt: skip

    failed = run.failed_rows()
    assert len(failed) == 1
    assert "503" in failed[0].result.error
