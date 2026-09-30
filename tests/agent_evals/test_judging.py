"""Domain judges, the agreement harness, and the experimental judge-backed trajectory scorers."""

from __future__ import annotations

import pytest

from agent_evals import (
    DatasetManifest,
    EvaluationCase,
    EvaluationDataset,
    EvaluationVariant,
    GenericStep,
    GenericTrajectory,
    JudgeScorer,
    JudgeVerdict,
    LabelledExample,
    PredictionResult,
    ScorecardMetrics,
    StoredVerdicts,
    TrajectoryJudge,
    agreement_report,
    build_report,
    build_scorecard,
    multi_step_coherence,
    parse_step_verdicts,
    plan_adherence,
    render_steps,
    report_markdown,
    run_repeated,
    validate_judge,
)
from agent_evals.judging import Comparison
from agent_evals.report import RunRecord


def _case(case_id: str = "c1") -> EvaluationCase:
    return EvaluationCase(case_id, {})


def _verdict(outcome: str, *codes: str) -> JudgeVerdict:
    return JudgeVerdict(outcome, codes)


def test_a_verdict_needs_an_outcome() -> None:
    with pytest.raises(ValueError, match="outcome must be non-empty"):
        JudgeVerdict(" ")


async def test_a_judge_becomes_a_scorer_through_a_table_of_what_each_outcome_is_worth() -> None:
    scores = {"verified": 1.0, "handled_correctly": 1.0, "failed": 0.0}
    outcomes = iter(["verified", "handled_correctly", "failed"])
    scorer = JudgeScorer(lambda case, output: _verdict(next(outcomes)), scores)

    assert [await scorer(_case(), None) for _ in range(3)] == [{"judged_success": 1.0}] * 2 + [{"judged_success": 0.0}]


async def test_a_judge_may_be_a_coroutine_function() -> None:
    async def judge(case: EvaluationCase, output: object) -> JudgeVerdict:
        return _verdict("verified")

    assert await JudgeScorer(judge, {"verified": 1.0}, name="ok")(_case(), None) == {"ok": 1.0}


async def test_an_outcome_the_score_table_does_not_name_is_an_error_not_a_zero() -> None:
    scorer = JudgeScorer(lambda case, output: _verdict("something_new"), {"verified": 1.0})

    with pytest.raises(ValueError, match="'something_new', which has no score"):
        await scorer(_case(), None)


async def test_a_judge_scorer_works_inside_a_run() -> None:
    scorer = JudgeScorer(
        lambda case, output: _verdict("verified" if case.case_id != "c2" else "failed"),
        {"verified": 1.0, "failed": 0.0},
    )
    cases = [_case("c1"), _case("c2")]

    run = await run_repeated(cases, [EvaluationVariant("v")], lambda c, v: PredictionResult(answer="a"), scorer)

    assert [row.result.metrics for row in run.rows] == [{"judged_success": 1.0}, {"judged_success": 0.0}]


def _examples() -> list[LabelledExample]:
    rows = [
        ("a", "verified", "cat1"),
        ("b", "verified", "cat1"),
        ("c", "failed", "cat2"),
        ("d", "handled_correctly", "cat2"),
    ]
    return [LabelledExample(key, _case(key), None, expected, group) for key, expected, group in rows]


async def test_the_harness_counts_agreement_and_lists_every_disagreement() -> None:
    answers = {
        "a": _verdict("verified"),
        "b": _verdict("failed", "wrong_scope"),
        "c": _verdict("failed"),
        "d": _verdict("verified"),
    }

    report = await validate_judge(_examples(), StoredVerdicts(answers))

    assert (report.agree, report.total) == (2, 4)
    assert report.rate == 0.5
    assert [(d.key, d.expected, d.got, list(d.codes)) for d in report.disagreements] == [
        ("b", "verified", "failed", ["wrong_scope"]),
        ("d", "handled_correctly", "verified", []),
    ]
    assert report.summary() == "agreement: 2 of 4 (50%)"


async def test_the_harness_reports_agreement_per_expected_outcome_and_a_confusion_table() -> None:
    answers = {"a": _verdict("verified"), "b": _verdict("failed"), "c": _verdict("failed"), "d": _verdict("verified")}

    report = await validate_judge(_examples(), StoredVerdicts(answers))

    assert report.by_expected == {
        "verified": {"cases": 2, "agree": 1},
        "failed": {"cases": 1, "agree": 1},
        "handled_correctly": {"cases": 1, "agree": 0},
    }
    assert report.confusion == {
        ("verified", "verified"): 1,
        ("verified", "failed"): 1,
        ("failed", "failed"): 1,
        ("handled_correctly", "verified"): 1,
    }


async def test_a_judge_that_raises_is_a_disagreement_and_never_skipped() -> None:
    def judge(case: EvaluationCase, output: object) -> JudgeVerdict:
        if case.case_id == "b":
            raise RuntimeError("model unavailable")
        return _verdict("verified")

    report = await validate_judge(_examples()[:2], judge)

    assert report.total == 2
    assert [(d.key, d.got) for d in report.disagreements] == [("b", "judge_error:RuntimeError")]


async def test_a_stored_verdict_missing_for_a_case_is_reported_as_a_judge_error() -> None:
    report = await validate_judge(_examples()[:1], StoredVerdicts({}))

    assert report.disagreements[0].got == "judge_error:KeyError"


def test_an_empty_report_has_no_rate() -> None:
    assert agreement_report([]).rate == 0.0


def test_the_group_table_matches_what_the_campaigns_rejudge_table_gave() -> None:
    """Recorded from the campaign's `agreement_table` on the same recorded and re-judged outcomes."""

    pairs = [
        ("alpha", "verified", "verified"), ("alpha", "verified", "verified"), ("alpha", "failed", "verified"),
        ("alpha", "verified", "failed"), ("alpha", "failed", "failed"), ("alpha", "failed", "agent_error"),
        ("beta", "failed", "failed"), ("beta", "handled_correctly", "verified"),
        ("beta", "verified", "handled_correctly"), ("gamma", "verified", "verified"),
    ]  # fmt: skip
    report = agreement_report([Comparison(str(i), rec, rej, (), group) for i, (group, rec, rej) in enumerate(pairs)])

    assert report.by_group({"verified"}) == {
        "alpha": {
            "agree": 3,
            "cases": 6,
            "different_outcome": 1,
            "judge_passed_label_failed": 1,
            "judge_failed_label_passed": 1,
        },
        "beta": {"agree": 1, "cases": 3, "judge_passed_label_failed": 1, "judge_failed_label_passed": 1},
        "gamma": {"agree": 1, "cases": 1},
    }


def _trajectory(*calls: str | tuple[str, dict], plan: str | None = "1. look it up 2. report it") -> GenericTrajectory:
    steps = [GenericStep(c) if isinstance(c, str) else GenericStep(c[0], c[1]) for c in calls]
    context = {"plan": plan} if plan else {}
    return GenericTrajectory(query="a private question", steps=steps, final_answer="a", llm_context=context)


def _output(*calls: str | tuple[str, dict], plan: str | None = "1. look it up 2. report it") -> PredictionResult:
    return PredictionResult(answer="a", trajectory=_trajectory(*calls, plan=plan))


def _client(reply: str, seen: list[str] | None = None):
    def client(prompt: str) -> str:
        if seen is not None:
            seen.append(prompt)
        return reply

    return client


async def test_plan_adherence_is_the_share_of_steps_the_judge_says_trace_to_the_plan() -> None:
    scorer = plan_adherence(_client('{"steps": [true, true, false, true]}'))

    score = await scorer(_case(), _output("lookup", "report", "search_web", "finish"))

    assert score == {"plan_adherence": 0.75}


async def test_coherence_is_scored_the_same_way_under_its_own_name() -> None:
    scorer = multi_step_coherence(_client('{"steps": [true, false]}'))

    assert await scorer(_case(), _output("a", "b", plan=None)) == {"multi_step_coherence": 0.5}


async def test_the_judge_client_may_be_a_coroutine_function() -> None:
    async def client(prompt: str) -> str:
        return '{"steps": [true]}'

    assert await plan_adherence(client)(_case(), _output("lookup")) == {"plan_adherence": 1.0}


async def test_a_reply_with_words_or_a_code_fence_around_the_json_is_still_read() -> None:
    reply = 'Here is my verdict:\n```json\n{"steps": [true, false]}\n```\nHope that helps.'

    assert await plan_adherence(_client(reply))(_case(), _output("a", "b")) == {"plan_adherence": 0.5}


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        ("I think it mostly followed the plan.", "did not contain a JSON object"),
        ('{"steps": "all true"}', "did not contain a JSON object"),
        ('{"steps": [1, 0]}', "did not contain a JSON object"),
        ('{"steps": [true]}', "1 verdicts for 2 steps"),
        ('{"steps": [true, true, true]}', "3 verdicts for 2 steps"),
    ],
)
async def test_a_reply_that_cannot_be_read_is_an_error_not_a_score(reply: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        await plan_adherence(_client(reply))(_case(), _output("a", "b"))


def test_step_verdicts_are_parsed_against_the_expected_count() -> None:
    assert parse_step_verdicts('{"steps": [true, false, true]}', 3) == [True, False, True]


async def test_plan_adherence_without_an_announced_plan_is_an_error_and_the_judge_is_not_asked() -> None:
    seen: list[str] = []

    with pytest.raises(ValueError, match="announced no plan"):
        await plan_adherence(_client('{"steps": [true]}', seen))(_case(), _output("a", plan=None))

    assert seen == []


async def test_a_run_with_no_steps_has_nothing_to_judge() -> None:
    with pytest.raises(ValueError, match="no steps to judge"):
        await plan_adherence(_client("{}"))(_case(), PredictionResult(answer="a", trajectory=_trajectory()))


async def test_the_judge_is_not_shown_argument_values_results_or_the_question_by_default() -> None:
    seen: list[str] = []
    trajectory = GenericTrajectory(
        query="what did Acme spend",
        steps=[
            GenericStep("lookup", {"acid": "SECRET-112774"}, observation="SECRET-RESULT"),
            GenericStep("send", error="x"),
        ],
        final_answer="SECRET-ANSWER",
        llm_context={"plan": "look up spend, then send"},
    )

    await plan_adherence(_client('{"steps": [true, true]}', seen))(
        _case(), PredictionResult(answer="a", trajectory=trajectory)
    )

    prompt = seen[0]
    assert "lookup(acid) [ok]" in prompt and "send() [failed]" in prompt
    assert "look up spend, then send" in prompt
    for private in ("SECRET", "Acme"):
        assert private not in prompt


async def test_values_and_the_question_are_sent_only_when_asked_for() -> None:
    seen: list[str] = []
    trajectory = GenericTrajectory(
        query="what did Acme spend", steps=[GenericStep("lookup", {"acid": "112774"}, observation="42")],
        llm_context={"plan": "look up"},
    )  # fmt: skip
    scorer = TrajectoryJudge(
        _client('{"steps": [true]}', seen), "plan_adherence", include_values=True, include_query=True
    )

    await scorer(_case(), PredictionResult(answer="a", trajectory=trajectory))

    assert "acid=112774" in seen[0] and "-> 42" in seen[0] and "what did Acme spend" in seen[0]


def test_long_values_are_cut_when_they_are_sent() -> None:
    rendered = render_steps([GenericStep("lookup", {"x": "y" * 500})], include_values=True)

    assert len(rendered) < 260


def test_a_criterion_that_is_not_one_of_the_two_is_refused() -> None:
    with pytest.raises(ValueError, match="criterion must be"):
        TrajectoryJudge(_client("{}"), "creativity")  # type: ignore[arg-type]


def test_the_judge_scorers_are_marked_experimental() -> None:
    assert plan_adherence(_client("{}")).experimental is True
    assert multi_step_coherence(_client("{}")).experimental is True


JUDGED = ScorecardMetrics(plan_adherence="plan_adherence")  # a judged metric is opted into


async def _plan_run():
    cases = [EvaluationCase(f"c{n}", {}) for n in range(1, 5)]
    manifest = DatasetManifest.from_dataset(EvaluationDataset("d", "1", cases), suite="capability")

    def runner(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
        return _output("lookup", "report")

    scorer = plan_adherence(_client('{"steps": [true, false]}'))
    run = await run_repeated(cases, [EvaluationVariant("v")], runner, [lambda c, o: {"success": 1.0}, scorer])
    return run, manifest


async def test_a_scorecard_with_a_judged_plan_adherence_says_the_judge_is_experimental_and_unmeasured() -> None:
    run, _ = await _plan_run()

    entry = build_scorecard(run, "v", metrics=JUDGED, resamples=100).entry("plan_adherence")

    assert entry.measured is True and entry.value == 0.5
    assert entry.reason == "experimental judge: agreement with labels not measured"


async def test_a_scorecard_carries_the_judges_measured_agreement() -> None:
    run, manifest = await _plan_run()
    agreement = agreement_report(
        [Comparison(str(n), "verified", "verified" if n < 41 else "failed") for n in range(50)]
    )

    report = build_report(
        run, manifest, "v", record=RunRecord.from_run(run, manifest, "v", run_id="r", created_at="t"),
        metrics=JUDGED, resamples=100, judge_agreement=agreement,
    )  # fmt: skip

    assert report.scorecard.entry("plan_adherence").reason == "experimental judge: agreed with labels on 41 of 50 (82%)"
    assert "measured (experimental judge: agreed with labels on 41 of 50 (82%))" in report_markdown(report)
