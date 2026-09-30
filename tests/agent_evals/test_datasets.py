"""Datasets on disk, frozen manifests, disjoint splits, and the two kinds of suite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_evals import (
    DatasetManifest,
    EvaluationCase,
    EvaluationDataset,
    EvaluationVariant,
    MetricMismatchError,
    PredictionResult,
    RunSettings,
    SuiteRule,
    assert_disjoint,
    load_dataset,
    load_frozen_dataset,
    load_manifest,
    run_repeated,
    run_suite,
    save_dataset,
    save_manifest,
    split_by_group,
    suite_verdict,
)

SECRET_TEXT = "what did the Acme campaign spend last week?"


def _dataset() -> EvaluationDataset:
    return EvaluationDataset(
        "questions",
        "v1",
        [
            EvaluationCase(
                "c1", {"query": SECRET_TEXT, "category": "lookup"}, expected={"value": 4.5}, source_trace_id="t1"
            ),
            EvaluationCase("c2", {"query": 'naïve, comma-separated, "quoted"', "category": "ranking"}, expected="a,b"),
            EvaluationCase("c3", {"query": "third"}, expected=None, source_investigation_digest="sha256:abc"),
        ],
    )


@pytest.mark.parametrize("suffix", [".json", ".jsonl", ".csv"])
def test_a_dataset_saved_loaded_and_saved_again_gives_the_same_bytes_and_digest(tmp_path: Path, suffix: str) -> None:
    original = _dataset()
    first, second = tmp_path / f"a{suffix}", tmp_path / f"b{suffix}"

    save_dataset(original, first)
    reloaded = load_dataset(first, dataset_id="questions", version="v1")
    save_dataset(reloaded, second)

    assert first.read_bytes() == second.read_bytes()
    assert reloaded.manifest_digest == original.manifest_digest
    assert list(reloaded.cases) == list(original.cases)


def test_a_json_and_a_jsonl_dataset_carry_their_own_id_and_version(tmp_path: Path) -> None:
    for suffix in (".json", ".jsonl"):
        save_dataset(_dataset(), tmp_path / f"d{suffix}")

        loaded = load_dataset(tmp_path / f"d{suffix}")

        assert (loaded.dataset_id, loaded.version) == ("questions", "v1")


def test_a_csv_dataset_needs_its_id_and_version_given_again(tmp_path: Path) -> None:
    save_dataset(_dataset(), tmp_path / "d.csv")

    with pytest.raises(ValueError, match="needs dataset_id and version"):
        load_dataset(tmp_path / "d.csv")


def test_a_format_that_is_not_supported_is_refused_on_save_and_on_load(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported dataset format"):
        save_dataset(_dataset(), tmp_path / "d.parquet")
    (tmp_path / "d.parquet").write_text("x")
    with pytest.raises(ValueError, match="unsupported dataset format"):
        load_dataset(tmp_path / "d.parquet")


def test_a_manifest_holds_ids_and_a_digest_and_no_case_text(tmp_path: Path) -> None:
    dataset = _dataset()
    manifest = DatasetManifest.from_dataset(dataset, suite="regression", metric_id="accuracy", metric_version="1")

    save_manifest(manifest, tmp_path / "m.json")

    written = (tmp_path / "m.json").read_text()
    assert SECRET_TEXT not in written
    assert "Acme" not in written
    assert json.loads(written)["case_ids"] == ["c1", "c2", "c3"]
    assert load_manifest(tmp_path / "m.json") == manifest
    assert manifest.digest == dataset.manifest_digest


def test_a_dataset_that_changed_since_it_was_frozen_is_refused(tmp_path: Path) -> None:
    dataset = _dataset()
    save_dataset(dataset, tmp_path / "d.jsonl")
    save_manifest(DatasetManifest.from_dataset(dataset, suite="capability"), tmp_path / "m.json")
    lines = (tmp_path / "d.jsonl").read_text().replace("third", "3rd")
    (tmp_path / "d.jsonl").write_text(lines)

    with pytest.raises(ValueError, match="has changed since it was frozen"):
        load_frozen_dataset(tmp_path / "d.jsonl", tmp_path / "m.json")


def test_an_unchanged_dataset_loads_with_its_manifest(tmp_path: Path) -> None:
    dataset = _dataset()
    save_dataset(dataset, tmp_path / "d.jsonl")
    save_manifest(DatasetManifest.from_dataset(dataset, suite="capability"), tmp_path / "m.json")

    loaded, manifest = load_frozen_dataset(tmp_path / "d.jsonl", tmp_path / "m.json")

    assert loaded.manifest_digest == manifest.digest


def test_a_manifest_with_an_unknown_suite_or_half_a_metric_is_refused() -> None:
    dataset = _dataset()
    with pytest.raises(ValueError, match="suite must be one of"):
        DatasetManifest.from_dataset(dataset, suite="smoke")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="given together"):
        DatasetManifest.from_dataset(dataset, suite="regression", metric_id="accuracy")


def answering(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
    return PredictionResult(answer="ok")


def score(case: EvaluationCase, output: PredictionResult) -> dict[str, float]:
    return {"accuracy": 1.0}


async def test_a_run_with_the_wrong_metric_version_fails_before_any_case_runs() -> None:
    dataset = _dataset()
    manifest = DatasetManifest.from_dataset(dataset, suite="regression", metric_id="accuracy", metric_version="1")
    calls = 0

    def counting(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
        nonlocal calls
        calls += 1
        return PredictionResult(answer="ok")

    with pytest.raises(MetricMismatchError, match="expects metric accuracy 1, not accuracy 2"):
        await run_suite(
            dataset, manifest, [EvaluationVariant("v")], counting, score, metric_id="accuracy", metric_version="2"
        )

    assert calls == 0


async def test_a_run_with_the_right_metric_runs() -> None:
    dataset = _dataset()
    manifest = DatasetManifest.from_dataset(dataset, suite="regression", metric_id="accuracy", metric_version="1")

    run = await run_suite(
        dataset, manifest, [EvaluationVariant("v")], answering, score, metric_id="accuracy", metric_version="1"
    )

    assert len(run.rows) == 3


async def test_a_dataset_that_declares_no_metric_accepts_any() -> None:
    dataset = _dataset()
    manifest = DatasetManifest.from_dataset(dataset, suite="capability")

    run = await run_suite(
        dataset, manifest, [EvaluationVariant("v")], answering, score, metric_id="anything", metric_version="9"
    )

    assert len(run.rows) == 3


async def test_a_run_of_a_dataset_other_than_the_frozen_one_fails_before_any_case_runs() -> None:
    manifest = DatasetManifest.from_dataset(_dataset(), suite="regression")
    other = EvaluationDataset("questions", "v1", [EvaluationCase("c1", {"query": "different"})])

    with pytest.raises(ValueError, match="has changed since it was frozen"):
        await run_suite(other, manifest, [EvaluationVariant("v")], answering, score, metric_id="m", metric_version="1")


def _records() -> list[EvaluationCase]:
    categories = {"breakdown": 7, "ranking": 1, "trend_over_time": 2, "comparison": 10}
    return [
        EvaluationCase(f"{category}-{index:02d}", {"category": category})
        for category, count in categories.items()
        for index in range(count)
    ]


def test_a_split_gives_the_parts_the_campaign_question_banks_gave() -> None:
    """Recorded from the campaign's `split_banks` on the same ids and seed."""

    mining, evaluation = split_by_group(_records(), lambda case: case.inputs["category"], seed=7)

    assert [case.case_id for case in mining] == [
        "breakdown-00", "breakdown-03", "breakdown-04", "breakdown-01", "breakdown-05",
        "comparison-02", "comparison-09", "comparison-08", "comparison-01", "comparison-04",
        "comparison-05", "comparison-06", "ranking-00", "trend_over_time-00",
    ]  # fmt: skip
    assert [case.case_id for case in evaluation] == [
        "breakdown-02", "breakdown-06", "comparison-03", "comparison-00", "comparison-07", "trend_over_time-01",
    ]  # fmt: skip


def test_the_two_parts_of_a_split_are_disjoint_and_cover_every_case() -> None:
    cases = _records()

    first, second = split_by_group(cases, lambda case: case.inputs["category"], seed=3)

    assert_disjoint(first, second)
    assert sorted(case.case_id for case in [*first, *second]) == sorted(case.case_id for case in cases)


def test_the_same_seed_gives_the_same_split_and_another_seed_a_different_one() -> None:
    group = lambda case: case.inputs["category"]  # noqa: E731

    assert split_by_group(_records(), group, seed=1) == split_by_group(_records(), group, seed=1)
    assert split_by_group(_records(), group, seed=1) != split_by_group(_records(), group, seed=2)


def test_a_group_of_one_goes_to_the_first_part_and_a_group_of_two_is_shared() -> None:
    first, second = split_by_group(_records(), lambda case: case.inputs["category"], seed=7)

    assert [case.case_id for case in first if case.inputs["category"] == "ranking"] == ["ranking-00"]
    assert [case.inputs["category"] for case in second].count("trend_over_time") == 1
    assert [case.inputs["category"] for case in first].count("trend_over_time") == 1


def test_a_first_share_outside_zero_and_one_is_refused() -> None:
    for share in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError, match="first_share must be greater than 0 and less than 1"):
            split_by_group(_records(), lambda case: "g", seed=1, first_share=share)


def test_parts_that_share_a_case_are_reported_with_the_shared_ids() -> None:
    cases = _records()

    with pytest.raises(ValueError, match=r"parts share cases: \['breakdown-00'\]"):
        assert_disjoint(cases[:2], cases[:1])


def test_a_dataset_can_be_checked_for_disjointness_too() -> None:
    cases = _records()

    assert_disjoint(EvaluationDataset("a", "1", cases[:3]), EvaluationDataset("b", "1", cases[3:6]))


async def _run_with_scores(scores: dict[str, float | None], *, repeats: int = 1):
    """Scores by case id; None makes the case raise."""

    cases = [EvaluationCase(case_id, {}) for case_id in scores]

    def runner(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
        if scores[case.case_id] is None:
            raise RuntimeError("tool down")
        return PredictionResult(answer="x")

    def scorer(case: EvaluationCase, output: PredictionResult) -> dict[str, float]:
        return {"quality": scores[case.case_id]}  # type: ignore[dict-item]

    dataset = EvaluationDataset("d", "1", cases)
    run = await run_repeated(cases, [EvaluationVariant("v")], runner, scorer, RunSettings(repeats=repeats))
    return dataset, run


async def test_a_regression_suite_fails_when_any_case_falls_below_the_bar() -> None:
    dataset, run = await _run_with_scores({"c1": 1.0, "c2": 1.0, "c3": 0.9})
    manifest = DatasetManifest.from_dataset(dataset, suite="regression")

    verdict = suite_verdict(run, "v", manifest, SuiteRule("quality"))

    assert verdict.passed is False
    assert verdict.failing_case_ids == ("c3",)
    assert verdict.pass_rate == pytest.approx(2 / 3)
    assert verdict.interval is None
    assert verdict.rule.startswith("regression:")


async def test_a_regression_suite_can_allow_a_share_of_cases_to_miss() -> None:
    dataset, run = await _run_with_scores({f"c{n}": (0.5 if n == 0 else 1.0) for n in range(10)})
    manifest = DatasetManifest.from_dataset(dataset, suite="regression")

    strict = suite_verdict(run, "v", manifest, SuiteRule("quality"))
    lenient = suite_verdict(run, "v", manifest, SuiteRule("quality", required_pass_rate=0.9))

    assert (strict.passed, lenient.passed) == (False, True)


async def test_a_case_that_failed_to_run_counts_as_zero_in_a_regression_suite() -> None:
    dataset, run = await _run_with_scores({"c1": 1.0, "c2": None})
    manifest = DatasetManifest.from_dataset(dataset, suite="regression")

    verdict = suite_verdict(run, "v", manifest, SuiteRule("quality"))

    assert verdict.passed is False
    assert verdict.failing_case_ids == ("c2",)


async def test_a_regression_case_with_one_bad_repeat_among_good_ones_does_not_pass_at_a_perfect_bar() -> None:
    outcomes = iter([1.0, 0.0, 1.0])

    def runner(case: EvaluationCase, variant: EvaluationVariant) -> PredictionResult:
        return PredictionResult(answer="x")

    def scorer(case: EvaluationCase, output: PredictionResult) -> dict[str, float]:
        return {"quality": next(outcomes)}

    cases = [EvaluationCase("c1", {})]
    run = await run_repeated(cases, [EvaluationVariant("v")], runner, scorer, RunSettings(repeats=3))
    manifest = DatasetManifest.from_dataset(EvaluationDataset("d", "1", cases), suite="regression")

    verdict = suite_verdict(run, "v", manifest, SuiteRule("quality"))

    assert verdict.passed is False
    assert verdict.mean_score == pytest.approx(2 / 3)


async def test_a_capability_suite_reports_partial_credit_and_an_interval_and_never_passes_or_fails() -> None:
    dataset, run = await _run_with_scores({f"c{n}": 0.2 + 0.1 * n for n in range(8)})
    manifest = DatasetManifest.from_dataset(dataset, suite="capability")

    verdict = suite_verdict(run, "v", manifest, SuiteRule("quality", resamples=2_000))

    assert verdict.passed is None
    assert verdict.pass_rate is None
    assert verdict.mean_score == pytest.approx(0.55)
    assert verdict.interval is not None
    assert verdict.interval.lower <= verdict.mean_score <= verdict.interval.upper
    assert verdict.rule.startswith("capability:")


async def test_the_same_results_get_different_verdicts_from_the_two_suite_types() -> None:
    dataset, run = await _run_with_scores({"c1": 1.0, "c2": 0.8, "c3": 1.0, "c4": 1.0})

    regression = suite_verdict(
        run, "v", DatasetManifest.from_dataset(dataset, suite="regression"), SuiteRule("quality")
    )
    capability = suite_verdict(
        run, "v", DatasetManifest.from_dataset(dataset, suite="capability"), SuiteRule("quality")
    )

    assert regression.passed is False
    assert capability.passed is None
    assert capability.mean_score == pytest.approx(0.95)


async def test_a_run_that_did_not_produce_the_metric_is_an_error_not_a_zero() -> None:
    dataset, run = await _run_with_scores({"c1": 1.0})
    manifest = DatasetManifest.from_dataset(dataset, suite="regression")

    with pytest.raises(ValueError, match="did not produce metric 'other'"):
        suite_verdict(run, "v", manifest, SuiteRule("other"))


async def test_a_verdict_for_a_variant_that_has_no_rows_is_an_error() -> None:
    dataset, run = await _run_with_scores({"c1": 1.0})
    manifest = DatasetManifest.from_dataset(dataset, suite="regression")

    with pytest.raises(ValueError, match="no rows for variant 'ghost'"):
        suite_verdict(run, "ghost", manifest, SuiteRule("quality"))


def test_a_suite_rule_with_a_threshold_outside_zero_and_one_is_refused() -> None:
    with pytest.raises(ValueError, match="case_pass_score must be between 0 and 1"):
        SuiteRule("quality", case_pass_score=1.5)
    with pytest.raises(ValueError, match="required_pass_rate must be between 0 and 1"):
        SuiteRule("quality", required_pass_rate=-0.1)
