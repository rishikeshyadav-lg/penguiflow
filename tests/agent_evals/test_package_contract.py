"""What `agent_evals` promises: it stands alone, the learning control plane sits on top of it
unchanged, and its numbers and events do not drift.

The golden values below were recorded from the evaluation code while it still lived inside the
learning control plane, before it moved here. If one of them changes, a refactor has altered
behaviour that stored evidence and frozen datasets depend on.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

import agent_evals
from agent_evals import (
    EvaluationCase,
    EvaluationDataset,
    EvaluationRequest,
    EvaluationVariant,
    EvidenceContext,
    EvidenceEvent,
    LocalEvaluationBackend,
    MetricSpecification,
)

# Recorded before the move (see the module docstring).
GOLDEN_DIGEST_A = "sha256:932cadeeb821ee3416a6b1084baaab64c3a809ec29cf2329a36c1a0f204f6fa1"
GOLDEN_DIGEST_B = "sha256:f558f9c416865d44c8409645d158dcfe2fca57c77900c853d576a99afc347955"


def _dataset_a() -> EvaluationDataset:
    return EvaluationDataset(
        "golden-a",
        "v1",
        [
            EvaluationCase("c1", {"query": "q1"}, expected="x"),
            EvaluationCase(
                "c2",
                {"query": "q2", "tool_context": {"tenant_id": "t"}},
                expected=2.5,
                source_trace_id="tr-2",
                source_investigation_digest="sha256:aa",
            ),
            EvaluationCase("c3", {"query": "q3"}, expected=None),
        ],
    )


def _dataset_b() -> EvaluationDataset:
    return EvaluationDataset(
        "golden-b",
        "2026-01-01",
        [
            EvaluationCase("k1", {"query": "a", "nested": {"n": [1, 2, 3]}}, expected={"rows": 3}),
            EvaluationCase("k2", {"query": "b"}, expected="y", source_trace_id="tr-9"),
        ],
    )


class _RecordingSink:
    def __init__(self) -> None:
        self.events: list[EvidenceEvent] = []

    def emit(self, event: EvidenceEvent) -> bool:
        self.events.append(event)
        return True


def _request(dataset: EvaluationDataset, candidate: EvaluationVariant) -> EvaluationRequest:
    return EvaluationRequest(
        evaluation_id="ev-1",
        evidence_context=EvidenceContext(
            agent_id="agent", deployment_digest="sha256:dep", evaluation_id="ev-1", dataset_version=dataset.version
        ),
        dataset=dataset,
        baseline=EvaluationVariant("base"),
        candidate=candidate,
    )


def _run_one(case: EvaluationCase, variant: EvaluationVariant) -> dict[str, int]:
    if case.case_id == "c3" and variant.variant_id == "cand":
        raise RuntimeError("boom")
    return {"score": len(case.inputs["query"]) + (1 if variant.advisory_skill else 0)}


async def _metric(case: EvaluationCase, output: dict[str, int]) -> dict[str, float]:
    return {"quality": float(output["score"]), "cost": 0.5 + len(case.case_id) / 10}


def test_the_package_imports_no_agent_framework_and_not_the_learning_control_plane() -> None:
    """Checked in a fresh interpreter, so modules other tests already loaded cannot hide a leak."""

    program = textwrap.dedent(
        """
        import sys
        import agent_evals  # noqa: F401
        forbidden = ("learning_control_plane", "penguiflow", "langchain", "langchain_core", "mlflow", "litellm")
        loaded = sorted({name.split(".")[0] for name in sys.modules} & set(forbidden))
        print(",".join(loaded))
        """
    )

    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, check=True)

    assert result.stdout.strip() == ""


def test_the_learning_control_plane_reexports_the_very_same_classes() -> None:
    pytest.importorskip("learning_control_plane", reason="this re-export check needs the monorepo")
    from learning_control_plane import evaluation as lcp_evaluation
    from learning_control_plane.contracts import evidence as lcp_evidence
    from learning_control_plane.contracts import steps as lcp_steps

    for name in ("EvaluationCase", "EvaluationDataset", "EvaluationVariant", "LocalEvaluationBackend"):
        assert getattr(lcp_evaluation, name) is getattr(agent_evals, name), name
    for name in ("MetricSpecification", "PairedEvaluationResult", "VariantCaseResult"):
        assert getattr(lcp_evaluation, name) is getattr(agent_evals, name), name
    assert lcp_evidence.EvidenceContext is agent_evals.EvidenceContext
    assert lcp_steps.GenericStep is agent_evals.GenericStep
    assert lcp_steps.GenericTrajectory is agent_evals.GenericTrajectory


def test_dataset_digests_are_the_ones_recorded_before_the_move() -> None:
    assert _dataset_a().manifest_digest == GOLDEN_DIGEST_A
    assert _dataset_b().manifest_digest == GOLDEN_DIGEST_B


async def test_a_paired_run_gives_the_numbers_recorded_before_the_move() -> None:
    dataset = _dataset_a()
    request = _request(dataset, EvaluationVariant("cand", advisory_skill="Be brief."))

    result = await LocalEvaluationBackend().evaluate(request, _run_one, _metric)

    assert result.mean_metrics("base") == {"cost": 0.6999999999999998, "quality": 2.0}
    assert result.mean_metrics("cand") == {"cost": 0.7, "quality": 3.0}
    assert result.failed_case_ids("cand") == ("c3",)
    assert result.incomplete_case_ids == ("c3",)
    assert result.metric_summary(MetricSpecification("quality")).mean_improvement == 1.0


async def test_the_events_a_run_emits_are_the_ones_recorded_before_the_move() -> None:
    dataset = _dataset_a()
    sink = _RecordingSink()
    request = _request(dataset, EvaluationVariant("cand", advisory_skill="Be brief."))

    await LocalEvaluationBackend(evidence_sink=sink).evaluate(request, _run_one, _metric)

    started, completed = (event.record() for event in sink.events)
    assert started["event_type"] == "evaluation.started"
    assert started["attributes"] == {"baseline_variant_id": "base", "case_count": 3, "dataset_digest": GOLDEN_DIGEST_A}
    assert started["candidate_id"] == "cand"
    assert completed["event_type"] == "evaluation.completed"
    assert completed["attributes"] == {
        "baseline_failed_case_count": 0,
        "baseline_failed_case_ids": [],
        "baseline_variant_id": "base",
        "candidate_failed_case_count": 1,
        "candidate_failed_case_ids": ["c3"],
        "case_count": 3,
        "complete_pair_count": 2,
        "dataset_digest": GOLDEN_DIGEST_A,
        "expected_pair_count": 3,
        "failed_pair_count": 1,
        "failed_pair_ids": ["c3"],
    }
    assert completed["metrics"] == {
        "baseline.cost": 0.6999999999999998,
        "baseline.quality": 2.0,
        "candidate.cost": 0.7,
        "candidate.quality": 3.0,
    }
    assert set(completed) == {
        "agent_id",
        "attributes",
        "candidate_id",
        "dataset_version",
        "deployment_digest",
        "evaluation_id",
        "event_id",
        "event_type",
        "metric_version",
        "metrics",
        "occurred_at",
        "policy_version",
        "scope_ref",
        "trace_id",
    }


def test_a_general_evaluation_needs_no_advisory_skill_but_a_skill_evaluation_does() -> None:
    """The skill rule belongs to the learning control plane; the general request has no such rule."""

    pytest.importorskip("learning_control_plane", reason="this check needs the monorepo")
    from learning_control_plane.evaluation import EvaluationRequest as SkillEvaluationRequest

    dataset = _dataset_a()
    without_skill = EvaluationVariant("cand")

    general = _request(dataset, without_skill)

    assert general.candidate.advisory_skill is None
    with pytest.raises(ValueError, match="candidate must include an advisory skill"):
        SkillEvaluationRequest(
            evaluation_id=general.evaluation_id,
            evidence_context=general.evidence_context,
            dataset=dataset,
            baseline=general.baseline,
            candidate=without_skill,
        )
    with pytest.raises(ValueError, match="baseline must not include an advisory skill"):
        SkillEvaluationRequest(
            evaluation_id=general.evaluation_id,
            evidence_context=general.evidence_context,
            dataset=dataset,
            baseline=EvaluationVariant("base", advisory_skill="x"),
            candidate=EvaluationVariant("cand", advisory_skill="y"),
        )


async def test_an_event_the_neutral_backend_emits_can_be_written_by_the_lcp_sinks() -> None:
    """The MLflow and OpenTelemetry sinks stay in the LCP; they must accept the neutral event."""

    pytest.importorskip("learning_control_plane", reason="this check needs the monorepo")
    from learning_control_plane.contracts.evidence import MlflowEvidenceSink

    class FakeMlflow:
        def __init__(self) -> None:
            self.tags: dict[str, str] = {}
            self.artifact_paths: list[str] = []

        def active_run(self) -> object:
            return object()

        def set_tags(self, tags: dict[str, str]) -> None:
            self.tags.update(tags)

        def log_dict(self, record: dict[str, object], path: str) -> None:
            self.artifact_paths.append(path)

    fake = FakeMlflow()
    sink = MlflowEvidenceSink(mlflow_module=fake)
    request = _request(_dataset_a(), EvaluationVariant("cand", advisory_skill="Be brief."))

    await LocalEvaluationBackend(evidence_sink=sink).evaluate(request, _run_one, _metric)

    assert fake.tags["lcp.event_type"] == "evaluation.completed"
    assert fake.tags["lcp.evaluation_id"] == "ev-1"
    assert len(fake.artifact_paths) == 2
