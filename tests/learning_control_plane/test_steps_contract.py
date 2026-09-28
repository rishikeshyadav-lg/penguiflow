"""The framework-neutral step contract every verifier reads, whatever framework produced the run."""

from __future__ import annotations

import pytest

from learning_control_plane.contracts.steps import GenericStep, GenericTrajectory


def test_a_step_needs_a_tool_name() -> None:
    with pytest.raises(ValueError, match="tool"):
        GenericStep(tool="")
    with pytest.raises(ValueError, match="tool"):
        GenericStep(tool="   ")


def test_a_steps_args_and_failure_are_copied_into_plain_mappings() -> None:
    args = {"metric": "clicks"}
    failure = {"code": "no_data"}
    step = GenericStep(tool="aggregate_report", args=args, failure=failure)

    args["metric"] = "changed after construction"
    failure["code"] = "changed after construction"

    assert step.args == {"metric": "clicks"}
    assert step.failure == {"code": "no_data"}


def test_a_steps_failure_must_be_a_mapping_or_none() -> None:
    with pytest.raises(ValueError, match="failure"):
        GenericStep(tool="x", failure="not a mapping")  # type: ignore[arg-type]

    assert GenericStep(tool="x", failure=None).failure is None


def test_a_trajectorys_steps_and_context_are_immutable_and_copied() -> None:
    steps = [GenericStep(tool="a"), GenericStep(tool="b")]
    context = {"category": "summary"}
    trajectory = GenericTrajectory(query="q", steps=steps, llm_context=context)

    steps.append(GenericStep(tool="c"))
    context["category"] = "changed after construction"

    assert [step.tool for step in trajectory.steps] == ["a", "b"]
    assert trajectory.llm_context == {"category": "summary"}


def test_a_trajectory_defaults_to_no_steps_no_answer_and_no_context() -> None:
    trajectory = GenericTrajectory(query="q")

    assert trajectory.steps == ()
    assert trajectory.final_answer is None
    assert trajectory.llm_context == {}
