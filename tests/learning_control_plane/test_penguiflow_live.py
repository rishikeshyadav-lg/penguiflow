"""Exercises `examples/penguiflow_demo/live.py` against a real PenguiFlow agent and a real,
Databricks-hosted model -- the live counterpart to `test_langchain_live.py`, and the first time
`PenguiFlowFrameworkAdapter` has ever seen a live model rather than a hand-built `Trajectory`.
Opt-in and skipped by default: it costs money and needs network access and Databricks credentials,
so CI and a normal local run never trigger it.

    LCP_LIVE_PENGUIFLOW=1 DATABRICKS_CONFIG_PROFILE=penguiflow-oauth uv run pytest -s \
        tests/learning_control_plane/test_penguiflow_live.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from examples.penguiflow_demo.live import run_demo

pytestmark = pytest.mark.skipif(
    os.environ.get("LCP_LIVE_PENGUIFLOW") != "1",
    reason="live PenguiFlow demo is opt-in: set LCP_LIVE_PENGUIFLOW=1 to run it against a real model",
)


@pytest.mark.asyncio
async def test_the_live_penguiflow_agent_projects_evaluates_and_reaches_a_gate_decision(tmp_path: Path) -> None:
    """Asserts what a live model's behavior always guarantees, not that the gate approves --
    a live model's answers aren't pinned, so approval itself is reported, never asserted.
    """

    result = await run_demo(tmp_path / "penguiflow-live-demo")

    assert result["candidate_id"]
    assert result["job_id"]
    assert result["baseline_metrics"] is not None
    assert result["candidate_metrics"] is not None
    assert Path(str(result["control_plane_db"])).exists()

    print(f"\nlive PenguiFlow demo result: {result}")
