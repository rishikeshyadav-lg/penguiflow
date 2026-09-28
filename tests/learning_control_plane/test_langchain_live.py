"""Exercises `examples/langchain_demo/live.py` against a real LangChain agent and a real,
Databricks-hosted model -- the live counterpart to `test_langchain_demo_end_to_end.py`'s offline,
LLM-free run. Opt-in and skipped by default: it costs money and needs network access and Databricks
credentials, so CI and a normal local run never trigger it.

    LCP_LIVE_LANGCHAIN=1 DATABRICKS_CONFIG_PROFILE=penguiflow-oauth uv run pytest -s \
        tests/learning_control_plane/test_langchain_live.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from examples.langchain_demo.live import run_demo

pytestmark = pytest.mark.skipif(
    os.environ.get("LCP_LIVE_LANGCHAIN") != "1",
    reason="live LangChain demo is opt-in: set LCP_LIVE_LANGCHAIN=1 to run it against a real model",
)


@pytest.mark.asyncio
async def test_the_live_langchain_agent_projects_evaluates_and_reaches_a_gate_decision(tmp_path: Path) -> None:
    """Asserts what a live model's behavior always guarantees, not that the gate approves --
    a live model's answers aren't pinned, so approval itself is reported, never asserted.
    """

    result = await run_demo(tmp_path / "langchain-live-demo")

    assert result["candidate_id"]
    assert result["job_id"]
    assert result["baseline_metrics"] is not None
    assert result["candidate_metrics"] is not None
    assert Path(str(result["control_plane_db"])).exists()

    print(f"\nlive LangChain demo result: {result}")
