"""Exercises `examples/langchain_demo/flow.py`: the same learning loop as
`test_end_to_end.py`, run against a LangChain agent through `LangChainFrameworkAdapter` instead of
PenguiFlow's projector. Passing both proves one control plane instance serves either framework's
agent with no control-plane code change.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from examples.langchain_demo.flow import run_demo


@pytest.mark.asyncio
async def test_the_langchain_demo_mines_evaluates_gates_and_delivers_a_skill(tmp_path: Path) -> None:
    result = await run_demo(tmp_path / "langchain-demo")

    assert result["candidate_id"]
    assert result["job_id"]
    assert result["authorization_id"]
    assert result["receipt_id"]
    assert len(result["investigation_digests"]) == 7  # all mining + held-out traces the decision covers
    control_plane_db = result["control_plane_db"]
    assert isinstance(control_plane_db, str)
    assert Path(control_plane_db).exists()
