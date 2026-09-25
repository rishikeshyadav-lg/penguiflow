"""Publish a PenguiFlow run's investigation when its turn ends, not when its trajectory completes.

A planner reports a completed trajectory before the turn is over: the answer the user receives may
still be completed or rewritten, and the MLflow trace is still being exported. Judging then scored
the wrong answer, and logging the assessment then raced the trace. `TurnStash` holds each completed
trajectory by its trace id until the turn has its final answer, then judges and publishes it once
through a `RunPublisher`. `PlannerTraceReadiness` lets the assessment publisher wait for the
planner's own trace persistence before it polls MLflow.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import replace
from time import monotonic
from typing import Any

from penguiflow.planner.trajectory import Trajectory

from ...providers.assessment_publisher import TraceReadiness
from ..generic import RunContext, RunPublication, RunPublisher
from .projector import agent_run_from_trajectory, expand_parallel_steps

logger = logging.getLogger("learning_control_plane.penguiflow")


class TurnStash:
    """Hold each completed trajectory until its turn has its final answer, then publish it once."""

    def __init__(
        self,
        publisher: RunPublisher,
        context_for: Callable[[Trajectory], RunContext | None],
        *,
        max_held: int = 1_000,
    ) -> None:
        if max_held < 1:
            raise ValueError("max_held must be at least 1")
        self._publisher = publisher
        self._context_for = context_for
        self._max_held = max_held
        self._held: OrderedDict[str, Trajectory] = OrderedDict()

    def hold(self, trajectory: Trajectory) -> bool:
        """Keep a completed trajectory under its trace id; return False when it has none to wait on."""

        trace_id = str((trajectory.tool_context or {}).get("trace_id") or "").strip()
        if not trace_id:
            return False
        self._held[trace_id] = trajectory
        self._held.move_to_end(trace_id)
        while len(self._held) > self._max_held:
            dropped, _ = self._held.popitem(last=False)
            # A turn that never ended would otherwise hold its trajectory forever.
            logger.warning("lcp_turn_stash_full dropped_trace_id=%s", dropped)
        return True

    def discard(self, trace_id: str) -> None:
        """Forget a held trajectory whose turn failed or was cancelled."""

        self._held.pop(str(trace_id), None)

    async def release(self, trace_id: str, final_answer: str | None) -> RunPublication | None:
        """Judge and publish the held trajectory with the answer the user received; never raises."""

        trajectory = self._held.pop(str(trace_id), None)
        if trajectory is None:
            return None
        try:
            context = self._context_for(trajectory)
            if context is None:
                return None
            run = agent_run_from_trajectory(replace(trajectory, steps=expand_parallel_steps(trajectory.steps)))
            if final_answer:
                run = replace(run, final_answer=final_answer)
            return await self._publisher.publish_after_turn(run, context)
        except Exception:  # noqa: BLE001 -- publishing is evidence collection, never part of the answer
            logger.warning("lcp_turn_stash_publish_failed trace_id=%s", trace_id, exc_info=True)
            return None


class PlannerTraceReadiness:
    """Wait for a planner's own trace persistence, then for the next readiness check if one is given.

    The planner's wait runs on the planner's event loop, so a publisher thread can call this safely.
    `native_trace_id` maps the MLflow trace id being assessed to the planner's trace id.
    """

    def __init__(
        self,
        planner: Any,
        loop: asyncio.AbstractEventLoop,
        *,
        then: TraceReadiness | None = None,
        native_trace_id: Callable[[str], str] = str,
    ) -> None:
        self._planner = planner
        self._loop = loop
        self._then = then
        self._native_trace_id = native_trace_id

    def wait_closed(self, trace_id: str, timeout_s: float) -> bool:
        """Return whether the planner persisted the trace (and the next check passed) within `timeout_s`."""

        started = monotonic()
        waiting = self._planner.wait_for_trace_persistence(self._native_trace_id(trace_id), timeout_s=timeout_s)
        try:
            persisted = asyncio.run_coroutine_threadsafe(waiting, self._loop).result(timeout=timeout_s + 1.0)
        except Exception:  # noqa: BLE001 -- not persisted in time; the publisher falls back or queues
            logger.info("planner trace persistence wait failed for %s", trace_id, exc_info=True)
            return False
        if not persisted:
            return False
        if self._then is None:
            return True
        remaining = max(0.0, timeout_s - (monotonic() - started))
        return self._then.wait_closed(trace_id, remaining)


__all__ = ["PlannerTraceReadiness", "TurnStash"]
