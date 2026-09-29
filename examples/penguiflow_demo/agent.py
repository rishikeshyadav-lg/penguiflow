"""A toy PenguiFlow agent for the Learning Control Plane's live PenguiFlow demo.

The PenguiFlow counterpart of `examples/langchain_demo/agent.py`: the same toy
campaign-analytics domain (one tool, `lookup_campaign_clicks`) so the two live demos are directly
comparable, but driven by a real `penguiflow.planner.ReactPlanner` instead of a hand-rolled loop.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from penguiflow.catalog import NodeSpec, build_catalog, tool
from penguiflow.node import Node
from penguiflow.registry import ModelRegistry

_CAMPAIGN_CLICKS = {"112774": 400, "112775": 128, "112776": 950}


class CampaignQuery(BaseModel):
    campaign_id: str


class ClickCount(BaseModel):
    clicks: int


@tool(desc="Look up how many clicks a campaign id received.", side_effects="read")
async def lookup_campaign_clicks(args: CampaignQuery, ctx: object) -> ClickCount:
    """Return the click count for a known campaign id, or 0 for an unknown one."""

    return ClickCount(clicks=_CAMPAIGN_CLICKS.get(args.campaign_id, 0))


def campaign_agent_catalog() -> Sequence[NodeSpec]:
    """Build the one-tool catalog `ReactPlanner` needs to run this toy agent."""

    registry = ModelRegistry()
    registry.register("lookup_campaign_clicks", CampaignQuery, ClickCount)
    nodes = [Node(lookup_campaign_clicks, name="lookup_campaign_clicks")]
    return build_catalog(nodes, registry)
