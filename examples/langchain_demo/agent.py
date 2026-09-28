"""A toy LangChain agent for the Learning Control Plane's plug-and-play demo.

Deliberately tiny and LLM-free (no API key needed): a real `langchain_core` tool, called by a
hand-rolled planner instead of an LLM, so the demo runs anywhere. What the rest of the demo cares
about is the *shape* of the result -- `{"input", "output", "intermediate_steps"}`, exactly what
`AgentExecutor.invoke()` returns -- not how the decision to call the tool was made.
"""

from __future__ import annotations

from langchain_core.agents import AgentAction
from langchain_core.tools import tool

_CAMPAIGN_CLICKS = {"112774": 400, "112775": 128, "112776": 950}


@tool
def lookup_campaign_clicks(campaign_id: str) -> int:
    """Look up how many clicks a campaign id received."""

    return _CAMPAIGN_CLICKS.get(campaign_id, 0)


def _campaign_id_in(query: str) -> str | None:
    return next((word for word in query.split() if word.isdigit()), None)


def run_campaign_agent(query: str, *, guidance: str | None = None) -> dict[str, object]:
    """Run the toy agent once, returning the shape `AgentExecutor.invoke()` would.

    Without `guidance` the agent hedges ("might have received around..."); with it, the agent
    states the number plainly. That behavior gap is the pattern the demo's mined advisory skill
    is meant to fix.
    """

    campaign_id = _campaign_id_in(query)
    if campaign_id is None:
        return {"input": query, "output": "I could not find a campaign id in that question.", "intermediate_steps": []}

    action = AgentAction(
        tool="lookup_campaign_clicks", tool_input={"campaign_id": campaign_id}, log="looking up clicks"
    )
    clicks = lookup_campaign_clicks.invoke({"campaign_id": campaign_id})
    answer = (
        f"{clicks} clicks."
        if guidance
        else f"Well, it looks like the campaign might have received around {clicks} clicks, though I'd double-check."
    )
    return {"input": query, "output": answer, "intermediate_steps": [(action, clicks)]}
