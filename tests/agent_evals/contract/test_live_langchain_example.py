"""The live LangChain example, driven by a scripted chat model so it needs no credentials."""

from __future__ import annotations

import importlib.util
import json
import re
from typing import Any

import pytest
from _paths import EXAMPLES

pytest.importorskip("langchain_core", reason="this example needs langchain; it is not a dependency")

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

EXAMPLE = EXAMPLES / "agent_evals_live_langchain" / "flow.py"


def _load():
    spec = importlib.util.spec_from_file_location("agent_evals_live_langchain_flow", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ScriptedModel(BaseChatModel):
    """Plays the agent's part: look the city up, format the number, answer; say unknown when it is not found."""

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedModel:
        return self

    def _generate(self, messages: list, stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        last = messages[-1]
        if isinstance(last, HumanMessage):
            city = re.search(r"of (\w+)\?", str(last.content)).group(1)
            reply = AIMessage(
                content="", tool_calls=[{"name": "lookup_population", "args": {"city": city}, "id": "c1"}]
            )
        elif isinstance(last, ToolMessage) and last.name == "lookup_population":
            if str(last.content) == "-1":
                reply = AIMessage(content="unknown")
            else:
                reply = AIMessage(
                    content="", tool_calls=[{"name": "format_number", "args": {"value": int(last.content)}, "id": "c2"}]
                )
        else:
            reply = AIMessage(content=str(last.content))
        return ChatResult(generations=[ChatGeneration(message=reply)])


async def test_the_live_example_runs_with_a_scripted_model_and_scores_every_case() -> None:
    markdown, run = await _load().main(model=ScriptedModel())

    assert len(run.rows) == 6 and all(row.result.error is None for row in run.rows)
    assert "| Task success rate | outcome | 1.000 rate |" in markdown
    assert "| Tool selection accuracy | trajectory | 1.000 score |" in markdown
    assert "| Argument correctness | trajectory | 1.000 score |" in markdown
    assert "| Cost per task | operational | not measured | - | - | the runner reported no cost |" in markdown


async def test_the_unknown_city_is_answered_with_one_tool_call_and_the_known_ones_with_two() -> None:
    _, run = await _load().main(model=ScriptedModel())

    tools = {row.case_id: [call["tool"] for call in row.tool_calls] for row in run.rows}

    assert tools["pop-atlantis"] == ["lookup_population"]
    assert tools["pop-paris"] == ["lookup_population", "format_number"]
    assert next(row for row in run.rows if row.case_id == "pop-atlantis").answer == "unknown"
    assert next(row for row in run.rows if row.case_id == "pop-paris").answer == "2,100,000"


async def test_the_runs_go_through_the_lcp_langchain_adapter_and_keep_their_arguments() -> None:
    _, run = await _load().main(model=ScriptedModel())

    lookup = next(row for row in run.rows if row.case_id == "pop-lyon").tool_calls[0]

    assert lookup["args"] == {"city": "Lyon"}
    assert json.dumps(lookup)  # JSON-safe, as the row file requires
