"""A LangChain adapter: the second real framework proving the contract is not PenguiFlow-only.

Reads either shape a real LangChain agent returns, and nothing else from LangChain: the legacy
`AgentExecutor.invoke()` shape (`{"input", "output", "intermediate_steps"}`, with
`intermediate_steps` a list of `(AgentAction, observation)` pairs), and current
`langchain.agents.create_agent(...).invoke()` (`{"messages": [...]}`, a list of
`HumanMessage`/`AIMessage`/`ToolMessage` objects, with each `AIMessage.tool_calls` entry paired to
the `ToolMessage` sharing its `tool_call_id`). `to_generic_trajectory`/`project` take the mapping
directly, so a real agent's result plugs in with no wrapper type, whichever shape it returns.

`attach_guidance`'s injection mechanism is LangChain's own idiom: a small callable a chain composes
(`prompt | guidance_injector | llm`), not a callback registered on a planner as PenguiFlow's is --
each framework decides how a hook actually runs; the protocol only promises that `attach_guidance`
returns something.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any
from uuid import uuid4

from langchain_core.agents import AgentAction
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from ...contracts.investigation import InvestigationTrajectoryV1, SourceTraceRef
from ...contracts.steps import GenericStep, GenericTrajectory
from ...control_plane.control_plane import ActivationReceipt, AdvisorySkillCandidate, DeliveryAuthorization

# Either shape a LangChain agent's `.invoke()` returns; accepted directly, never wrapped.
LangChainRun = Mapping[str, Any]


def _tool_args(tool_input: str | dict[Any, Any]) -> dict[str, Any]:
    return tool_input if isinstance(tool_input, dict) else {"input": tool_input}


@dataclass(frozen=True, slots=True)
class _LangChainStep:
    """One tool call, read from either native shape, before it becomes a `GenericStep` or an
    investigation step dict -- the one place both `to_generic_trajectory` and `project` share.
    """

    tool: str
    args: Mapping[str, Any]
    observation: Any
    failed: bool


def _steps_from_native_run(native_run: LangChainRun) -> tuple[str, list[_LangChainStep], str | None]:
    """Read either shape a LangChain agent might return, dispatching on which key is present."""

    if "messages" in native_run:
        return _steps_from_messages(native_run["messages"])
    return _steps_from_intermediate_steps(native_run)


def _steps_from_intermediate_steps(native_run: LangChainRun) -> tuple[str, list[_LangChainStep], str | None]:
    """Read the legacy `AgentExecutor.invoke()` shape."""

    steps = []
    intermediate_steps: Sequence[tuple[AgentAction, Any]] = native_run.get("intermediate_steps", ())
    for action, observation in intermediate_steps:
        failed = isinstance(observation, Exception)
        steps.append(
            _LangChainStep(
                tool=action.tool,
                args=_tool_args(action.tool_input),
                observation=str(observation) if failed else observation,
                failed=failed,
            )
        )
    return str(native_run.get("input", "")), steps, native_run.get("output")


def _steps_from_messages(messages: Sequence[BaseMessage]) -> tuple[str, list[_LangChainStep], str | None]:
    """Read `langchain.agents.create_agent(...).invoke()`'s shape: a list of messages instead of
    `intermediate_steps`. Each `AIMessage.tool_calls` entry is paired with the `ToolMessage`
    sharing its `tool_call_id`; the final answer is the last tool-call-free `AIMessage`.
    """

    query = ""
    for message in messages:
        if isinstance(message, HumanMessage):
            query = str(message.content)
            break

    tool_messages_by_call_id = {
        message.tool_call_id: message for message in messages if isinstance(message, ToolMessage)
    }

    steps: list[_LangChainStep] = []
    final_answer: str | None = None
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        if not message.tool_calls:
            final_answer = str(message.content)
            continue
        for call in message.tool_calls:
            call_id = call.get("id")
            observation_message = tool_messages_by_call_id.get(call_id) if call_id is not None else None
            failed = observation_message is not None and observation_message.status == "error"
            content = str(observation_message.content) if observation_message is not None else None
            steps.append(
                _LangChainStep(tool=call["name"], args=dict(call.get("args") or {}), observation=content, failed=failed)
            )

    return query, steps, final_answer


def to_generic_trajectory(native_run: LangChainRun) -> GenericTrajectory:
    """Translate one LangChain agent result into the framework-neutral step shape."""

    query, lc_steps, final_answer = _steps_from_native_run(native_run)
    return GenericTrajectory(
        query=query,
        steps=tuple(
            GenericStep(
                tool=step.tool,
                args=step.args,
                observation=None if step.failed else step.observation,
                error=step.observation if step.failed else None,
            )
            for step in lc_steps
        ),
        final_answer=final_answer,
    )


class LangChainInvestigationContext:
    """The LangChain adapter's equivalent of `PenguiFlowInvestigationContext`."""

    def __init__(
        self,
        *,
        agent_ref: str,
        scope_ref: str,
        execution_fingerprint: str,
        experiment_id: str = "langchain-experiment",
        provider_ref: str = "langchain",
        redaction_profile: str = "langchain-investigation-safe:v1",
    ) -> None:
        self.agent_ref = agent_ref
        self.scope_ref = scope_ref
        self.execution_fingerprint = execution_fingerprint
        self.experiment_id = experiment_id
        self.provider_ref = provider_ref
        self.redaction_profile = redaction_profile


class LangChainGuidanceInjector:
    """The object `LangChainFrameworkAdapter.attach_guidance` returns.

    Not a PenguiFlow-style `LLMContextHook`: this is meant to be composed into a LangChain chain
    (`prompt | injector.as_runnable(category) | llm`) or, as in the demo agent, called directly
    before the prompt is sent. Either way, the caller decides when it runs -- that decision is
    always the framework's, never the protocol's.
    """

    def __init__(self, *, guidance: str, categories: tuple[str, ...]) -> None:
        self.guidance = guidance
        self.categories = categories

    def apply(self, messages: Sequence[BaseMessage], *, category: str | None = None) -> list[BaseMessage]:
        """Append the guidance to the system message (or add one) when the category matches."""

        if self.categories and category not in self.categories:
            return list(messages)
        messages = list(messages)
        for index, message in enumerate(messages):
            if isinstance(message, SystemMessage):
                messages[index] = SystemMessage(content=f"{message.content}\nApproved guidance: {self.guidance}")
                return messages
        return [SystemMessage(content=f"Approved guidance: {self.guidance}"), *messages]


class LangChainSkillStore:
    """A tiny in-memory skill store, exactly as small as the mock's -- delivery needs no particular backend."""

    def __init__(self) -> None:
        self._skills: dict[str, dict[str, str]] = {}

    def put(self, *, scope_ref: str, candidate_id: str, guidance: str) -> str:
        digest = f"sha256:{sha256(guidance.encode()).hexdigest()}"
        self._skills[f"{scope_ref}:{candidate_id}"] = {"guidance": guidance, "digest": digest}
        return digest

    def get(self, *, scope_ref: str, candidate_id: str) -> dict[str, str] | None:
        return self._skills.get(f"{scope_ref}:{candidate_id}")


class LangChainFrameworkAdapter:
    """LangChain's implementation of `integrations.protocol.FrameworkAdapter`."""

    def __init__(self, context: LangChainInvestigationContext, skill_store: LangChainSkillStore | None = None) -> None:
        self._context = context
        self._skill_store = skill_store or LangChainSkillStore()

    def to_generic_trajectory(self, native_run: LangChainRun) -> GenericTrajectory:
        return to_generic_trajectory(native_run)

    def project(
        self,
        native_run: LangChainRun,
        *,
        completed_at: datetime | None = None,
        investigation_id: str | None = None,
    ) -> InvestigationTrajectoryV1:
        query, lc_steps, answer = _steps_from_native_run(native_run)
        steps = [
            {"index": index, "tool": step.tool, "status": "failed" if step.failed else "completed"}
            for index, step in enumerate(lc_steps)
        ]
        return InvestigationTrajectoryV1(
            investigation_id=investigation_id or f"investigation_{uuid4().hex[:24]}",
            source_trace_ref=SourceTraceRef(
                tracking_store_ref="langchain",
                experiment_id=self._context.experiment_id,
                mlflow_trace_id=f"langchain-run-{uuid4().hex[:12]}",
                deployment_ref=self._context.execution_fingerprint,
            ),
            agent_ref=self._context.agent_ref,
            provider_ref=self._context.provider_ref,
            scope_ref=self._context.scope_ref,
            started_at=datetime.now(UTC),
            completed_at=completed_at,
            status="completed" if answer else "failed",
            execution_fingerprint=self._context.execution_fingerprint,
            request={"has_text": bool(query)},
            steps=steps,
            redaction_profile=self._context.redaction_profile,
            step_signature=">".join(str(step["tool"]) for step in steps) or "no_steps",
            execution_context={
                "step_count": len(steps),
                "has_final_answer": answer is not None,
                "verified_success": answer is not None and all(step["status"] == "completed" for step in steps),
            },
        )

    def attach_guidance(self, *, guidance: str, categories: tuple[str, ...]) -> LangChainGuidanceInjector:
        return LangChainGuidanceInjector(guidance=guidance, categories=categories)

    def deliver(
        self,
        authorization: DeliveryAuthorization,
        candidate: AdvisorySkillCandidate,
        *,
        now: datetime | None = None,
    ) -> ActivationReceipt:
        delivered_at = now or datetime.now(UTC)
        if not authorization.is_active(delivered_at):
            raise ValueError("delivery authorization is expired or revoked")
        if authorization.candidate_id != candidate.candidate_id:
            raise ValueError("candidate does not match delivery authorization")

        digest = self._skill_store.put(
            scope_ref=authorization.scope_ref,
            candidate_id=candidate.candidate_id,
            guidance=candidate.advisory_skill,
        )
        return ActivationReceipt(
            receipt_id=f"receipt_{uuid4().hex}",
            authorization_id=authorization.authorization_id,
            candidate_id=candidate.candidate_id,
            scope_ref=authorization.scope_ref,
            provider_ref=f"langchain.skills:{authorization.scope_ref}:{candidate.candidate_id}",
            delivered_at=delivered_at,
            skill_digest=digest,
            investigation_digests=authorization.investigation_digests,
        )


__all__ = [
    "LangChainFrameworkAdapter",
    "LangChainGuidanceInjector",
    "LangChainInvestigationContext",
    "LangChainRun",
    "LangChainSkillStore",
    "to_generic_trajectory",
]
