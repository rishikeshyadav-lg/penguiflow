"""A synthetic framework adapter used only to prove the framework-adapter contract is generic."""

from .adapter import (
    MockEvent,
    MockFrameworkAdapter,
    MockGuidanceHook,
    MockInvestigationContext,
    MockRun,
    MockSkillStore,
)

__all__ = [
    "MockEvent",
    "MockFrameworkAdapter",
    "MockGuidanceHook",
    "MockInvestigationContext",
    "MockRun",
    "MockSkillStore",
]
