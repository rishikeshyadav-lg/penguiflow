"""The step signature a run is mined under: the work it did, not every call it made.

A mined pattern key is the question's class plus the step signature. With every projected step in
the signature, two runs of the same kind of question only matched when they made exactly the same
calls in the same order: one extra lookup, one failed probe or a different render tool gave a new
key, and no pattern ever reached the repeat count mining needs.

`SignatureRules` keeps the calls that do the work, in order: failed and recovered calls are left
out, presentation and schema calls the integration names are left out, calls that do the same job
are given one name, and the same call repeated back to back counts once.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .steps import RECOVERED_STEP_CODE


class SignatureNormalizer(Protocol):
    """Turn projected steps into the node names that make up a run's step signature."""

    def __call__(self, steps: Sequence[Mapping[str, Any]]) -> Sequence[str]:
        """Return the signature's node names for these projected steps."""
        ...


@dataclass(frozen=True, slots=True)
class SignatureRules:
    """An integration's rule for which projected steps make up a step signature.

    Usable directly as a `SignatureNormalizer`: `rules(steps)` returns the signature's node names.
    """

    left_out: frozenset[str] = frozenset()
    left_out_prefixes: tuple[str, ...] = ()
    renamed: Mapping[str, str] = field(default_factory=dict)
    collapse_repeats: bool = True
    failure_codes: frozenset[str] = frozenset({"tool_returned_error", RECOVERED_STEP_CODE})

    def __call__(self, steps: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
        """Return the node names of one run's work, for its step signature."""

        return normalized_signature(steps, self)


def normalized_signature(steps: Sequence[Mapping[str, Any]], rules: SignatureRules) -> tuple[str, ...]:
    """Return the node names that make up a run's step signature under these rules."""

    signature: list[str] = []
    for step in steps:
        node = str(step.get("node", ""))
        if _failed(step, rules.failure_codes) or node in rules.left_out or node.startswith(rules.left_out_prefixes):
            continue
        name = rules.renamed.get(node, node)
        if rules.collapse_repeats and signature and signature[-1] == name:
            continue
        signature.append(name)
    return tuple(signature)


def _failed(step: Mapping[str, Any], failure_codes: frozenset[str]) -> bool:
    """Return whether a projected step failed, or its tool check says the agent only recovered from it."""

    if step.get("status") == "failed":
        return True
    for check in step.get("result_checks") or ():
        if isinstance(check, Mapping) and check.get("check_id") == "tool_execution":
            if failure_codes & set(check.get("reason_codes") or ()):
                return True
    return False


__all__ = ["SignatureNormalizer", "SignatureRules", "normalized_signature"]
