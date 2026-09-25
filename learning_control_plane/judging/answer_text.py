"""Read an answer the way its user saw it, before any value in it is checked.

Three mistakes a judge made before these helpers existed:
- an answer cut off mid-sentence was scored on the numbers it reached, instead of as an agent error;
- a table or report the agent rendered beside its text was ignored, so values shown only there
  counted as missing;
- the "... [N more items]" note a planner puts in place of a long list's tail was read as one more
  (malformed) item.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence

from .runs import AgentRun

# A planner keeps the first items of a long list and replaces the rest with this note.
SHORTENED_LIST_NOTE = re.compile(r"\.\.\. \[\d+ more items\]")
_ENDS_A_THOUGHT = re.compile(r"[.!?:)\]*`|%\"'”’>]$|https?://\S+$")
_STRUCTURED_LINE = re.compile(r"^([|#>*-]|\d+[.)]\s)")
_CHOOSE_PATTERN = re.compile(
    r"\b(which (one|campaign|of these)|choose|select one|reply with|let me know which|do you mean)\b",
    re.IGNORECASE,
)


def looks_truncated(answer: str) -> bool:
    """Return whether the answer stops mid-sentence: its last line is prose with no closing punctuation."""

    lines = [line.strip() for line in answer.strip().splitlines() if line.strip()]
    if not lines:
        return True
    last = lines[-1]
    if _STRUCTURED_LINE.match(last):
        return False
    return not _ENDS_A_THOUGHT.search(last)


def asks_user_to_choose(answer: str, candidates: Sequence[str], *, is_named: Callable[[str, str], bool]) -> bool:
    """Return whether the answer asks the user to pick among two or more candidates it names.

    `is_named(answer, candidate)` decides whether the answer names one candidate; an integration
    passes its own naming rule, for example `scope.entity_named` with its name suffixes.
    """

    if not _CHOOSE_PATTERN.search(answer):
        return False
    named = [candidate for candidate in candidates if is_named(answer, candidate)]
    return len(named) >= 2


def is_shortened_list_note(value: object) -> bool:
    """Return whether a list item is the planner's "... [N more items]" note rather than an item."""

    return isinstance(value, str) and SHORTENED_LIST_NOTE.fullmatch(value) is not None


def strip_shortened_list_note(items: Sequence[object]) -> list[object]:
    """Return the list's real items, without the planner's "... [N more items]" note."""

    return [item for item in items if not is_shortened_list_note(item)]


def shown_to_user(run: AgentRun) -> str:
    """Return the answer as the user saw it: its text, then each rendered table's rows and each report section."""

    parts = [run.final_answer or ""]
    for output in run.rendered:
        content = output.content
        if output.kind == "table":
            columns = [column for column in content.get("columns") or () if isinstance(column, Mapping)]
            headers = [str(column.get("header") or column.get("field")) for column in columns]
            fields = [str(column.get("field")) for column in columns]
            rows = [
                " | ".join(str(row.get(name, "")) for name in fields)
                for row in content.get("rows") or ()
                if isinstance(row, Mapping)
            ]
            parts.append("[Rendered table]\n" + " | ".join(headers) + "\n" + "\n".join(rows))
        elif output.kind == "report":
            sections = [
                f"## {section.get('title') or ''}\n{section.get('content') or ''}"
                for section in content.get("sections") or ()
                if isinstance(section, Mapping)
            ]
            parts.append("[Rendered report]\n" + "\n\n".join(sections))
    return "\n\n".join(parts)


def rendered_text(run: AgentRun) -> str:
    """Return every rendered table and report flattened to text, for finding the names they show."""

    return " ".join(str(output.content) for output in run.rendered)


__all__ = [
    "SHORTENED_LIST_NOTE",
    "asks_user_to_choose",
    "is_shortened_list_note",
    "looks_truncated",
    "rendered_text",
    "shown_to_user",
    "strip_shortened_list_note",
]
