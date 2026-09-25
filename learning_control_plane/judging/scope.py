"""Decide whether an answer is about what its question named.

The scope rules that survived a week of judge mistakes:
- names are compared on letters and digits only, so "Brand_X (Q3)" and "brand x q3" match;
- a name may carry a suffix the answer drops (a date range, a version); an integration passes
  patterns for those suffixes and the name also counts without them;
- the question's own terms decide scope, and naming any one of them is enough: whether every named
  entity is covered is completeness, not scope;
- only when the question names nothing do the agent's own filters stand in, and then each of them
  must be named.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ..evaluation.verification import VerificationCheck

_NOT_LETTER_OR_DIGIT = re.compile(r"[^a-z0-9]+")


def squash(text: str) -> str:
    """Return the text's letters and digits only, case-folded, for comparing names."""

    return _NOT_LETTER_OR_DIGIT.sub("", text.casefold())


def name_forms(name: str, *, suffix_patterns: Sequence[re.Pattern[str]] = ()) -> tuple[str, ...]:
    """Return the squashed forms a name may appear in: whole, and without each droppable suffix."""

    forms = [squash(name)]
    for pattern in suffix_patterns:
        forms.append(squash(pattern.sub("", name).strip("_ ")))
    return tuple(dict.fromkeys(form for form in forms if form))


def entity_named(text: str, name: str, *, suffix_patterns: Sequence[re.Pattern[str]] = ()) -> bool:
    """Return whether the text names the entity, in any of its forms."""

    squashed_text = squash(text)
    return any(form in squashed_text for form in name_forms(name, suffix_patterns=suffix_patterns))


def scope_result(named: bool) -> VerificationCheck:
    """Return the scope check for an answer that does or does not name the requested scope."""

    if named:
        return VerificationCheck("scope_correctness", "passed", ("requested_scope_present",))
    return VerificationCheck("scope_correctness", "failed", ("requested_scope_missing",))


def question_scope_check(
    shown: str,
    *,
    question_terms: Sequence[str],
    fallback_terms: Sequence[str] = (),
    suffix_patterns: Sequence[re.Pattern[str]] = (),
) -> VerificationCheck:
    """Check the answer names what the question named; the agent's own filters stand in only when it named nothing.

    `shown` is everything the user saw (see `answer_text.shown_to_user`). `question_terms` are the
    identifiers and names read from the question itself; any one of them is enough. `fallback_terms`
    are the entities the agent filtered on; each of them must be named.
    """

    if question_terms:
        return scope_result(any(entity_named(shown, term, suffix_patterns=suffix_patterns) for term in question_terms))
    if not fallback_terms:
        return VerificationCheck("scope_correctness", "not_applicable", ("no_explicit_scope_term",))
    return scope_result(all(entity_named(shown, term, suffix_patterns=suffix_patterns) for term in fallback_terms))


__all__ = [
    "entity_named",
    "name_forms",
    "question_scope_check",
    "scope_result",
    "squash",
]
