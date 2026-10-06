"""Split one set of cases into disjoint parts, per group, with a seeded shuffle.

Lifted from the campaign's question-bank builder so that any agent can keep, say, a mining bank and an
evaluation bank apart: within each group (a question category, say) the cases are ordered by id, shuffled
with a generator seeded from the seed and the group, and cut. The same seed and groups always give the
same parts, and no case is in two of them.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Callable, Sequence

from .evaluation import EvaluationCase, EvaluationDataset


def split_by_group(
    cases: Sequence[EvaluationCase],
    group_of: Callable[[EvaluationCase], str],
    *,
    seed: int,
    first_share: float = 0.65,
) -> tuple[list[EvaluationCase], list[EvaluationCase]]:
    """Cut every group into a first part of about `first_share` and a second part with the rest.

    A group with one case sends it to the first part; with two or more, each part gets at least one.
    """

    if not 0 < first_share < 1:
        raise ValueError("first_share must be greater than 0 and less than 1")
    by_group: dict[str, list[EvaluationCase]] = defaultdict(list)
    for case in cases:
        by_group[group_of(case)].append(case)
    first: list[EvaluationCase] = []
    second: list[EvaluationCase] = []
    for group in sorted(by_group):
        members = sorted(by_group[group], key=lambda case: case.case_id)
        random.Random(f"{seed}:{group}").shuffle(members)
        count = len(members)
        to_first = math.ceil(first_share * count)
        if count >= 2:
            to_first = min(count - 1, max(1, to_first))
        first.extend(members[:to_first])
        second.extend(members[to_first:])
    return first, second


def assert_disjoint(*parts: EvaluationDataset | Sequence[EvaluationCase]) -> None:
    """Prove that no case id is in more than one part; raise naming the shared ids if one is."""

    seen: dict[str, int] = {}
    shared: set[str] = set()
    for index, part in enumerate(parts):
        cases = part.cases if isinstance(part, EvaluationDataset) else part
        for case in cases:
            if case.case_id in seen and seen[case.case_id] != index:
                shared.add(case.case_id)
            seen.setdefault(case.case_id, index)
    if shared:
        raise ValueError(f"parts share cases: {sorted(shared)}")


__all__ = ["assert_disjoint", "split_by_group"]
