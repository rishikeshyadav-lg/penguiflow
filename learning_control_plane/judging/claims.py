"""Catch an answer that compares a metric with a "typical" or "standard" level no tool returned.

"The click rate of 0.04% is in the typical range" asserts a benchmark. When no benchmark tool
returned data, that claim is invented. A meaning-check model scored such sentences right at its bar,
so they flipped between reads and some verified; this deterministic check fails them every time.
A sentence counts only when it pairs the comparison wording with one of the integration's metric
names, so "typical for a store winding down" is not flagged.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from .runs import AgentRun

UNSUPPORTED_BENCHMARK_CLAIM = "unsupported_benchmark_claim"
BENCHMARK_WORDING = re.compile(
    r"\b(?:typical(?:ly)? (?:range|for)|normal (?:for|range)"
    r"|industry(?:[- ]wide)? (?:average|benchmarks?|standards?|norms?)"
    r"|(?:standard|typical|industry) benchmarks?"
    r"|(?:above|below|in line with|consistent with|within) (?:the )?(?:typical|standard|industry|expected)(?: \w+)? "
    r"(?:range|benchmarks?|thresholds?|levels?|averages?))\b",
    re.IGNORECASE,
)
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(frozen=True, slots=True)
class BenchmarkClaimRule:
    """Which metric names make a comparison a benchmark claim, and which tools can support one."""

    metric_names: re.Pattern[str]
    benchmark_tools: frozenset[str]
    wording: re.Pattern[str] = BENCHMARK_WORDING


def states_unsupported_benchmark(run: AgentRun, rule: BenchmarkClaimRule) -> bool:
    """Return whether a sentence compares a metric with a typical or standard level no benchmark tool returned."""

    has_benchmark_data = any(
        step.tool in rule.benchmark_tools and isinstance(step.result, Mapping) and step.result.get("rows")
        for step in run.steps
    )
    if has_benchmark_data:
        return False
    return any(
        rule.wording.search(sentence) and rule.metric_names.search(sentence)
        for sentence in _SENTENCE_BREAK.split(run.final_answer or "")
    )


__all__ = ["BENCHMARK_WORDING", "BenchmarkClaimRule", "UNSUPPORTED_BENCHMARK_CLAIM", "states_unsupported_benchmark"]
