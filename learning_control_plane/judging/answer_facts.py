"""Read the metric values an answer states, the way a reader sees them.

A judge that accepts any number anywhere in the answer as "the value of" a metric passes an answer
that puts the right number beside the wrong metric, and a judge that never looks at rendered tables
fails an answer whose numbers are only there. This module turns the answer text and every rendered
table or report into stated values: which metric, which value, and which row or section it belongs to.

Three layouts are read:
- tables (markdown in the text or a report section, and rendered tables): a metric column or a
  metric row gives the metric, the row label or column header gives what the value belongs to;
- "Metric: value" lines, for example "**Units sold:** 1,872";
- prose, where each number belongs to the metric named closest to it in the same sentence.

A value qualified as a single period or a change ("Latest units", "WoW change", "share") is not a
total and is skipped. Which words name which metric is the integration's `MetricVocabulary`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .runs import RenderedOutput

# Words that make a value something other than the metric's total.
_QUALIFIERS = re.compile(
    r"\b(latest|last week|this week|prior week|previous week|wow|week[- ]over[- ]week|change|delta|share|"
    r"%\s*of|vs\.?|versus|diff|growth|avg|average|per day|daily)\b|Δ",
    re.IGNORECASE,
)
_NUMBER = re.compile(
    r"(?P<currency>\$)?(?P<number>-?\d[\d,]*(?:\.\d+)?)"
    r"\s*(?:(?P<magnitude>thousand|million|billion|k|m|b)\b)?"
    r"\s*(?P<percent>%)?",
    re.IGNORECASE,
)
_MAGNITUDES = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "b": 1e9, "billion": 1e9}
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(?P<title>.+?)\s*#*\s*$")
_SEPARATOR_CELL = re.compile(r"^:?-{2,}:?$")
_PROSE_DISTANCE = 24
_GENERIC_VALUE_HEADERS = frozenset({"", "value", "values", "total", "totals", "overall", "total / overall", "amount"})
_NAME_PREFIX = re.compile(r"^\s*(?:[-*]\s+|\d+[.)]\s+)?\**(?P<name>[^:|*]{2,80}?)\**\s*[:–—]\s+")
_TOTAL_LABEL = re.compile(r"\b(total|overall|all|grand total|campaign total|sum)\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class MetricVocabulary:
    """Which words name which metric, and which metrics are rates.

    `aliases` maps each canonical metric to the names an answer uses for it; longer names win, so
    "device reach" is read before "reach". A rate written as a bare number ("0.15") may be a fraction
    or a percent, so both readings are kept.
    """

    aliases: Mapping[str, Sequence[str]]
    rate_metrics: frozenset[str] = frozenset()
    alias_pattern: re.Pattern[str] = field(init=False, repr=False, compare=False)
    _metric_by_alias: Mapping[str, str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        names = sorted(
            {re.escape(alias) for aliases in self.aliases.values() for alias in aliases},
            key=len,
            reverse=True,
        )
        if not names:
            raise ValueError("a metric vocabulary needs at least one alias")
        object.__setattr__(self, "alias_pattern", re.compile(r"\b(" + "|".join(names) + r")\b", re.IGNORECASE))
        object.__setattr__(
            self,
            "_metric_by_alias",
            {alias.casefold(): metric for metric, aliases in self.aliases.items() for alias in aliases},
        )

    def metric_for(self, alias: str) -> str:
        """Return the canonical metric an alias names."""

        return self._metric_by_alias[alias.casefold()]


@dataclass(frozen=True, slots=True)
class StatedValue:
    """One value the answer states for one metric, and what it belongs to (a row, section or heading).

    `row` marks a value from one row of a table (a group, a period, an entity) rather than a total or
    a labelled line. `explicit` is False when the metric was not named and was inferred from the only
    metric the question asks for; such a value can confirm an expectation but never contradict one.
    """

    metric: str
    value: float
    abbreviated: bool
    label: str
    source: str
    row: bool = False
    explicit: bool = True


def metric_named(text: str, vocabulary: MetricVocabulary) -> str | None:
    """Return the canonical metric a label or header names, or None when it names none or a qualified variant."""

    cleaned = _clean(text)
    if not cleaned or _QUALIFIERS.search(cleaned):
        return None
    match = vocabulary.alias_pattern.search(cleaned)
    return vocabulary.metric_for(match.group(1)) if match else None


def stated_values(
    answer: str,
    vocabulary: MetricVocabulary,
    rendered: Iterable[RenderedOutput] = (),
    *,
    implicit_metric: str | None = None,
) -> tuple[StatedValue, ...]:
    """Return every metric value in the answer text and in each rendered table or report.

    `implicit_metric` is the only metric the question asks for; a number in prose with no metric named
    near it is then read as that metric (marked not explicit).
    """

    values = _values_in_markdown(answer or "", vocabulary, heading="", source="text", implicit_metric=implicit_metric)
    for output in rendered:
        if output.kind == "table":
            values.extend(_values_in_rendered_table(output.content, vocabulary))
        elif output.kind == "report":
            values.extend(_values_in_report(output.content, vocabulary))
    return tuple(values)


def values_in_text(
    text: str, vocabulary: MetricVocabulary, *, label: str, implicit_metric: str | None = None
) -> list[StatedValue]:
    """Return the values in a stretch of answer text, all labelled with `label` (text about one entity)."""

    return _values_in_markdown(text, vocabulary, heading=label, source="anchored", implicit_metric=implicit_metric)


def _values_in_markdown(
    text: str,
    vocabulary: MetricVocabulary,
    *,
    heading: str,
    source: str,
    implicit_metric: str | None = None,
) -> list[StatedValue]:
    values: list[StatedValue] = []
    table: list[list[str]] = []
    for line in [*text.splitlines(), ""]:
        if line.strip().startswith("|"):
            table.append(_cells(line))
            continue
        if table:
            values.extend(_values_in_table(table, vocabulary, heading=heading, source=f"{source}_table"))
            table = []
        heading_match = _HEADING.match(line)
        if heading_match:
            heading = _clean(heading_match.group("title"))
            continue
        values.extend(
            _values_in_line(line, vocabulary, heading=heading, source=source, implicit_metric=implicit_metric)
        )
    return values


def _cells(line: str) -> list[str]:
    inner = line.strip().strip("|")
    return [cell.strip() for cell in inner.split("|")]


def _values_in_table(
    rows: list[list[str]], vocabulary: MetricVocabulary, *, heading: str, source: str
) -> list[StatedValue]:
    body = [row for row in rows if not all(_SEPARATOR_CELL.match(cell.replace(" ", "")) for cell in row if cell)]
    if len(body) < 2:
        return []
    header, data = body[0], body[1:]
    column_metrics = [metric_named(cell, vocabulary) for cell in header]
    values: list[StatedValue] = []
    if any(column_metrics):
        # Wide table: one row per group, one column per metric.
        label_column = next((index for index, metric in enumerate(column_metrics) if metric is None), None)
        for row in data:
            label = _clean(row[label_column]) if label_column is not None and label_column < len(row) else heading
            for index, metric in enumerate(column_metrics):
                if metric and index < len(row):
                    values.extend(
                        _parsed(metric, row[index], vocabulary, label=label, source=source, row=_is_row_label(label))
                    )
        return values
    # Tall table: one row per metric, one column per group (or a single value column).
    for row in data:
        metric = metric_named(row[0], vocabulary) if row else None
        if metric is None:
            continue
        for index in range(1, len(row)):
            column_header = _clean(header[index]) if index < len(header) else ""
            label = heading if column_header.casefold() in _GENERIC_VALUE_HEADERS else column_header
            values.extend(_parsed(metric, row[index], vocabulary, label=label, source=source))
    return values


def _values_in_line(
    line: str,
    vocabulary: MetricVocabulary,
    *,
    heading: str,
    source: str,
    implicit_metric: str | None = None,
) -> list[StatedValue]:
    """Return the values in one line of prose: each number belongs to the closest metric name in its sentence.

    A line that starts with a name ("North Store: units 1,500") is labelled by that name, otherwise
    by the heading above it.
    """

    values: list[StatedValue] = []
    line_label = heading
    for position, raw_sentence in enumerate(re.split(r"(?<=[.!?])\s+|\s+\|\s+|;", line)):
        sentence = raw_sentence
        label = line_label
        prefix = _NAME_PREFIX.match(sentence)
        if prefix and not vocabulary.alias_pattern.search(prefix.group("name")):
            label = _clean(prefix.group("name"))
            sentence = sentence[prefix.end() :]
            if position == 0:
                line_label = label
        mentions = [
            (match.start(), match.end(), vocabulary.metric_for(match.group(1)))
            for match in vocabulary.alias_pattern.finditer(sentence)
            if not _QUALIFIERS.search(sentence[max(0, match.start() - 12) : match.start()])
        ]
        if not mentions and implicit_metric is None:
            continue
        for number in _NUMBER.finditer(sentence):
            if _is_part_of_date_or_id(sentence, number):
                continue
            metric = _closest_metric(number, mentions)
            if metric is not None:
                values.extend(_parsed(metric, number.group(0), vocabulary, label=label, source=f"{source}_line"))
            elif implicit_metric is not None and not _QUALIFIERS.search(sentence):
                values.extend(
                    _parsed(
                        implicit_metric,
                        number.group(0),
                        vocabulary,
                        label=label,
                        source=f"{source}_line",
                        explicit=False,
                    )
                )
    return values


def _closest_metric(number: re.Match[str], mentions: Sequence[tuple[int, int, str]]) -> str | None:
    best: tuple[int, str] | None = None
    for start, end, metric in mentions:
        distance = number.start() - end if number.start() >= end else start - number.end()
        if 0 <= distance <= _PROSE_DISTANCE and (best is None or distance < best[0]):
            best = (distance, metric)
    return best[1] if best else None


def _is_part_of_date_or_id(text: str, number: re.Match[str]) -> bool:
    before = text[max(0, number.start() - 1) : number.start()]
    after = text[number.end() : number.end() + 1]
    return before in {"/", "-", "#"} or after in {"/"} or bool(re.match(r"\d{4}-\d{2}", text[number.start() :]))


def _values_in_rendered_table(content: Mapping[str, Any], vocabulary: MetricVocabulary) -> list[StatedValue]:
    columns = content.get("columns")
    rows = content.get("rows")
    if not isinstance(columns, Sequence) or not isinstance(rows, Sequence):
        return []
    named: list[tuple[str, str | None]] = []
    for column in columns:
        if not isinstance(column, Mapping):
            continue
        field_name = str(column.get("field") or "")
        header = str(column.get("header") or field_name)
        metric = metric_named(header, vocabulary)
        if metric is None and not _QUALIFIERS.search(header):
            metric = metric_named(field_name, vocabulary)
        named.append((field_name, metric))
    label_field = next((field_name for field_name, metric in named if metric is None), None)
    values: list[StatedValue] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        label = _clean(str(row.get(label_field, ""))) if label_field else ""
        for field_name, metric in named:
            if metric and field_name in row:
                values.extend(
                    _parsed(
                        metric,
                        row[field_name],
                        vocabulary,
                        label=label,
                        source="rendered_table",
                        row=_is_row_label(label),
                    )
                )
    return values


def _values_in_report(content: Mapping[str, Any], vocabulary: MetricVocabulary) -> list[StatedValue]:
    values: list[StatedValue] = []

    def visit(sections: Any) -> None:
        if not isinstance(sections, Sequence) or isinstance(sections, (str, bytes)):
            return
        for section in sections:
            if not isinstance(section, Mapping):
                continue
            title = _clean(str(section.get("title") or ""))
            section_text = section.get("content")
            if isinstance(section_text, str):
                values.extend(_values_in_markdown(section_text, vocabulary, heading=title, source="rendered_report"))
            visit(section.get("subsections"))

    visit(content.get("sections"))
    return values


def _parsed(
    metric: str,
    raw: Any,
    vocabulary: MetricVocabulary,
    *,
    label: str,
    source: str,
    row: bool = False,
    explicit: bool = True,
) -> list[StatedValue]:
    """Return the value(s) one cell or number stands for; a bare rate may be a fraction or a percent."""

    def value(number: float, abbreviated: bool = False) -> StatedValue:
        return StatedValue(metric, number, abbreviated, label, source, row=row, explicit=explicit)

    is_rate = metric in vocabulary.rate_metrics
    if isinstance(raw, bool):
        return []
    if isinstance(raw, (int, float)):
        number = float(raw)
        return [value(number), value(number / 100)] if is_rate else [value(number)]
    match = _NUMBER.search(str(raw))
    if match is None:
        return []
    number = float(match.group("number").replace(",", ""))
    magnitude = match.group("magnitude")
    if match.group("percent"):
        return [value(number / 100)] if is_rate else []
    if magnitude:
        return [value(number * _MAGNITUDES[magnitude.casefold()], abbreviated=True)]
    return [value(number), value(number / 100)] if is_rate else [value(number)]


def _is_row_label(label: str) -> bool:
    """Return whether a table row stands for one group rather than the total line."""

    return not _TOTAL_LABEL.search(label)


def _clean(text: str) -> str:
    return re.sub(r"[*`]+", "", str(text)).replace("_", " ").strip().strip(":").strip()


__all__ = ["MetricVocabulary", "StatedValue", "metric_named", "stated_values", "values_in_text"]
