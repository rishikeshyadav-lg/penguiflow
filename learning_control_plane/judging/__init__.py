"""The judge kit: framework-neutral pieces for judging an agent's answers before learning from them.

Each piece encodes a judge mistake an integration hit and fixed (see docs/judging/). An integration
supplies its domain parts (metric names, name suffixes, which tools fetch data) as data and small
callables; the kit supplies the rules.
"""

from .answer_facts import MetricVocabulary, StatedValue, metric_named, stated_values, values_in_text
from .answer_text import (
    SHORTENED_LIST_NOTE,
    asks_user_to_choose,
    is_shortened_list_note,
    looks_truncated,
    rendered_text,
    shown_to_user,
    strip_shortened_list_note,
)
from .expectations import (
    OVERALL,
    Expectation,
    RelativeTolerance,
    RubricJudgment,
    Tolerance,
    anchored_values,
    judge_expectation,
    judgment_passes,
    row_is_named,
    row_name_forms,
)
from .no_data import NO_DATA_PHRASES, no_data_judgment, nothing_fetched_judgment
from .outcomes import (
    AGENT_ERROR_CODE,
    HANDLED_CORRECTLY_CODES,
    JUDGE_HARD_FAILURE_CODES,
    JUDGE_OUTCOMES,
    NOT_JUDGEABLE_CODES,
    JudgeOutcome,
    all_criteria,
    judge_rubric,
    outcome_of,
    outcome_reason_codes,
)
from .runs import AgentRun, AgentStep, RenderedKind, RenderedOutput
from .scope import entity_named, name_forms, question_scope_check, scope_result, squash
from .signature import SignatureRules, normalized_signature
from .steps import RECOVERED_STEP_CODE, recovered_step_indexes, step_failed

__all__ = [
    "AGENT_ERROR_CODE",
    "AgentRun",
    "AgentStep",
    "Expectation",
    "HANDLED_CORRECTLY_CODES",
    "JUDGE_HARD_FAILURE_CODES",
    "JUDGE_OUTCOMES",
    "JudgeOutcome",
    "MetricVocabulary",
    "NOT_JUDGEABLE_CODES",
    "NO_DATA_PHRASES",
    "OVERALL",
    "RECOVERED_STEP_CODE",
    "RelativeTolerance",
    "RenderedKind",
    "RenderedOutput",
    "RubricJudgment",
    "SHORTENED_LIST_NOTE",
    "SignatureRules",
    "StatedValue",
    "Tolerance",
    "all_criteria",
    "anchored_values",
    "asks_user_to_choose",
    "entity_named",
    "is_shortened_list_note",
    "judge_expectation",
    "judge_rubric",
    "judgment_passes",
    "looks_truncated",
    "metric_named",
    "name_forms",
    "no_data_judgment",
    "normalized_signature",
    "nothing_fetched_judgment",
    "outcome_of",
    "outcome_reason_codes",
    "question_scope_check",
    "recovered_step_indexes",
    "rendered_text",
    "row_is_named",
    "row_name_forms",
    "scope_result",
    "shown_to_user",
    "squash",
    "stated_values",
    "step_failed",
    "strip_shortened_list_note",
    "values_in_text",
]
