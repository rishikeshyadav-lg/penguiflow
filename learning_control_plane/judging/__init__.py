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
from .claims import BENCHMARK_WORDING, UNSUPPORTED_BENCHMARK_CLAIM, BenchmarkClaimRule, states_unsupported_benchmark
from .classification import ClassificationRead, freeze_classification, majority, mining_skip_reason
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
from .judge import (
    DomainJudge,
    OutcomeLadder,
    ReferenceBuilder,
    VerificationProjector,
    excluding_inapplicable_required_criteria,
    requiring_a_substantive_pass,
)
from .meaning import (
    DEFAULT_MEANING_QUESTIONS,
    FINDING_BAR,
    MeaningCheck,
    MeaningFindings,
    MeaningJudge,
    MeaningQuestion,
    tool_evidence,
)
from .no_data import (
    FINDINGS_ANSWERED_BY_NO_DATA,
    NO_DATA_PHRASES,
    EmptyLookupRule,
    no_data_confirmed_by_empty_lookups,
    no_data_judgment,
    nothing_fetched_judgment,
)
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
from .signature import SignatureNormalizer, SignatureRules, normalized_signature
from .steps import RECOVERED_STEP_CODE, recovered_step_indexes, step_failed

__all__ = [
    "AGENT_ERROR_CODE",
    "AgentRun",
    "AgentStep",
    "BENCHMARK_WORDING",
    "BenchmarkClaimRule",
    "ClassificationRead",
    "DEFAULT_MEANING_QUESTIONS",
    "DomainJudge",
    "EmptyLookupRule",
    "Expectation",
    "FINDINGS_ANSWERED_BY_NO_DATA",
    "FINDING_BAR",
    "HANDLED_CORRECTLY_CODES",
    "JUDGE_HARD_FAILURE_CODES",
    "JUDGE_OUTCOMES",
    "JudgeOutcome",
    "MeaningCheck",
    "MeaningFindings",
    "MeaningJudge",
    "MeaningQuestion",
    "MetricVocabulary",
    "NOT_JUDGEABLE_CODES",
    "NO_DATA_PHRASES",
    "OVERALL",
    "OutcomeLadder",
    "RECOVERED_STEP_CODE",
    "ReferenceBuilder",
    "RelativeTolerance",
    "RenderedKind",
    "RenderedOutput",
    "RubricJudgment",
    "SHORTENED_LIST_NOTE",
    "SignatureNormalizer",
    "SignatureRules",
    "StatedValue",
    "Tolerance",
    "UNSUPPORTED_BENCHMARK_CLAIM",
    "VerificationProjector",
    "all_criteria",
    "anchored_values",
    "asks_user_to_choose",
    "entity_named",
    "excluding_inapplicable_required_criteria",
    "freeze_classification",
    "is_shortened_list_note",
    "judge_expectation",
    "judge_rubric",
    "judgment_passes",
    "looks_truncated",
    "majority",
    "metric_named",
    "mining_skip_reason",
    "name_forms",
    "no_data_confirmed_by_empty_lookups",
    "no_data_judgment",
    "normalized_signature",
    "nothing_fetched_judgment",
    "outcome_of",
    "outcome_reason_codes",
    "question_scope_check",
    "recovered_step_indexes",
    "rendered_text",
    "requiring_a_substantive_pass",
    "row_is_named",
    "row_name_forms",
    "scope_result",
    "shown_to_user",
    "squash",
    "stated_values",
    "states_unsupported_benchmark",
    "step_failed",
    "strip_shortened_list_note",
    "tool_evidence",
    "values_in_text",
]
