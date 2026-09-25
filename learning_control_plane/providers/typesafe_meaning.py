"""TypeSafe (Jev) as a `MeaningJudge`, behind its data-governance gate.

The meaning check sends the question, the answer the user saw and the tool results to TypeSafe,
outside the deployment's own data platform. It may run only when a human has acknowledged that
(`TYPESAFE_GOVERNANCE_ACKNOWLEDGED`) and a key is configured; otherwise `from_environment` returns
None and no check runs. Jev returns a calibrated probability for each yes/no question, which is what
`MeaningCheck` averages across reads.

`typesafe_sdk` is an optional extra (`lcp-typesafe`) and is imported only when a read is made.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ..judging.meaning import MeaningQuestion

GOVERNANCE_ACKNOWLEDGED_ENV = "TYPESAFE_GOVERNANCE_ACKNOWLEDGED"
API_KEY_ENV = "TYPESAFE_API_KEY"
MODEL_ENV = "LCP_MEANING_JUDGE_MODEL"
DEFAULT_MODEL = "jev-latest"


def governance_acknowledged(environ: Mapping[str, str] | None = None) -> bool:
    """Return whether a human confirmed that answers and tool results may be sent to TypeSafe."""

    values = os.environ if environ is None else environ
    return values.get(GOVERNANCE_ACKNOWLEDGED_ENV, "").strip().lower() in {"1", "true", "yes"}


class TypeSafeMeaningJudge:
    """Ask Jev each meaning question and return its probability of "yes"."""

    def __init__(
        self,
        *,
        api_key: str,
        model_ref: str = DEFAULT_MODEL,
        system_one: Callable[..., Any] | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("TypeSafe API key must be configured")
        self.model_ref = model_ref
        self._api_key = api_key.strip()
        self._system_one = system_one

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        model_env: str = MODEL_ENV,
    ) -> TypeSafeMeaningJudge | None:
        """Return a judge when TypeSafe may be used (governance acknowledged and a key set), else None."""

        values = os.environ if environ is None else environ
        api_key = values.get(API_KEY_ENV, "").strip()
        if not governance_acknowledged(values) or not api_key:
            return None
        return cls(api_key=api_key, model_ref=values.get(model_env, DEFAULT_MODEL))

    def read(self, state: Mapping[str, str], questions: Sequence[MeaningQuestion]) -> dict[str, float]:
        """Return question name -> Jev's probability of "yes" for one read."""

        from typesafe_sdk import Noul, NoulCriteria

        noul_questions = {
            question.name: Noul(
                instructions=question.instruction, criteria=NoulCriteria(true=question.yes, false=question.no)
            )
            for question in questions
        }
        response = self._client_system_one()(state=dict(state), questions=noul_questions, model=self.model_ref)
        answers = getattr(response, "answers", None) or {}
        return {
            name: float(answer.noul)
            for name, answer in answers.items()
            if isinstance(getattr(answer, "noul", None), (int, float))
        }

    def _client_system_one(self) -> Callable[..., Any]:
        if self._system_one is None:
            from typesafe_sdk import RetryPolicy, TypeSafeClient

            self._system_one = TypeSafeClient(api_key=self._api_key, retry=RetryPolicy()).system_one
        return self._system_one


__all__ = [
    "API_KEY_ENV",
    "GOVERNANCE_ACKNOWLEDGED_ENV",
    "MODEL_ENV",
    "TypeSafeMeaningJudge",
    "governance_acknowledged",
]
