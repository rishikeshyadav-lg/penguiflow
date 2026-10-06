"""Where the fixtures, examples and the sibling package are, in either repository layout.

These tests are the same files in the monorepo (`tests/agent_evals/`) and in the standalone
`agent-evals` repository (`tests/`), where everything sits one directory nearer the root. Searching
upward keeps one copy of each test working in both. Each search names a distinctive child rather than
a bare directory name, because this monorepo has more than one directory called `examples`.
"""

from __future__ import annotations

from pathlib import Path


def _holding(relative: str) -> Path | None:
    """The nearest ancestor containing `relative`, or None when nothing does."""

    return next((parent for parent in Path(__file__).parents if (parent / relative).exists()), None)


_EXAMPLES_ROOT = _holding("examples/agent_evals_quickstart")
_FIXTURES_ROOT = _holding("fixtures/agent_evals")
_CONFORMANCE = "learning_control_plane/test_provider_conformance.py"
_CONFORMANCE_ROOT = _holding(_CONFORMANCE)

EXAMPLES = None if _EXAMPLES_ROOT is None else _EXAMPLES_ROOT / "examples"
FIXTURES = None if _FIXTURES_ROOT is None else _FIXTURES_ROOT / "fixtures"
# Only in the monorepo: the standalone repository does not ship the sibling package's tests.
CONFORMANCE_CASES = None if _CONFORMANCE_ROOT is None else _CONFORMANCE_ROOT / _CONFORMANCE
