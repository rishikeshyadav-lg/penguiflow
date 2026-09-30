# Publishing checklist: `agent-evals`

Status: **not published, and not to be published without the owner's go.** Written 2026-09-30.

## The name
- Checked 2026-09-30 (read-only, `https://pypi.org/pypi/<name>/json`): `agent-evals` and `agent_evals` returned 404,
  so unclaimed *at that moment*; `agent-eval` and `agent-evaluation` exist. Re-check on the day: a name can be
  claimed at any time.
- `agent-eval` (taken) and `agent-evals` (free) differ by one letter. Decide whether that confusion is acceptable
  or whether a more distinct name is worth it; renaming is cheap only until something is published.

## Order of publishing
1. `agent-evals` first: it depends on nothing, and the LCP will depend on it.
2. `learning-control-plane` second, with its dependency `agent-evals` pinned to the version just published.
3. The campaign repo's next pin bump adds a third source (`agent-evals`) at the same commit as the other two; it
   cannot do that until the commit is pushed to the fork it pins to.

## Before the first upload
- [ ] Owner's explicit go to publish, and to push the local commits the pin needs.
- [ ] Name decided and re-checked on PyPI; `pyproject.toml` name, README and import name agree.
- [ ] Version chosen (`0.1.0`) and a changelog entry written.
- [ ] `LICENSE` present in the package folder and named in `pyproject.toml` (it is).
- [ ] `uv build` in `packages/agent-evals`; install the wheel into a clean environment and run
      `python -c "import agent_evals"`; confirm no `learning_control_plane`, `penguiflow` or `langchain` module loads
      (the import-purity test covers this).
- [ ] Run `tests/agent_evals` and `tests/learning_control_plane` from that clean environment.
- [ ] Try the optional extra: `pip install "agent-evals[mlflow]"` and log a report.
- [ ] Look through the wheel's contents for anything that should not ship (fixtures with case text, local paths).
- [ ] Upload to TestPyPI first; install from it and run the quickstart example.
- [ ] Tag the release commit.

## Known limits to state in the release notes
- Plan adherence and coherence are experimental; their agreement with labels is unmeasured.
- `dry_run` in shadow comparison is a request the agent must honour; nothing is sandboxed.
- A generic `explain_verdict` is not included.
- Policy checks detect violations; they do not enforce.
