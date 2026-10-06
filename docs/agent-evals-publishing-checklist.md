# Distribution of `agent-evals`: in-house, not on PyPI

Status: **decided 2026-09-30: in-house, not published to PyPI or open-sourced.** That still holds.

**Decided 2026-10-06, after an agnosticism audit:** the package moves to a repository the organisation owns
rather than a personal fork, so other teams can depend on it without depending on one account. Distribution
stays a pinned source archive; only the URL changes. Still to do, and it needs someone with org access:

- [ ] Create the org repository and decide whether it holds only `agent-evals` or the whole monorepo.
- [ ] Push `feat/lcp-framework-agnostic` there and re-point the install URL in `packages/agent-evals/README.md`
      and `Homepage` in its `pyproject.toml` (both currently name the fork).
- [ ] Re-point `[tool.uv.sources]` in the campaign repo's `pyproject.toml` (three entries, one commit) and
      `uv lock`.
- [ ] Write a CHANGELOG and tag `0.1.0`, so consumers bump a version rather than a commit SHA.

**Done 2026-10-06 so another team can pick this up unaided:**

- `python -m agent_evals.selfcheck` ships inside the package and proves, from a standalone install, both that
  importing it loads no framework, model client or backend and that unrelated agent shapes score identically.
  Verified from a cold install with `agent-evals` as the only package present.
- The run row names the variant `variant_id`, not the campaign's `arm`; `from_record` still reads the old name
  and the campaign's `legacy_shaped_row` maps it back, so recorded runs keep working.
- The README quickstart runs as written (it called an undefined `my_agent` and an undefined scorer), and the
  README no longer names two different GitHub remotes.

Known residue, not yet addressed: `category`, `pattern_key`, `set` and `native_trace_id` remain optional row
labels from the first consumer's vocabulary, and `advisory_skill` / `source_investigation_digest` sit in the
neutral core. All are optional. A tool-less chat agent and an HTTP agent are not covered end to end by a test.

## How it is distributed instead
Consumers install a pinned source archive of the owner's fork, not a package index:

- the campaign repo declares `agent-evals`, `learning-control-plane` and `penguiflow` as sources in
  `[tool.uv.sources]`, all at the same commit of `rishikeshyadav-lg/penguiflow` (currently `e1be846`);
- to ship a change: push it to the fork's `feat/lcp-framework-agnostic`, then change the commit in those three source
  lines in the campaign's `pyproject.toml`, run `uv lock`, and run the campaign tests;
- to try it locally without the pin: `uv pip install -e packages/agent-evals`.

Keep the name `agent-evals` unchanged: the import name and the pin already use it, and with no publishing the
PyPI name clash below does not matter.

## If publishing is ever reconsidered
The checklist below is the plan from when it was still open; it is kept, not scheduled.

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
