# The golden-set protocol

A golden set is a list of stored runs, each labelled with the outcome a correct judge should
reach: `verified`, `failed`, `handled_correctly`, `agent_error` or `not_judgeable`. It is how a
judge change is checked before it judges anything that costs money.

## Rules

1. **Labels come from a person reading the run**, not from the judge. Each label records why:
   `GoldenLabel(case_key, expected_outcome, reason)`.
2. **Stored runs hold customer text, so they stay outside the repository.** Pass them to the runner
   as paths. Repository fixtures are synthetic; `tests/learning_control_plane/test_privacy_scan.py`
   enforces it.
3. **Known disagreements are accepted misses, listed explicitly.** An `AcceptedMiss(case_key,
   judged_outcome, reason)` says the label is right and the judge is known to reach
   `judged_outcome`. Never change a label or add an accepted miss just to make the set pass.
4. **The set passes only when every other label agrees.**
   - A new disagreement is a new miss.
   - An accepted miss whose judged outcome changes is also a new miss, because the judge's
     behaviour changed.
   - An accepted miss that now agrees is reported as fixed, so it can be removed from the list.
5. **Run it after every judge change and before every paid replay.** A promotion policy with
   `require_golden_set=True` refuses to run a job unless it is given a passing report.

## Running it

From code:

```python
report = await run_golden_set(labels, load_run, judge, reference=reference_builder, accepted_misses=misses)
print(report.summary())
```

`load_run(label)` returns the stored `AgentRun`, or None, which never agrees. `judge(run,
reference)` returns an `InvestigationVerification`, for example `OutcomeLadder.judge`. A reference
that fails is recorded on the result, and the run is judged without it, as the live judge would be.

From the command line, with stored runs as JSON:

```bash
python -m learning_control_plane.judging.golden \
    --labels labels.json --runs runs.json \
    --judge my_package.judging:build_judge \
    --reference my_package.judging:build_reference \
    --accepted-misses accepted_misses.json --output results.json
```

The command exits non-zero on a new miss.

## Keeping it honest

- Grow the set whenever mining finds a new kind of judge mistake. Add the run with its correct
  label before fixing the judge, so the fix is proven by the set.
- The meaning check reads with a model and varies slightly from run to run. A label that flips
  between runs sits near a bar. Fix it with a deterministic rule where you can, as the benchmark
  claim check does, rather than by moving the bar.
