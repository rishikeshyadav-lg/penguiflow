# Direction-aware scoring

Each scorer still returns a mapping of named numeric outcomes for one case. For
example, one scorer can return task success, policy compliance, latency, cost,
and tool-error rate together.

`MetricSpecification` declares the direction for every metric used by a
promotion policy:

- `higher_is_better` is the default for scores such as task success and policy
  compliance.
- `lower_is_better` is for outcomes such as latency, cost, customer correction
  rate, and tool-error rate.

For every complete baseline/candidate pair, `MetricSummary` keeps both raw values
and a direction-normalized improvement. A positive improvement always means the
candidate is better:

```text
higher_is_better: candidate - baseline
lower_is_better:  baseline - candidate
```

The summary exposes mean, median, minimum, and maximum paired improvements.
`GateDecision` stores these paired summaries and the normalized mean improvement
alongside the existing baseline and candidate means. A primary metric must meet
its configured minimum improvement; every protected metric must remain at or above
zero normalized improvement.

`candidate_metric_thresholds` adds an absolute candidate gate. Higher-is-better
metrics must be at least their threshold, while lower-is-better metrics must be
at most their threshold. This prevents a candidate from passing merely because
it improved over a weak baseline while still missing an acceptable floor.

`confidence_interval_requirements` add bootstrap confidence bounds on top of the
point estimates, and `minimum_complete_cases` and
`minimum_complete_pairs_per_source_case` set the sample-size floor.

## Pairs a metric does not apply to

`MetricSpecification.denominator` names a metric that is 0 when this metric does
not apply to a run. A pair where either arm's denominator is 0 is left out of the
comparison and recorded in `MetricSummary.excluded_case_ids`. A pair missing the
denominator is a missing metric. If every pair is excluded, the candidate is
rejected with `no judged pairs for metric <name>`: no evidence is never a pass.
The denominator is opt-in, so existing policies behave as before.

## Re-judging both arms

`judging.VerificationMetric` judges each baseline and candidate run with the
integration's outcome ladder and reports `verified_success`, `hard_failure`,
`handled_correctly` and `judged`. `judging.verification_policy` makes
`verified_success` the primary metric and protects `hard_failure`, both with
`judged` as their denominator. A candidate must verify more runs without adding
hard failures, and a run handled correctly on either arm does not count against
it. `CombinedMetric` adds the integration's own metrics, such as latency or cost,
beside the judge's.

`PromotionPolicy.require_golden_set` makes `run_job` refuse, before any run and
without using an attempt, unless it is given a passing golden-set report (see
[the golden-set protocol](../judging/golden_set.md)).
