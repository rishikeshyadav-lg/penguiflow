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

The means and improvements above are descriptive. The gate's confidence claim comes
from case-clustered bootstrap intervals: a `PromotionPolicy` lists
`confidence_interval_requirements`, and each one is checked against a percentile
interval (`confidence_level`, `bootstrap_resamples`) computed by
`agent_evals.statistics.bootstrap_paired_intervals`, which resamples whole source
cases so the repeats of one case stay together. Sample-size rules are expressed as
`minimum_complete_pairs_per_source_case` and the detectability arithmetic in
`agent_evals.statistics.accuracy_improvement_detectability`; the thresholds
themselves are calibrated per agent (`agent_evals.calibration.propose_thresholds`).
