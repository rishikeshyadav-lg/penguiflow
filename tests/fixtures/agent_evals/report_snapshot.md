# Evaluation report: cand

- run `run-1` created 2026-09-29T12:00:00+00:00
- dataset `questions` v1 (`sha256:3e32f8767558534c1d313465bbf498bc690d8c1304a682c8ea7e1c4e6ec9b316`), suite: capability
- verdict grade: yes

## Suite

Rule applied: capability: partial credit; mean success with a 95% case-clustered interval; no pass or fail

Result: mean 0.833 over 6 cases, range 0.500 to 1.000; no pass or fail for a capability suite.

## Scorecard

| Metric | Layer | Value | 95% range | Runs | Status |
|---|---|---|---|---|---|
| Task success rate | outcome | 0.833 rate | 0.500 to 1.000 | 6 used, 0 excluded | measured |
| Tool selection accuracy | trajectory | 0.861 score | 0.694 to 1.000 | 6 used, 0 excluded | measured |
| Argument correctness | trajectory | 1.000 score | 1.000 to 1.000 | 6 used, 0 excluded | measured |
| Plan adherence | trajectory | not measured | - | - | not configured: plan adherence needs a judge |
| Execution efficiency | operational | 0.944 score | 0.833 to 1.000 | 6 used, 0 excluded | measured |
| Cost per task | operational | 0.0500 usd | 0.0500 to 0.0500 | 6 used, 0 excluded | measured |
| p95 latency | operational | 1475 ms | 1275 to 1500 | 6 used, 0 excluded | measured |
| Policy violations | policy | 0.167 share of runs | - | 6 used, 0 excluded | FLAG RAISED (cases: c6) |

## Audit signals

- success 0.83 against tool_selection_accuracy 0.86: no gap worth auditing.

## Failed runs

None.
