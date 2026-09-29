# agent-evals

Agent-agnostic evaluation. Bring any agent (any framework, or a plain function), a fixed dataset and
a scorer; get repeatable runs, honest statistics and a report. This package has **no required
dependencies** and imports nothing from any agent framework.

```bash
pip install agent-evals
```

## What it is, and is not

- **Is:** cases and datasets with a content digest, a run contract every agent can meet, a run
  backend, scorers, and the statistics an evaluation needs (a case-clustered paired bootstrap, the
  agent's own run-to-run noise, minimum detectable effect).
- **Is not:** an agent framework, a tracing system, or a learning loop. The learning control plane
  (`learning-control-plane`) is one consumer of this package; it decides when a skill may ship.

## Status

`0.1.0`, unpublished, under construction. The design and milestones live in
[`docs/evaluation-framework.md`](https://github.com/hurtener/penguiflow/blob/main/docs/evaluation-framework.md).
The package name is a working name: `agent-eval` is already taken on PyPI, so rename before publishing.
