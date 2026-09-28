# Learning Control Plane

A framework-agnostic offline learning loop for agents: mine repeated failure or weakness patterns
out of an agent's traces, draft an advisory skill, evaluate it against held-out cases, gate its
promotion, record human review, and deliver it back to the live agent. None of that -- the control
plane, the evidence contracts, the evaluation math, the mining and drafting -- knows anything about
which agent framework produced the traces.

This package has **zero required dependencies**. `mlflow`, and each agent framework's adapter, are
optional extras:

```bash
pip install learning-control-plane                    # core only: control plane, contracts, mining
pip install "learning-control-plane[mlflow]"           # + MLflow-backed evidence/publishers
pip install "learning-control-plane[penguiflow]"       # + the PenguiFlow FrameworkAdapter
pip install "learning-control-plane[langchain]"        # + the LangChain FrameworkAdapter
```

## Plugging in a framework

Every agent framework implements one small `FrameworkAdapter` (four methods: translate a run into a
framework-neutral shape, project it for mining, inject an approved skill at runtime, and deliver an
authorized skill). See
[`learning-control-plane-plug-and-play.md`](https://github.com/hurtener/penguiflow/blob/main/docs/learning-control-plane-plug-and-play.md)
for exactly what to implement, using the PenguiFlow and LangChain adapters
(`learning_control_plane/integrations/`) as templates.

## Docs and examples

This package's docs, design proposals, and runnable examples live in the main
[PenguiFlow repository](https://github.com/hurtener/penguiflow):

- [`docs/learning-control-plane-plug-and-play.md`](https://github.com/hurtener/penguiflow/blob/main/docs/learning-control-plane-plug-and-play.md) -- how to add a new framework.
- [`docs/proposals/FRAMEWORK_AGNOSTIC_LEARNING_CONTROL_PLANE.md`](https://github.com/hurtener/penguiflow/blob/main/docs/proposals/FRAMEWORK_AGNOSTIC_LEARNING_CONTROL_PLANE.md) -- the design this package implements a first cut of.
- `examples/learning_control_plane_local_demo/` and `examples/langchain_demo/` -- runnable, end-to-end demos of the full mine -> evaluate -> gate -> deliver loop.

## Status

This is a first, unpublished cut (`0.1.0`) of the Learning Control Plane as its own distribution,
split out of the `penguiflow` package it previously shipped inside. It is not yet published to PyPI.
