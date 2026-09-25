# The judge kit

`learning_control_plane.judging` holds the framework-neutral judging logic every integration
needs. The judge decides which runs are mined, and it decides whether a candidate skill improves
anything, so a judge mistake becomes a learning mistake.

- [Integrator guide](integrator_guide.md): describe a run, write a domain judge, publish after the
  turn, and gate with the judge.
- [Lessons](lessons.md): each judge mistake found in production mining and the kit piece that
  prevents it.
- [Golden-set protocol](golden_set.md): labels, accepted misses, and the runner that guards every
  judge change.

Runnable reference: `examples/lcp_plain_python_agent/` (`uv run python
examples/lcp_plain_python_agent/flow.py`).
