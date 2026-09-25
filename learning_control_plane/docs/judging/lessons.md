# Judge mistakes and the kit piece that prevents each

The first integration (a campaign-performance agent) found these mistakes in its judge over a
week of mining real runs. Each fix is now a generic piece of `learning_control_plane.judging`, so a
new integration starts with it instead of rediscovering it.

| Judge mistake | Kit piece |
|---|---|
| Answers cut off mid-sentence were scored on the numbers they reached | `answer_text.looks_truncated`: the ladder records an `agent_error` |
| A clarification, a "not found", or a switched-off service was reported as a failure | `outcomes`: `handled_correctly`, never verified or mined, and left out of rates |
| A tool error the agent recovered from blocked verification | `steps.recovered_step_indexes` |
| The planner's `"... [N more items]"` note was read as a malformed item | `answer_text.SHORTENED_LIST_NOTE`, `strip_shortened_list_note` |
| Values in rendered tables were ignored, and any number anywhere was accepted | `answer_text.shown_to_user` and `answer_facts.stated_values`: labelled table, line and prose values, explicit or inferred |
| A rank column was taken for a table row's name | `answer_facts` skips `#`, `Rank`, `No.` columns when choosing the label column |
| Rounding was read as a contradiction; a combined total was compared with one row | `expectations.judge_expectation` with a per-metric `Tolerance` and separate rows |
| Scope was matched by substring, required full names, or required every ID | `scope.question_scope_check`: terms from the question first, any named term is enough, letters and digits only, droppable suffixes, optional words-on-one-line |
| Missing rows were summed into zero totals, and "not found" was failed | `no_data.no_data_judgment`: confirmed no data versus stated zero totals |
| A correct "not found" had no reference to confirm it | `no_data.EmptyLookupRule`: every scoped lookup came back empty; searches for part of a name are fallbacks |
| The judge's expectation came from the agent's own calls, which is circular | `judge.ReferenceBuilder`: the integration supplies an independent query |
| The model reading answers was unstable, and flags flipped at the bar | `meaning.MeaningCheck`: fail-only, several reads averaged, a bar per question, failed reads skipped |
| An invented "typical range" scored right at the meaning bar | `claims.states_unsupported_benchmark`: deterministic, and benchmarks the user supplied do not count |
| "Verify the budget is on track" was read as a verdict | `claims.states_as_fact`: a claim after a request to check is not stated as fact |
| The classification flipped near its cutoff, so the rubric changed between replays | `classification.freeze_classification`: majority of several reads, frozen before replay |
| Pattern keys never repeated | `signature.SignatureRules` |
| Judge changes regressed silently | `golden.run_golden_set` with accepted misses, and `PromotionPolicy.require_golden_set` before paid runs |
| The gate never re-judged candidates | `metric.VerificationMetric`, `verification_policy`, and `MetricSpecification.denominator` |
| Verdicts were lost when the trace was not ready | `TraceReadiness`, `PendingAssessmentQueue`, and `TurnStash` |
| A verdict found wrong after it was published could not be corrected | `providers.verdict_revision.revise_verdict`: supersede, never delete |
