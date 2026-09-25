# Investigation digest lineage

Milestone 16 links every governance record to the immutable investigation
documents that supplied its evidence. A digest is a content identity, not a copy
of the document.

```text
investigation digest
  -> mining record / held-out evaluation case
  -> candidate and gate decision
  -> human review
  -> delivery authorization
  -> provider activation receipt
```

Candidate mining retains the digests of earlier successful investigations.
Held-out evaluation cases retain the digests of later investigations. When the
gate runs, it creates one ordered, duplicate-free evidence list from both sets.
That list is copied unchanged into the review, authorization, and receipt.

The digest fields do not change candidate drafting, paired evaluation, gate
thresholds, review policy, or delivery permissions. They make the existing
workflow auditable: a receipt can be traced back to the exact investigation
documents considered before delivery.

## Revising a verdict: supersede, never delete

An investigation document is write-once: the publisher refuses to publish an
investigation ID again with different bytes, and the mining reader trusts the
verdict inside every document tagged `learning.investigation.status =
"completed"`. When a judge fix shows a published verdict was wrong,
`revise_verdict` (in `providers/verdict_revision.py`) corrects it without
deleting anything:

1. It builds a new document from the original with the new verification:
   `extensions["learning.verification"]`, `execution_context["verified_success"]`,
   the per-step evidence fields and `assessment_refs` are replaced. The new ID
   is `<root investigation id>:rev<n>`, and `extensions["learning.revision"]`
   records `supersedes`, `supersedes_digest`, `root_investigation_id`,
   `revision`, `reason`, `judge_version`, `revised_at` and a `revision_key`
   (a digest of the inputs). `reason` and `judge_version` are codes, not prose.
2. It logs the new assessments on the same source trace, then publishes the new
   document as its own attachment trace with status `completed`. A revision's
   query index also carries `learning.investigation.supersedes` and
   `learning.investigation.revision_key`.
3. It retags the old document's trace: first
   `learning.investigation.superseded_by = <new id>`, then
   `learning.investigation.status = "superseded"`. The old attachment, its tags
   and its assessments stay as they were.

The call is safe to repeat. With the same inputs it returns `already_revised`
and changes nothing. If an earlier call stopped after publishing but before
retagging, the new document is recognised by its `revision_key` and the retag
is finished (`finished_interrupted_revision`). A document that is already
superseded, or whose revision slot holds a different revision, is refused:
revise the latest revision instead, which supersedes it and keeps the root ID.
Assessments are logged at least once; a crash between logging them and
publishing the document logs them again, with the same investigation ID in
their metadata.

The reader never mines a superseded trace, because it loads only `completed`
traces. If a crash left both the old and the new document `completed`, the
reader drops every document that another loaded document supersedes, so only
the latest revision of a run is mined.
