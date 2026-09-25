from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from learning_control_plane.contracts.investigation import InvestigationTrajectoryV1, SourceTraceRef
from learning_control_plane.providers.local_store import LocalInvestigationStore


def _document(investigation_id: str = "investigation-1", *, minute: int = 0) -> InvestigationTrajectoryV1:
    return InvestigationTrajectoryV1(
        investigation_id=investigation_id,
        source_trace_ref=SourceTraceRef("local", "exp-1", f"trace-{investigation_id}", "sha256:inventory-v1"),
        agent_ref="inventory_agent",
        provider_ref="plain-python",
        scope_ref="tenant:demo",
        started_at=datetime(2026, 9, 1, 12, minute, tzinfo=UTC),
        status="completed",
        execution_fingerprint="sha256:inventory-v1",
        request={"has_text": True},
        steps=({"node": "query_stock"},),
        redaction_profile="safe:v1",
        step_signature="query_stock",
    )


def test_a_document_is_written_once_and_republishing_it_is_a_no_op(tmp_path: Path) -> None:
    store = LocalInvestigationStore(tmp_path)
    document = _document()

    first = store.publish(document)
    second = store.publish(document)

    assert first == second == document.digest()
    assert [stored.digest() for stored in store.documents()] == [document.digest()]


def test_different_bytes_under_the_same_investigation_id_are_refused(tmp_path: Path) -> None:
    store = LocalInvestigationStore(tmp_path)
    store.publish(_document())

    with pytest.raises(ValueError, match="different digest"):
        store.publish(replace(_document(), step_signature="lookup>query_stock"))


def test_records_come_from_completed_documents_oldest_first(tmp_path: Path) -> None:
    store = LocalInvestigationStore(tmp_path)
    store.publish(_document("b", minute=2))
    store.publish(_document("a", minute=1))
    store.publish(replace(_document("c", minute=0), status="failed"))

    records = store.load_records()

    assert [record.trace_id for record in records] == ["trace-a", "trace-b"]


def test_a_superseded_document_is_not_mined(tmp_path: Path) -> None:
    store = LocalInvestigationStore(tmp_path)
    original = _document("a")
    revision = replace(original, investigation_id="a:rev1", extensions={"learning.revision": {"supersedes": "a"}})
    store.publish(original)
    store.publish(revision)

    records = store.load_records()

    assert [record.investigation_digest for record in records] == [revision.digest()]


def test_an_empty_store_has_no_records(tmp_path: Path) -> None:
    assert LocalInvestigationStore(tmp_path / "missing").load_records() == ()
