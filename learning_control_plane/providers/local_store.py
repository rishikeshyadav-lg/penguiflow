"""A local directory as the investigation store, for integrations and demos without MLflow.

It keeps the same guarantees as the MLflow publisher: each investigation is written once as its
canonical bytes, publishing the same document again is a no-op, and publishing different bytes
under the same investigation ID is refused. Reading it back gives the same safe mining records the
MLflow reader gives: completed documents only, and only the latest revision of each run.
"""

from __future__ import annotations

from pathlib import Path

from ..contracts.investigation import InvestigationTrajectoryV1, investigation_from_canonical_bytes
from ..mining.investigation_mining import latest_revisions, learning_record_from_investigation
from ..mining.mining import TraceLearningRecord


class LocalInvestigationStore:
    """Write-once investigation documents in one directory, one canonical JSON file each."""

    def __init__(self, directory: Path | str) -> None:
        self._directory = Path(directory)

    def publish(self, document: InvestigationTrajectoryV1) -> str:
        """Write the document once and return its digest; refuse different bytes under the same ID."""

        path = self._path(document.investigation_id)
        content = document.canonical_bytes()
        if path.exists():
            if path.read_bytes() != content:
                raise ValueError(
                    f"investigation_id {document.investigation_id!r} was already published with a different digest"
                )
            return document.digest()
        self._directory.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
        return document.digest()

    def documents(self) -> tuple[InvestigationTrajectoryV1, ...]:
        """Return every stored document."""

        if not self._directory.exists():
            return ()
        return tuple(
            investigation_from_canonical_bytes(path.read_bytes()) for path in sorted(self._directory.glob("*.json"))
        )

    def load_records(self) -> tuple[TraceLearningRecord, ...]:
        """Return safe mining records for completed documents, latest revision only, oldest run first."""

        completed = [document for document in self.documents() if document.status == "completed"]
        records = [learning_record_from_investigation(document) for document in latest_revisions(completed)]
        return tuple(sorted(records, key=lambda record: (record.recorded_at, record.trace_id)))

    def _path(self, investigation_id: str) -> Path:
        safe_name = "".join(
            character if character.isalnum() or character in "-_." else "_" for character in investigation_id
        )
        return self._directory / f"{safe_name}.json"


__all__ = ["LocalInvestigationStore"]
