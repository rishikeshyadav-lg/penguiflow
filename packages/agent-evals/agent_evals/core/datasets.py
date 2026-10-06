"""Datasets on disk: load, save, and a text-free manifest that pins what a run may use.

A dataset file holds the cases (and so may hold customer text). Its manifest never does: it holds
ids, the content digest, the suite type and the metric the dataset was built for, so it can live in a
repository while the dataset lives elsewhere. Loading with a manifest checks the digest, so a dataset
that changed since it was frozen is refused, and a run declares the metric it scores with so a dataset
built for one metric version is not silently scored with another.

Files are written deterministically: saving, loading and saving again gives the same bytes.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from .evaluation import EvaluationCase, EvaluationDataset, _non_empty

Suite = Literal["regression", "capability"]
_SUITES = ("regression", "capability")
_CASE_FIELDS = ("case_id", "inputs", "expected", "source_trace_id", "source_investigation_digest")


class MetricMismatchError(ValueError):
    """A run asked to score a dataset with a metric it was not built for."""


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """What a frozen dataset is: no case text, only ids, digest, suite and the metric it expects.

    `suite` says which rule judges a run: a regression suite guards against slipping backward and
    wants a hard threshold near 100%; a capability suite raises the ceiling and reports partial credit.
    """

    dataset_id: str
    version: str
    digest: str
    case_ids: Sequence[str]
    suite: Suite
    metric_id: str | None = None
    metric_version: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_id", _non_empty(self.dataset_id, "dataset_id"))
        object.__setattr__(self, "version", _non_empty(self.version, "version"))
        if self.suite not in _SUITES:
            raise ValueError(f"suite must be one of {', '.join(_SUITES)}")
        if (self.metric_id is None) != (self.metric_version is None):
            raise ValueError("metric_id and metric_version must be given together")
        object.__setattr__(self, "case_ids", tuple(self.case_ids))

    @classmethod
    def from_dataset(
        cls,
        dataset: EvaluationDataset,
        *,
        suite: Suite,
        metric_id: str | None = None,
        metric_version: str | None = None,
    ) -> DatasetManifest:
        """Freeze a dataset: record its digest and ids."""

        return cls(
            dataset_id=dataset.dataset_id,
            version=dataset.version,
            digest=dataset.manifest_digest,
            case_ids=[case.case_id for case in dataset.cases],
            suite=suite,
            metric_id=metric_id,
            metric_version=metric_version,
        )

    def check(self, dataset: EvaluationDataset) -> None:
        """Refuse a dataset that is not the one this manifest froze."""

        if dataset.manifest_digest != self.digest:
            raise ValueError(
                f"dataset {dataset.dataset_id} {dataset.version} has changed since it was frozen "
                f"(digest {dataset.manifest_digest}, manifest {self.digest})"
            )

    def check_metric(self, metric_id: str, metric_version: str) -> None:
        """Refuse a metric other than the one the dataset declares; a dataset that declares none accepts any."""

        if self.metric_id is None:
            return
        if (metric_id, metric_version) != (self.metric_id, self.metric_version):
            raise MetricMismatchError(
                f"dataset {self.dataset_id} {self.version} expects metric {self.metric_id} "
                f"{self.metric_version}, not {metric_id} {metric_version}"
            )

    def record(self) -> dict[str, Any]:
        return {**asdict(self), "case_ids": list(self.case_ids)}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> DatasetManifest:
        return cls(**record)


def _dumps(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=indent, separators=None if indent else (",", ":")
    )


def _case_from_record(record: Mapping[str, Any]) -> EvaluationCase:
    return EvaluationCase(
        case_id=record["case_id"],
        inputs=record["inputs"],
        expected=record.get("expected"),
        source_trace_id=record.get("source_trace_id"),
        source_investigation_digest=record.get("source_investigation_digest"),
    )


def save_dataset(dataset: EvaluationDataset, path: str | Path) -> None:
    """Write a dataset as `.json`, `.jsonl` or `.csv`, chosen by the file's suffix."""

    path = Path(path)
    cases = [asdict(case) for case in dataset.cases]
    header = {"dataset_id": dataset.dataset_id, "version": dataset.version}
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".json":
        path.write_text(_dumps({**header, "cases": cases}, indent=1) + "\n", encoding="utf-8")
    elif suffix == ".jsonl":
        lines = [_dumps(header), *(_dumps(case) for case in cases)]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    elif suffix == ".csv":
        # A CSV has no room for a header record; its dataset id and version are given again on load.
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(_CASE_FIELDS)
            for case in cases:
                writer.writerow(
                    [
                        case["case_id"],
                        _dumps(case["inputs"]),
                        _dumps(case["expected"]),
                        case["source_trace_id"] or "",
                        case["source_investigation_digest"] or "",
                    ]
                )
    else:
        raise ValueError(f"unsupported dataset format {path.suffix!r}; use .json, .jsonl or .csv")


def load_dataset(path: str | Path, *, dataset_id: str | None = None, version: str | None = None) -> EvaluationDataset:
    """Read a dataset. A CSV needs `dataset_id` and `version`; for the other formats they are read from the file."""

    path = Path(path)
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        payload = json.loads(text)
        header, records = payload, payload["cases"]
    elif suffix == ".jsonl":
        lines = [json.loads(line) for line in text.splitlines() if line.strip()]
        header, records = lines[0], lines[1:]
    elif suffix == ".csv":
        if dataset_id is None or version is None:
            raise ValueError("a CSV dataset needs dataset_id and version")
        header = {"dataset_id": dataset_id, "version": version}
        rows = csv.DictReader(text.splitlines())
        records = [
            {
                "case_id": row["case_id"],
                "inputs": json.loads(row["inputs"]),
                "expected": json.loads(row["expected"]),
                "source_trace_id": row["source_trace_id"] or None,
                "source_investigation_digest": row["source_investigation_digest"] or None,
            }
            for row in rows
        ]
    else:
        raise ValueError(f"unsupported dataset format {path.suffix!r}; use .json, .jsonl or .csv")
    return EvaluationDataset(
        dataset_id=dataset_id or header["dataset_id"],
        version=version or header["version"],
        cases=[_case_from_record(record) for record in records],
    )


def save_manifest(manifest: DatasetManifest, path: str | Path) -> None:
    """Write a manifest as JSON. It holds no case text."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dumps(manifest.record(), indent=1) + "\n", encoding="utf-8")


def load_manifest(path: str | Path) -> DatasetManifest:
    return DatasetManifest.from_record(json.loads(Path(path).read_text(encoding="utf-8")))


def load_frozen_dataset(
    dataset_path: str | Path, manifest_path: str | Path, *, dataset_id: str | None = None, version: str | None = None
) -> tuple[EvaluationDataset, DatasetManifest]:
    """Load a dataset and its manifest and refuse the pair if the dataset has changed since it was frozen."""

    manifest = load_manifest(manifest_path)
    dataset = load_dataset(
        dataset_path, dataset_id=dataset_id or manifest.dataset_id, version=version or manifest.version
    )
    manifest.check(dataset)
    return dataset, manifest


__all__ = [
    "DatasetManifest",
    "MetricMismatchError",
    "Suite",
    "load_dataset",
    "load_frozen_dataset",
    "load_manifest",
    "save_dataset",
    "save_manifest",
]
