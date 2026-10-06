"""Log a report to MLflow. Optional: `pip install agent-evals[mlflow]`.

The aggregates written to MLflow are the scorecard's own numbers, so a report read back from MLflow equals
the report the local backend produced. MLflow is imported only when this is called, so the package keeps
no required dependency.
"""

from __future__ import annotations

import json

from .report import Report, report_json, report_markdown


def report_metrics(report: Report) -> dict[str, float]:
    """The scorecard as flat MLflow metrics: `scorecard.<key>` and, when there is a range, `.lower` and `.upper`."""

    metrics: dict[str, float] = {}
    for entry in report.scorecard.entries:
        if not entry.measured or entry.value is None:
            continue
        metrics[f"scorecard.{entry.key}"] = entry.value
        if entry.lower is not None and entry.upper is not None:
            metrics[f"scorecard.{entry.key}.lower"] = entry.lower
            metrics[f"scorecard.{entry.key}.upper"] = entry.upper
    metrics["operational.runs"] = float(report.operational.runs)
    metrics["operational.failed_runs"] = float(report.operational.failed_runs)
    return metrics


def log_report_to_mlflow(
    report: Report, *, tracking_uri: str, experiment_name: str, artifact_location: str | None = None
) -> str:
    """Log a report as one MLflow run (parameters, scorecard metrics, and the JSON and Markdown report) and return its id.

    With a database tracking store MLflow keeps artifacts under `./mlruns` unless told otherwise; pass
    `artifact_location` to say where a new experiment's artifacts go.
    """

    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    if artifact_location is not None and client.get_experiment_by_name(experiment_name) is None:
        client.create_experiment(experiment_name, artifact_location=artifact_location)
    mlflow.set_experiment(experiment_name)
    record = report.record
    with mlflow.start_run(run_name=record.run_id) as active:
        mlflow.log_params(
            {
                "variant_id": record.variant_id,
                "dataset_id": record.dataset_id,
                "dataset_version": record.dataset_version,
                "dataset_digest": record.dataset_digest,
                "suite": record.suite,
                "verdict_grade": str(record.verdict_grade),
                "settings": json.dumps(dict(record.settings), sort_keys=True),
            }
        )
        mlflow.log_metrics(report_metrics(report))
        mlflow.log_dict(report_json(report), "report.json")
        mlflow.log_text(report_markdown(report), "report.md")
        return active.info.run_id


__all__ = ["log_report_to_mlflow", "report_metrics"]
