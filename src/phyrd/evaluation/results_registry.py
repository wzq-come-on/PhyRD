from __future__ import annotations

import csv
from datetime import date
from pathlib import Path
from typing import Any


FIELDS = (
    "result_id",
    "recorded_date",
    "experiment",
    "deterministic_backbone",
    "probabilistic_module",
    "protocol",
    "split",
    "samples_or_events",
    "ensemble_size",
    "world_size",
    "checkpoint_epoch",
    "CRPS",
    "CSI",
    "CSI_pool4",
    "CSI_pool16",
    "HSS",
    "MAE",
    "MSE",
    "SSIM",
    "status",
    "evidence",
    "notes",
)


def append_full_test_result(
    registry_path: str | Path,
    *,
    metrics: dict[str, Any],
    experiment: str,
    deterministic_backbone: str,
    probabilistic_module: str,
    evidence: str,
    notes: str = "Automatically registered after full post-training test",
) -> str:
    """Append one completed full-test result, idempotently by evidence path."""
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if row.get("evidence") == evidence:
                return str(row["result_id"])
    numeric_ids = [
        int(row["result_id"].split("-")[-1])
        for row in rows
        if row.get("result_id", "").startswith("R-")
        and row["result_id"].split("-")[-1].isdigit()
    ]
    result_id = f"R-{max(numeric_ids, default=-1) + 1:03d}"
    row: dict[str, Any] = {
        "result_id": result_id,
        "recorded_date": date.today().isoformat(),
        "experiment": experiment,
        "deterministic_backbone": deterministic_backbone,
        "probabilistic_module": probabilistic_module,
        "protocol": metrics.get("protocol", "5to20@128"),
        "split": metrics["split"],
        "samples_or_events": metrics["samples"],
        "ensemble_size": metrics["ensemble_size"],
        "world_size": metrics["world_size"],
        "checkpoint_epoch": metrics.get("epoch", ""),
        "CRPS": metrics.get("CRPS", ""),
        "CSI": metrics.get("CSI", ""),
        "CSI_pool4": metrics.get("CSI_pool4", ""),
        "CSI_pool16": metrics.get("CSI_pool16", ""),
        "HSS": metrics.get("HSS", ""),
        "MAE": metrics.get("MAE", ""),
        "MSE": metrics.get("MSE", ""),
        "SSIM": metrics.get("SSIM", ""),
        "status": metrics.get("status", "completed"),
        "evidence": evidence,
        "notes": notes,
    }
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return result_id
