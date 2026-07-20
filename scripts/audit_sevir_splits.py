from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

from phyrd.data import SEVIRDataset
from phyrd.utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit frozen event-time SEVIR splits")
    parser.add_argument("--data-root", default="/test1/wzq/data/sevir")
    parser.add_argument("--output", default="artifacts/data_audit/sevir_split_audit.json")
    args = parser.parse_args()
    splits = ("train", "val_model", "val_calib", "report_test")
    identifiers: dict[str, set[str]] = {}
    report: dict[str, Any] = {"splits": {}}
    resolved = None
    for split in splits:
        dataset = SEVIRDataset(args.data_root, split)
        ids = set(dataset.rows["id"].astype(str))
        identifiers[split] = ids
        times = dataset.rows["time_utc"]
        report["splits"][split] = {
            "events": len(dataset),
            "time_min": times.min().isoformat(),
            "time_max": times.max().isoformat(),
        }
        resolved = dataset.paths
        dataset.close()
    overlaps = {}
    for first, second in combinations(splits, 2):
        overlap = identifiers[first].intersection(identifiers[second])
        overlaps[f"{first}__{second}"] = len(overlap)
        if overlap:
            raise AssertionError(f"event leakage between {first} and {second}: {len(overlap)}")
    report["overlaps"] = overlaps
    report["status"] = "PASS"
    report["catalog_path"] = str(resolved.catalog_path)
    report["data_root"] = str(resolved.data_root)
    write_json(Path(args.output), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
