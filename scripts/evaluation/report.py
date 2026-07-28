"""Build a paper-style comparison table from unified evaluation JSON files."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence


MISSING = "—"


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        if value.upper() in {"SKIPPED", "N/A", "NONE", "NULL", "—", "-"}:
            return None
        try:
            return float(value)
        except ValueError:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric(payload: dict[str, Any], name: str) -> float | None:
    aliases = {
        "CSI": ("CSI_global", "CSI", "csi_global", "csi"),
        "SSIM": ("SSIM", "ssim"),
        "HSS": ("HSS", "hss"),
        "CRPS": ("CRPS", "crps"),
        "LPIPS": ("LPIPS", "lpips"),
    }
    for key in aliases.get(name, (name,)):
        if key in payload and payload[key] is not None:
            return _number(payload[key])
    return None


def _fmt(value: float | None, digits: int = 5) -> str:
    return MISSING if value is None else f"{value:.{digits}f}"


def _conclusion(reference: dict[str, Any], current: dict[str, Any]) -> str:
    """Summarize changes against the first (reference) row."""
    checks = (
        ("CRPS", "lower", "CRPS 改善", "CRPS 变差"),
        ("CSI", "higher", "CSI 提升", "CSI 下降"),
        ("SSIM", "higher", "SSIM 提升", "SSIM 下降"),
        ("HSS", "higher", "HSS 提升", "HSS 下降"),
        ("LPIPS", "lower", "LPIPS 改善", "LPIPS 变差"),
    )
    changes: list[str] = []
    for metric, direction, better, worse in checks:
        base, value = _metric(reference, metric), _metric(current, metric)
        if base is None or value is None or abs(value - base) < 1e-8:
            continue
        improved = value < base if direction == "lower" else value > base
        changes.append(better if improved else worse)
    return "，".join(changes) if changes else "—"


def load_rows(row_specs: Sequence[Sequence[str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in row_specs:
        if len(spec) != 4:
            raise ValueError("each --row requires: NAME TYPE K JSON_PATH")
        name, kind, k, path = spec
        result_path = Path(path)
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"evaluation result must be a JSON object: {result_path}")
        rows.append(
            {
                "name": name,
                "kind": kind,
                "k": k,
                "path": str(result_path),
                "payload": payload,
            }
        )
    if not rows:
        raise ValueError("at least one --row is required")
    return rows


def make_table(rows: list[dict[str, Any]]) -> str:
    reference = rows[0]["payload"]
    headers = [
        "方法（自己复现）",
        "类型（简单描述）",
        "K",
        "CSI↑",
        "SSIM↑",
        "HSS↑",
        "CRPS↓",
        "LPIPS↓",
        "结论",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for index, row in enumerate(rows):
        payload = row["payload"]
        values = [
            row["name"],
            row["kind"],
            str(row["k"]),
            _fmt(_metric(payload, "CSI")),
            _fmt(_metric(payload, "SSIM")),
            _fmt(_metric(payload, "HSS")),
            _fmt(_metric(payload, "CRPS")),
            _fmt(_metric(payload, "LPIPS")),
            "" if index == 0 else _conclusion(reference, payload),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def make_csv(rows: list[dict[str, Any]]) -> str:
    headers = ["method", "type", "K", "CSI", "SSIM", "HSS", "CRPS", "LPIPS", "conclusion"]
    output: list[list[str]] = []
    reference = rows[0]["payload"]
    for index, row in enumerate(rows):
        payload = row["payload"]
        output.append(
            [
                row["name"],
                row["kind"],
                str(row["k"]),
                _fmt(_metric(payload, "CSI")),
                _fmt(_metric(payload, "SSIM")),
                _fmt(_metric(payload, "HSS")),
                _fmt(_metric(payload, "CRPS")),
                _fmt(_metric(payload, "LPIPS")),
                "" if index == 0 else _conclusion(reference, payload),
            ]
        )
    stream = sys.stdout
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(headers)
    writer.writerows(output)
    return ""


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create a paper-style table from evaluation JSONs")
    parser.add_argument(
        "--row",
        action="append",
        nargs=4,
        metavar=("NAME", "TYPE", "K", "JSON_PATH"),
        required=True,
        help="row metadata followed by a unified evaluation JSON path; repeat for each method",
    )
    parser.add_argument("--output", help="write Markdown/CSV table to this path")
    parser.add_argument("--format", choices=("markdown", "csv"), default="markdown")
    args = parser.parse_args(argv)
    rows = load_rows(args.row)
    if args.format == "markdown":
        content = make_table(rows)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(content, encoding="utf-8")
        else:
            print(content, end="")
    else:
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with Path(args.output).open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(["method", "type", "K", "CSI", "SSIM", "HSS", "CRPS", "LPIPS", "conclusion"])
                reference = rows[0]["payload"]
                for index, row in enumerate(rows):
                    payload = row["payload"]
                    writer.writerow(
                        [
                            row["name"], row["kind"], row["k"], _fmt(_metric(payload, "CSI")),
                            _fmt(_metric(payload, "SSIM")), _fmt(_metric(payload, "HSS")),
                            _fmt(_metric(payload, "CRPS")), _fmt(_metric(payload, "LPIPS")),
                            "" if index == 0 else _conclusion(reference, payload),
                        ]
                    )
        else:
            make_csv(rows)


if __name__ == "__main__":
    main()
