from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected a mapping in {path}")
    return value


def write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def query_gpu_health() -> list[dict[str, int]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,temperature.gpu,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    records: list[dict[str, int]] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        index, temperature, memory_used, utilization = [
            int(part.strip()) for part in line.split(",")
        ]
        records.append(
            {
                "index": index,
                "temperature_c": temperature,
                "memory_used_mib": memory_used,
                "utilization_percent": utilization,
            }
        )
    return records


def select_healthy_gpus(config: dict[str, Any]) -> tuple[list[int], list[dict[str, Any]]]:
    requested = {int(index) for index in config["requested_gpus"]}
    health_config = config["health"]
    max_temperature = int(health_config["max_temperature_c"])
    max_memory = int(health_config["max_memory_used_mib"])
    max_utilization = int(health_config["max_utilization_percent"])
    health = query_gpu_health()
    selected: list[int] = []
    annotated: list[dict[str, Any]] = []
    for record in health:
        reasons: list[str] = []
        if record["index"] not in requested:
            reasons.append("not_requested")
        if record["temperature_c"] > max_temperature:
            reasons.append("temperature")
        if record["memory_used_mib"] > max_memory:
            reasons.append("memory_in_use")
        if record["utilization_percent"] > max_utilization:
            reasons.append("gpu_busy")
        accepted = not reasons
        annotated.append({**record, "accepted": accepted, "reasons": reasons})
        if accepted:
            selected.append(record["index"])
    return selected, annotated


def make_seed_configs(
    launcher_config: dict[str, Any], seed: int, run_root: Path
) -> tuple[Path, Path]:
    deterministic = copy.deepcopy(read_yaml(Path(launcher_config["deterministic_config"])))
    residual = copy.deepcopy(read_yaml(Path(launcher_config["residual_config"])))
    seed_root = (run_root / f"seed_{seed}").resolve()
    deterministic_dir = seed_root / "deterministic"
    residual_dir = seed_root / "residual"
    deterministic["seed"] = seed
    residual["seed"] = seed
    deterministic["device"] = "cuda:0"
    residual["device"] = "cuda:0"
    deterministic["data"]["root"] = launcher_config["data_root"]
    residual["data"]["root"] = launcher_config["data_root"]
    deterministic["artifacts"]["directory"] = str(deterministic_dir)
    residual["artifacts"]["directory"] = str(residual_dir)
    residual["model"]["deterministic_checkpoint"] = str(
        deterministic_dir / "checkpoint_best.pt"
    )
    deterministic_path = run_root / "configs" / f"seed_{seed}_deterministic.yaml"
    residual_path = run_root / "configs" / f"seed_{seed}_residual.yaml"
    write_yaml(deterministic_path, deterministic)
    write_yaml(residual_path, residual)
    return deterministic_path.resolve(), residual_path.resolve()


def run_worker(args: argparse.Namespace) -> None:
    if args.det_config is None or args.res_config is None or args.status_path is None:
        raise ValueError("worker mode requires config and status paths")
    status_path = Path(args.status_path)
    status: dict[str, Any] = {
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pid": os.getpid(),
        "started_unix": time.time(),
        "stage": "deterministic",
        "status": "running",
    }
    write_json(status_path, status)
    for stage, config_path in (
        ("deterministic", args.det_config),
        ("residual", args.res_config),
    ):
        status["stage"] = stage
        status[f"{stage}_started_unix"] = time.time()
        write_json(status_path, status)
        command = [sys.executable, "-u", "scripts/train.py", "--config", config_path]
        completed = subprocess.run(command, check=False)
        status[f"{stage}_exit_code"] = completed.returncode
        status[f"{stage}_ended_unix"] = time.time()
        if completed.returncode != 0:
            status["status"] = "failed"
            write_json(status_path, status)
            raise SystemExit(completed.returncode)
    status["status"] = "completed"
    status["ended_unix"] = time.time()
    write_json(status_path, status)


def launch(config_path: Path, dry_run: bool) -> None:
    config = read_yaml(config_path)
    run_root = Path(config["run_root"])
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(
            f"refusing to reuse non-empty run directory {run_root}; choose a new run_root"
        )
    healthy_gpus, health = select_healthy_gpus(config)
    seeds = [int(seed) for seed in config["seeds"]]
    assignments = list(zip(healthy_gpus, seeds, strict=False))
    if not assignments:
        raise RuntimeError("no healthy requested GPU is available")
    run_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "experiment_name": config["experiment_name"],
        "config": str(config_path.resolve()),
        "created_unix": time.time(),
        "gpu_health": health,
        "assignments": [],
        "unassigned_seeds": seeds[len(assignments) :],
        "dry_run": dry_run,
    }
    jobs: list[tuple[subprocess.Popen[bytes], Any]] = []
    for gpu, seed in assignments:
        det_config, res_config = make_seed_configs(config, seed, run_root)
        log_path = (run_root / "logs" / f"gpu_{gpu}_seed_{seed}.log").resolve()
        status_path = (run_root / "status" / f"gpu_{gpu}_seed_{seed}.json").resolve()
        assignment = {
            "physical_gpu": gpu,
            "seed": seed,
            "deterministic_config": str(det_config),
            "residual_config": str(res_config),
            "log": str(log_path),
            "status": str(status_path),
        }
        manifest["assignments"].append(assignment)
        if dry_run:
            continue
        log_path.parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log_handle = log_path.open("ab", buffering=0)
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--worker",
            "--det-config",
            str(det_config),
            "--res-config",
            str(res_config),
            "--status-path",
            str(status_path),
        ]
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        assignment["pid"] = process.pid
        jobs.append((process, log_handle))
    write_json(run_root / "launcher_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    if dry_run:
        return
    exit_codes: list[int] = []
    for process, log_handle in jobs:
        exit_codes.append(process.wait())
        log_handle.close()
    manifest["worker_exit_codes"] = exit_codes
    manifest["ended_unix"] = time.time()
    write_json(run_root / "launcher_manifest.json", manifest)
    if any(exit_codes):
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one two-stage PhyRD seed per healthy GPU")
    parser.add_argument("--config", default="configs/archive/train_8gpu_seeds.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--det-config", help=argparse.SUPPRESS)
    parser.add_argument("--res-config", help=argparse.SUPPRESS)
    parser.add_argument("--status-path", help=argparse.SUPPRESS)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.worker:
        run_worker(arguments)
    else:
        launch(Path(arguments.config), arguments.dry_run)
