from __future__ import annotations

import json

import torch


def main() -> None:
    report = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        torch.cuda.init()
        current = torch.cuda.current_device()
        tensor = torch.ones(1, device=f"cuda:{current}")
        torch.cuda.reset_peak_memory_stats()
        report.update(
            {
                "current_device": current,
                "device_name": torch.cuda.get_device_name(current),
                "tensor_device": str(tensor.device),
                "peak_memory": torch.cuda.max_memory_allocated(),
            }
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
