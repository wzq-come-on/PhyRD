from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:  # compatibility for ``python scripts/evaluate.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluation.cli import main


if __name__ == "__main__":
    main()
