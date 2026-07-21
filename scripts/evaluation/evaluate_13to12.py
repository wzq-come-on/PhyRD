from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.evaluation.common import run


if __name__ == "__main__":
    run("13to12", sys.argv[1:])
