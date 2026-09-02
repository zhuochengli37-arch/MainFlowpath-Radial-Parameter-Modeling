from __future__ import annotations

import sys
from pathlib import Path


WORKBASE_SRC = Path(__file__).resolve().parent.parent / "workbase" / "src"
workbase_src = str(WORKBASE_SRC)
if workbase_src not in sys.path:
    sys.path.insert(0, workbase_src)

from project1.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
