#!/usr/bin/env python3
"""Repository-local bootstrap for ``dex_teleop.omnigibson.replay``."""

from __future__ import annotations

import os
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPOSITORY_ROOT / "datasets"
configured_data_root = os.environ.get("OMNIGIBSON_DATA_PATH")
if configured_data_root is not None and Path(configured_data_root).expanduser().resolve() != DATA_ROOT.resolve():
    raise SystemExit(
        "OMNIGIBSON_DATA_PATH points outside Behavior-1K-arat; unset it or set it to "
        f"{DATA_ROOT}"
    )
os.environ["OMNIGIBSON_DATA_PATH"] = str(DATA_ROOT)

for path in (REPOSITORY_ROOT / "dex_teleop" / "src", REPOSITORY_ROOT / "OmniGibson", REPOSITORY_ROOT / "bddl3"):
    sys.path.insert(0, str(path))

from dex_teleop.omnigibson.replay import main  # noqa: E402


if __name__ == "__main__":
    main()
