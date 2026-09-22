#!/usr/bin/env python3
"""Repository wrapper for the installed VIVE calibration command."""

from __future__ import annotations

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "dex_teleop" / "src"))

from dex_teleop.tracking.vive_calibration import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
