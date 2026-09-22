"""Strict protocol-v1 CLI wrapper that rejects v2-only bridge arguments."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


PROTOCOL = "dex_teleop.manus"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-version", action="store_true")
    parser.add_argument("--protocol-fd", type=int)
    parser.add_argument("--mode", choices=("integrated",), default="integrated")
    parser.add_argument("--connect-timeout")
    parser.add_argument("--glove-timeout")
    parser.add_argument("--reconnect-timeout")
    parser.add_argument("--discovery-wait")
    parser.add_argument("--hand-motion")
    parser.add_argument("--tracker-diagnostics", action="store_true")
    parser.add_argument("--no-tracker-diagnostics", action="store_true")
    parser.add_argument("--loopback-only", action="store_true")
    parser.add_argument("--settings-dir")
    parser.add_argument("--log-dir")
    parser.add_argument("--left-calibration")
    parser.add_argument("--right-calibration")
    args = parser.parse_args()
    if args.protocol_version:
        print(f"{PROTOCOL} 1")
        return
    if args.protocol_fd is None:
        parser.error("--protocol-fd is required")

    fake = Path(__file__).with_name("fake_manus_bridge.py")
    os.execv(
        sys.executable,
        [
            sys.executable,
            str(fake),
            "--scenario",
            "legacy_protocol",
            *sys.argv[1:],
        ],
    )


if __name__ == "__main__":
    main()
