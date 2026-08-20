#!/usr/bin/env python3
"""Launch normal OmniGibson teleoperation with wrist-flip tracing enabled."""

from __future__ import annotations

from datetime import datetime
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

from dex_teleop.omnigibson.launcher import main  # noqa: E402


def _has_option(arguments: list[str], option: str) -> bool:
    return option in arguments or any(argument.startswith(f"{option}=") for argument in arguments)


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if not _has_option(arguments, "--wrist-flip-output"):
        run_name = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        output_dir = REPOSITORY_ROOT / "dex_teleop" / "outputs" / "wrist_flip" / run_name
        arguments = ["--wrist-flip-output", str(output_dir), *arguments]
    else:
        option_index = arguments.index("--wrist-flip-output") if "--wrist-flip-output" in arguments else None
        if option_index is not None:
            output_dir = Path(arguments[option_index + 1])
        else:
            output_dir = Path(
                next(
                    argument.split("=", 1)[1]
                    for argument in arguments
                    if argument.startswith("--wrist-flip-output=")
                )
            )

    if not _has_option(arguments, "--recording-path") and output_dir is not None:
        arguments = ["--recording-path", str(output_dir / "teleop.hdf5"), *arguments]
    if not _has_option(arguments, "--maximum-frame-age"):
        arguments = ["--maximum-frame-age", "0.5", *arguments]
    if not _has_option(arguments, "--stale-frame-policy"):
        arguments = ["--stale-frame-policy", "hold", *arguments]
    if "--no-score" not in arguments:
        arguments = ["--no-score", *arguments]
    main(arguments)
