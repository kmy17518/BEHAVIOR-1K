import json
from pathlib import Path
import shlex
import sys

from dex_teleop.tracking.manus_diagnostics import main


FAKE_BRIDGE = Path(__file__).with_name("fake_manus_bridge.py")


def test_manus_diagnostics_records_acceptance_evidence(tmp_path):
    wrapper = tmp_path / "fake-manus-bridge"
    wrapper.write_text(
        "#!/bin/sh\n"
        f'exec {shlex.quote(sys.executable)} {shlex.quote(str(FAKE_BRIDGE))} "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    output = tmp_path / "acceptance.jsonl"

    result = main(
        [
            "--bridge",
            str(wrapper),
            "--mode",
            "remote",
            "--core-host",
            "192.0.2.10",
            "--duration",
            "0.03",
            "--minimum-rate-hz",
            "0",
            "--output",
            str(output),
        ]
    )

    records = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
        if line
    ]
    pair = next(record for record in records if record["type"] == "raw_skeleton_pair")
    tracker = next(record for record in records if record["type"] == "tracker_stream")
    summary = records[-1]

    assert result == 0
    assert pair["same_callback"] is True
    assert pair["joint_count"] == 25
    assert pair["joint_positions"]["wrist"] == [0.0, 0.0, 0.0]
    assert tracker["trackers"][0]["quality"] == "trackable"
    assert summary["type"] == "summary"
    assert summary["passed"] is True
    assert summary["all_pairs_same_callback"] is True
