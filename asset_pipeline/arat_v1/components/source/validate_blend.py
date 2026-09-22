"""Validate the Blender source for the version-1 ARAT apparatus."""

from __future__ import annotations

import json
from pathlib import Path

import bpy


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "validation" / "asset_manifest.json"
REPORT_PATH = ROOT / "validation" / "blend_validation.json"


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected = {asset["model"] for asset in manifest["assets"]}
    roots = {obj.get("arat_model") for obj in bpy.data.objects if obj.name.startswith("ROOT_")}
    if expected != roots or len(expected) != 27:
        raise AssertionError(
            f"Blender model roots differ: missing={sorted(expected - roots)}, extra={sorted(roots - expected)}"
        )
    if bpy.context.scene.unit_settings.system != "METRIC" or bpy.context.scene.unit_settings.scale_length != 1.0:
        raise AssertionError("Blender source must use metres")
    if int(bpy.context.scene.get("asset_count", -1)) != 27:
        raise AssertionError("Blender asset_count metadata must be 27")
    report = {"status": "passed", "asset_version": 1, "asset_count": 27, "models": sorted(expected)}
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("ARAT_V1_BLEND_VALIDATION_PASSED assets=27")


if __name__ == "__main__":
    main()
