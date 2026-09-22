"""Convert all validated ARAT v1 component URDFs into OmniGibson USD models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import traceback

import omnigibson as og
from omnigibson.utils.asset_conversion_utils import (
    convert_urdf_to_usd,
    import_obj_metadata,
    record_obj_metadata_from_urdf,
)
from omnigibson.utils.asset_utils import get_dataset_path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "validation" / "asset_manifest.json"
EXPORT_REPORT_PATH = ROOT / "validation" / "export_validation.json"
REPORT_PATH = ROOT / "validation" / "conversion.json"
DATASET_NAME = "arat-assets-v1"
REPOSITORY_ROOT = ROOT.parents[2]
LEGACY_SPLIT_COMPONENT_CATEGORY = "arat_tube_planks_and_pegs"
SPLIT_COMPONENT_MODELS = {
    "plank_starting_point",
    "plank_target_point",
    "plank_washer_target_point",
    "bolt_large_starting_point",
    "bolt_large_target_point",
    "bolt_small_starting_point",
    "bolt_small_target_point",
    "bolt_washer_target_point",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        help="Convert only these model IDs. Intended for resuming a conversion after a Kit shutdown.",
    )
    args = parser.parse_args()
    export_report = json.loads(EXPORT_REPORT_PATH.read_text(encoding="utf-8"))
    if export_report.get("status") != "passed":
        raise RuntimeError("Run validate_exports.py successfully before conversion")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    dataset_root = Path(get_dataset_path(DATASET_NAME)).resolve()
    expected_dataset_root = REPOSITORY_ROOT / "datasets" / DATASET_NAME
    if dataset_root != expected_dataset_root.resolve():
        raise RuntimeError(f"Unexpected dataset root {dataset_root}; expected {expected_dataset_root.resolve()}")

    prepared = []
    known_models = {asset["model"] for asset in manifest["assets"]}
    requested_models = set(args.models) if args.models else known_models
    unknown_models = requested_models - known_models
    if unknown_models:
        raise ValueError(f"Unknown component models: {sorted(unknown_models)}")
    for asset in manifest["assets"]:
        category = asset["category"]
        model = asset["model"]
        if model not in requested_models:
            continue
        source_urdf = ROOT / "models" / model / f"{model}.urdf"
        model_dir = dataset_root / "objects" / category / model
        if model_dir.exists():
            shutil.rmtree(model_dir)
        model_dir.mkdir(parents=True)
        record_obj_metadata_from_urdf(
            urdf_path=str(source_urdf),
            obj_dir=str(model_dir),
            joint_setting="zero",
            overwrite=True,
        )
        metadata_path = model_dir / "misc" / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["arat_component"] = True
        metadata["arat_asset_version"] = 1
        metadata["arat_authority"] = "ARAT_specification.md"
        metadata["arat_source_unit"] = asset["source_unit"]
        metadata["arat_specs"] = "asset_pipeline/arat_v1/arat_components_specs.yaml"
        metadata["arat_mass_kg"] = asset["mass_kg"]
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        prepared.append((asset, source_urdf, model_dir, metadata_path))

    og.launch()
    if len(og.sim.scenes) != 0:
        raise RuntimeError("Component conversion requires an empty OmniGibson simulator")
    converted = []
    try:
        for index, (asset, source_urdf, model_dir, metadata_path) in enumerate(prepared, start=1):
            category = asset["category"]
            model = asset["model"]
            print(f"[ARAT components] converting {index}/{len(prepared)}: {category}/{model}", flush=True)
            processed_urdf, usd_path = convert_urdf_to_usd(
                urdf_path=str(source_urdf),
                obj_category=category,
                obj_model=model,
                dataset_root=str(dataset_root),
                use_omni_convex_decomp=False,
                use_usda=False,
                merge_fixed_joints=False,
                import_inertia_tensor=True,
            )
            import_obj_metadata(
                usd_path=usd_path,
                obj_category=category,
                obj_model=model,
                dataset_root=str(dataset_root),
            )
            converted.append(
                {
                    "category": category,
                    "model": model,
                    "source_urdf": str(source_urdf),
                    "processed_urdf": str(processed_urdf),
                    "usd": str(usd_path),
                    "metadata": str(metadata_path),
                    "usd_size_bytes": Path(usd_path).stat().st_size,
                }
            )
        # Remove the superseded copies only after every requested conversion
        # has succeeded, so an interrupted category migration remains
        # recoverable from the legacy dataset paths.
        for asset, _, model_dir, _ in prepared:
            model = asset["model"]
            if model not in SPLIT_COMPONENT_MODELS:
                continue
            legacy_dir = dataset_root / "objects" / LEGACY_SPLIT_COMPONENT_CATEGORY / model
            if legacy_dir.is_dir() and legacy_dir.resolve() != model_dir.resolve():
                shutil.rmtree(legacy_dir)
        if args.models:
            print(f"ARAT_COMPONENT_PARTIAL_CONVERSION_PASSED assets={len(converted)}", flush=True)
        else:
            report = {
                "status": "passed",
                "dataset_name": DATASET_NAME,
                "asset_version": 1,
                "authority": "ARAT_specification.md",
                "asset_count": len(converted),
                "assets": converted,
            }
            REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"ARAT_COMPONENT_CONVERSION_PASSED assets={len(converted)} report={REPORT_PATH}", flush=True)
    except BaseException as exc:
        traceback.print_exc()
        REPORT_PATH.with_name("conversion_error.json").write_text(
            json.dumps({"exception": type(exc).__name__, "message": str(exc)}, indent=2) + "\n",
            encoding="utf-8",
        )
        raise
    finally:
        og.shutdown()


if __name__ == "__main__":
    main()
