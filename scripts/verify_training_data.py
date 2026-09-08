#!/usr/bin/env python3
"""Verify all raw training / inference data required for SIH26071 UP demo.

Run before submission or when judges ask to see source data:

    python scripts/verify_training_data.py
    python scripts/verify_training_data.py --json report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config.regions import get_active_region

IMD_YEARS_REQUIRED = [2012, 2013, 2014, 2016, 2018, 2019]
TERRAIN_FILES = ("dem", "slope", "twi", "flow_accum")


def _size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024) if path.exists() else 0.0


def verify(region_id: str = "uttar_pradesh") -> dict:
    region = get_active_region(region_id)
    root = PROJECT_ROOT
    manifest_path = region.data_root(root) / "training_manifest.json"
    rows: list[dict] = []
    missing: list[str] = []

    def check(category: str, label: str, path: Path, required: bool = True) -> None:
        ok = path.exists()
        rows.append(
            {
                "category": category,
                "label": label,
                "path": str(path.relative_to(root)),
                "exists": ok,
                "size_mb": round(_size_mb(path), 2),
                "required": required,
            }
        )
        if required and not ok:
            missing.append(str(path.relative_to(root)))

    # Training manifest
    check("manifest", "training_manifest.json", manifest_path)

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for year, files in manifest.get("downloaded", {}).items():
            for source, rel in files.items():
                check("training", f"{source} {year}", root / rel)
        merged = manifest.get("chirps_merged")
        if merged:
            check("training", "CHIRPS merged", root / merged, required=False)

    # IMD reference (shared all-India gridded)
    imd_dir = region.imd_dir(root)
    for year in IMD_YEARS_REQUIRED:
        check("imd", f"IMD RF25 {year}", imd_dir / f"RF25_ind{year}_rfp25.nc")

    # Terrain
    for name in TERRAIN_FILES:
        check("terrain", name, region.terrain_dir(root) / f"{name}_{region.id}.tif")

    # Trained model artefacts
    model_dir = region.model_dir(root)
    check("model", "multi_source_corrector.joblib", model_dir / "multi_source_corrector.joblib")
    check("model", "multi_source_corrector_metrics.json", model_dir / "multi_source_corrector_metrics.json")
    check("model", "training_summary.json", model_dir / "training_summary.json", required=False)

    # Pipeline output (generated, not training raw)
    check("output", "pipeline_results.nc", root / "pipeline_results.nc", required=False)

    # IMD Doppler sample (nowcast integration — optional but recommended for judges)
    imd_sample = root / "data" / "radar" / "imd_doppler_sample" / "imd_doppler_rainfall.nc"
    check("radar", "IMD Doppler sample (processed)", imd_sample, required=False)
    imd_raw = root / "data" / "radar" / "imd_doppler_sample" / "raw" / "jpr"
    if imd_raw.exists():
        n_sweeps = len(list(imd_raw.glob("*-IMD-B.nc*")))
        rows.append(
            {
                "category": "radar",
                "label": "IMD Doppler raw sweeps (Jaipur)",
                "path": str(imd_raw.relative_to(root)),
                "exists": n_sweeps > 0,
                "size_mb": round(sum(p.stat().st_size for p in imd_raw.glob("*")) / (1024 * 1024), 2),
                "required": False,
                "file_count": n_sweeps,
            }
        )

    ok = len(missing) == 0
    return {
        "region": region.id,
        "region_name": region.name,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "ok": ok,
        "missing_required": missing,
        "file_count": len(rows),
        "present_count": sum(1 for r in rows if r["exists"]),
        "total_size_mb": round(sum(r["size_mb"] for r in rows if r["exists"]), 1),
        "files": rows,
        "restore_commands": [
            "python scripts/download_training_years.py --region uttar_pradesh",
            "python scripts/download_region_terrain.py --region uttar_pradesh",
            "PYTHONPATH=src python src/forecasting/train_multi_source.py",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify SIH26071 UP training data inventory")
    parser.add_argument("--region", default="uttar_pradesh")
    parser.add_argument("--json", metavar="PATH", help="Write JSON report")
    args = parser.parse_args()

    report = verify(args.region)
    print(f"Region: {report['region_name']} ({report['region']})")
    print(f"Files: {report['present_count']}/{report['file_count']} present · {report['total_size_mb']} MB")
    if report["ok"]:
        print("STATUS: OK — all required raw training data present")
    else:
        print("STATUS: MISSING required files:")
        for path in report["missing_required"]:
            print(f"  - {path}")
        print("\nRe-download with:")
        for cmd in report["restore_commands"]:
            print(f"  {cmd}")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nReport saved to {args.json}")

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
