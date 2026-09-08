#!/usr/bin/env python3
"""Run holdout flood inundation validation (CSI / F1 vs IMD references).

    python scripts/run_flood_validation.py
    python scripts/run_flood_validation.py --region uttar_pradesh
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config.threading_env import limit_cpu_threads

limit_cpu_threads()

from validation.flood_event_validation import run_all_event_validations

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Flood event inundation validation")
    parser.add_argument("--region", default="uttar_pradesh")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = run_all_event_validations(region_id=args.region, output_path=args.output)
    agg = report.get("aggregate", {})
    if agg:
        print("\n=== Aggregate holdout inundation validation ===")
        print(f"Events validated: {agg.get('n_events')}")
        print(f"Mean CSI (vs satellite observed):  {agg.get('mean_csi_satellite') or 0:.3f}")
        print(f"Mean F1  (vs satellite observed):  {agg.get('mean_f1_satellite') or 0:.3f}")
        print(f"Mean CSI (pluvial vs IMD hydrology): {agg.get('mean_csi_hydrology', 0):.3f}")
        print(f"Mean F1  (pluvial vs IMD hydrology): {agg.get('mean_f1_hydrology', 0):.3f}")
        print(f"Mean CSI (runoff vs IMD):            {agg.get('mean_csi_runoff', 0):.3f}")
        print(f"Mean F1  (runoff vs IMD):            {agg.get('mean_f1_runoff', 0):.3f}")
        print(f"Mean CSI (vs IMD extreme proxy):     {agg.get('mean_csi_proxy', 0):.3f}")
        print(f"Mean peak rainfall MAE:              {agg.get('mean_peak_rainfall_mae_mm', 0):.2f} mm")
        print(f"Mean runoff RMSE:                    {agg.get('mean_runoff_rmse_mm', 0):.2f} mm")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
