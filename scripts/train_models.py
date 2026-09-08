#!/usr/bin/env python3
"""Train all ML models used by the SIH26071 pipeline."""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from config.bootstrap import bootstrap_project

bootstrap_project()

from forecasting.train_actual_data import train as train_chirps_legacy
from forecasting.train_multi_source import train as train_multi_source
from forecasting.train_convlstm_radar import train as train_convlstm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default=None, help="Region: belagavi, uttar_pradesh, india")
    args = parser.parse_args()
    if args.region:
        os.environ["SIH_REGION"] = args.region

    region = args.region or os.environ.get("SIH_REGION", "uttar_pradesh")

    results = {}
    print("=== Multi-source bias correctors (CHIRPS + ERA5 + MERRA-2 -> IMD) ===")
    results["multi_source"] = train_multi_source()

    if region == "belagavi":
        print("\n=== Legacy CHIRPS-only corrector (kept for comparison) ===")
        results["chirps_legacy"] = train_chirps_legacy()

        print("\n=== ConvLSTM nowcast on MERRA-2 hourly proxy ===")
        try:
            results["convlstm"] = train_convlstm(epochs=20)
        except Exception as exc:
            results["convlstm"] = {"error": str(exc)}
    else:
        results["chirps_legacy"] = {"skipped": f"Belagavi-only trainer; not used for {region}"}
        results["convlstm"] = {"skipped": f"Belagavi-only trainer; not used for {region}"}

    summary_path = PROJECT_ROOT / "models" / "regions" / region / "training_summary.json"
    if region == "belagavi":
        summary_path = PROJECT_ROOT / "models" / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved training summary -> {summary_path}")


if __name__ == "__main__":
    main()
