#!/usr/bin/env python3
"""Download satellite-observed inundation reference data for flood validation.

Uses JRC Global Surface Water (Landsat monthly water detection, 30 m) for the
Uttar Pradesh bbox. Required before running validation with real observed inundation.

    python scripts/download_flood_observations.py
    python scripts/download_flood_observations.py --region uttar_pradesh
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config.threading_env import limit_cpu_threads

limit_cpu_threads()

from validation.observed_inundation import download_jrc_tiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download JRC GSW observed water masks for UP")
    parser.add_argument("--region", default="uttar_pradesh")
    args = parser.parse_args()

    # Holdout monsoon months covering all catalogued 2019 UP flood events.
    months = [(2019, 7), (2019, 8), (2019, 9)]
    out = download_jrc_tiles(region_id=args.region, years_months=months, project_root=PROJECT_ROOT)
    print(f"\nDownloaded JRC GSW reference tiles -> {out}")
    print("Next: PYTHONPATH=src SIH_REGION=uttar_pradesh python scripts/run_flood_validation.py")


if __name__ == "__main__":
    main()
