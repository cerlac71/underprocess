#!/usr/bin/env python3
"""Download and process open IMD Doppler radar sample volumes.

Downloads official-format IMD NetCDF sweep files (Jaipur × 2 volumes, Goa × 1)
and builds ``data/radar/imd_doppler_sample/imd_doppler_rainfall.nc`` for the
pipeline nowcast path until real UP radar data is procured from IMD.

Run from project root:
    python scripts/download_imd_radar_sample.py
    python scripts/download_imd_radar_sample.py --process-only
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from integerations.imd_radar_ingest import (
    IMD_SAMPLE_PROCESSED,
    IMD_SAMPLE_RAW,
    SAMPLE_VOLUMES,
    process_imd_sample_to_netcdf,
    write_sample_manifest,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _sweep_paths(stem: str) -> list[str]:
    return [stem] + [f"{stem}.{i}" for i in range(1, 10)]


def download_volume(session: requests.Session, volume: dict, raw_root: Path) -> list[Path]:
    out_dir = raw_root / volume["subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for name in _sweep_paths(volume["stem"]):
        dest = out_dir / name
        if dest.exists() and dest.stat().st_size > 1000:
            logger.info("Skip existing %s", dest.name)
            saved.append(dest)
            continue
        url = volume["url_base"] + name
        logger.info("Downloading %s", url)
        resp = session.get(url, timeout=180)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        saved.append(dest)
        time.sleep(0.1)

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Download open IMD Doppler radar samples.")
    parser.add_argument(
        "--process-only",
        action="store_true",
        help="Reprocess existing raw sweeps without re-downloading.",
    )
    parser.add_argument(
        "--include-goa-in-processed",
        action="store_true",
        help="Include Goa volume in processed NetCDF (default: Jaipur pair only for nowcast).",
    )
    args = parser.parse_args()

    raw_root = IMD_SAMPLE_RAW
    raw_root.mkdir(parents=True, exist_ok=True)

    if not args.process_only:
        session = requests.Session()
        total_bytes = 0
        for volume in SAMPLE_VOLUMES:
            files = download_volume(session, volume, raw_root)
            total_bytes += sum(f.stat().st_size for f in files)
        logger.info(
            "Downloaded %d sweep files (~%.1f MB raw).",
            sum(len(list((raw_root / v["subdir"]).glob(f"{v['stem']}*"))) for v in SAMPLE_VOLUMES),
            total_bytes / (1024 * 1024),
        )

    manifest = write_sample_manifest(raw_root)
    logger.info("Wrote manifest -> %s", manifest)

    out = process_imd_sample_to_netcdf(
        raw_root=raw_root,
        out_path=IMD_SAMPLE_PROCESSED,
        use_jaipur_nowcast_pair=not args.include_goa_in_processed,
    )
    logger.info("Processed IMD rainfall ready -> %s (%.2f MB)", out, out.stat().st_size / (1024 * 1024))
    logger.info("Set SIH_USE_IMD_RADAR_SAMPLE=1 (default) to use in pipeline nowcast.")


if __name__ == "__main__":
    main()
