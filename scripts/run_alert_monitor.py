#!/usr/bin/env python3
"""Run the pipeline and send flood/heavy-rain alerts to subscribed channels.

Usage:
    PYTHONPATH=src SIH_REGION=uttar_pradesh python scripts/run_alert_monitor.py
    PYTHONPATH=src python scripts/run_alert_monitor.py --channels console,desktop,email --email you@example.com
    PYTHONPATH=src python scripts/run_alert_monitor.py --skip-pipeline  # use existing results

Schedule with cron (every 30 min during monsoon):
    */30 * * * * cd /path/to/SIH2026 && PYTHONPATH=src SIH_REGION=uttar_pradesh python scripts/run_alert_monitor.py --channels console,desktop,email
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from config.bootstrap import bootstrap_project

bootstrap_project()

import xarray as xr

from alerts.alert_scanner import build_alert_summary, load_alert_summary, save_alert_summary
from alerts.notification_service import NotificationService
from alerts.subscriptions import SubscriptionStore
from config.regions import get_active_region
from pipeline.pipeline import run_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_grid_results(path: Path) -> dict:
    with xr.open_dataset(path) as ds:
        return {
            "lat": ds.lat.values,
            "lon": ds.lon.values,
            "fused_rainfall": ds["fused_rainfall"].values,
            "runoff": ds["runoff"].values,
            "flood_depth": ds["flood_depth"].values,
            "flood_risk": ds["flood_risk"].values,
            "confidence": ds["confidence"].values,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="SIH26071 alert monitor")
    parser.add_argument("--region", default=None, help="Region id (default: SIH_REGION env)")
    parser.add_argument("--skip-pipeline", action="store_true", help="Use existing pipeline_results.nc")
    parser.add_argument(
        "--channels",
        default="console,desktop",
        help="Comma-separated: console,desktop,email,sms,webhook",
    )
    parser.add_argument("--email", default=None, help="Override email recipient")
    parser.add_argument("--sms", default=None, help="Override SMS recipient")
    parser.add_argument("--notify-always", action="store_true", help="Notify even when all Green")
    args = parser.parse_args()

    if args.region:
        os.environ["SIH_REGION"] = args.region

    region = get_active_region()
    results_path = PROJECT_ROOT / "pipeline_results.nc"
    alert_path = PROJECT_ROOT / "alerts" / "district_alerts.json"
    log_path = PROJECT_ROOT / "alerts" / "notification_log.json"
    sub_path = PROJECT_ROOT / "alerts" / "subscriptions.json"

    if not args.skip_pipeline:
        logger.info("Running forecast pipeline for %s…", region.name)
        run_pipeline()

    if not results_path.exists():
        logger.error("No pipeline results at %s", results_path)
        return 1

    results = load_grid_results(results_path)
    summary = build_alert_summary(results, region_name=region.name)
    save_alert_summary(summary, alert_path)

    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    notifier = NotificationService(log_path=log_path)

    active = summary.get("active_count", 0)
    logger.info("Scanned %d districts — %d active alerts", summary["total_districts"], active)

    if active == 0 and not args.notify_always:
        logger.info("No heavy-rain or flood alerts. Skipping notification.")
        notifier.send_console(f"✅ {region.name}: All districts Green — no flood/heavy-rain risk.")
        return 0

    # Broadcast summary
    notifier.notify_summary(
        summary,
        channels=channels,
        email_to=args.email or os.environ.get("SIH_ALERT_EMAIL"),
        sms_to=args.sms or os.environ.get("SIH_ALERT_PHONE"),
    )

    # Per-subscriber delivery
    store = SubscriptionStore(sub_path)
    subs = store.list_all()
    if subs:
        sent = notifier.notify_subscribers(summary, subs, channels=channels)
        logger.info("Subscriber notifications: %s", sent)

    # Persist last run status
    status_path = PROJECT_ROOT / "alerts" / "last_monitor_run.json"
    status_path.write_text(
        json.dumps(
            {
                "region": region.id,
                "active_count": active,
                "channels": channels,
                "summary_path": str(alert_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
