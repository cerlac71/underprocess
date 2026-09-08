#!/usr/bin/env python3
"""End-to-end health check for SIH26071. Run: PYTHONPATH=src python scripts/e2e_check.py"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("SIH_REGION", "uttar_pradesh")

from config.bootstrap import bootstrap_project

bootstrap_project()

PASS = "✅"
FAIL = "❌"
results: list[tuple[str, bool, str]] = []


def check(name: str, fn) -> None:
    try:
        fn()
        results.append((name, True, ""))
        print(f"{PASS} {name}")
    except Exception as exc:
        results.append((name, False, str(exc)))
        print(f"{FAIL} {name}: {exc}")
        traceback.print_exc()


def main() -> int:
    region = os.environ.get("SIH_REGION", "uttar_pradesh")
    print(f"SIH26071 E2E check — region={region}\n")

    def training_data():
        import subprocess
        r = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts/verify_training_data.py")],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
        )
        assert r.returncode == 0 and "STATUS: OK" in r.stdout

    def pipeline():
        from pipeline.pipeline import run_pipeline
        run_pipeline()
        assert (PROJECT_ROOT / "pipeline_results.nc").exists()
        assert (PROJECT_ROOT / "alerts/district_alerts.json").exists()

    def netcdf_vars():
        import xarray as xr
        with xr.open_dataset(PROJECT_ROOT / "pipeline_results.nc") as ds:
            for v in ("fused_rainfall", "runoff", "flood_depth", "flood_risk", "confidence"):
                assert v in ds

    def district_alerts():
        summary = json.loads((PROJECT_ROOT / "alerts/district_alerts.json").read_text())
        assert summary["total_districts"] == 75
        assert len(summary["districts"]) == 75

    def ml_model():
        p = PROJECT_ROOT / "models/regions/uttar_pradesh/multi_source_corrector.joblib"
        assert p.exists() and p.stat().st_size > 1000

    def location_forecast():
        import xarray as xr
        from ui.location_guide import location_forecast, search_locations
        with xr.open_dataset(PROJECT_ROOT / "pipeline_results.nc") as ds:
            results_dict = {v: ds[v].values for v in ds.data_vars}
            results_dict["lat"] = ds.lat.values
            results_dict["lon"] = ds.lon.values
        loc = search_locations("Lucknow")[0]
        fc = location_forecast(results_dict, loc)
        assert fc["alert_level"] in ("Green", "Yellow", "Orange", "Red")

    def nowcast():
        from ui.page_bootstrap import nowcast_context
        ctx = nowcast_context()
        assert ctx["frames"] >= 2
        assert ctx["lead_hours"] > 0

    def map_viz():
        import xarray as xr
        from ui.location_guide import search_locations
        from ui.map_viz import build_warning_map
        with xr.open_dataset(PROJECT_ROOT / "pipeline_results.nc") as ds:
            r = {v: ds[v].values for v in ds.data_vars}
            r["lat"], r["lon"] = ds.lat.values, ds.lon.values
        loc = search_locations("Varanasi")[0]
        assert build_warning_map(r, selected=loc) is not None

    def alert_monitor():
        import subprocess
        r = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts/run_alert_monitor.py"),
                "--skip-pipeline",
                "--channels",
                "console",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
        )
        assert r.returncode == 0

    def subscriptions():
        from alerts.subscriptions import SubscriptionStore
        from alerts.notification_service import NotificationService, format_summary_message
        path = PROJECT_ROOT / "alerts" / "_e2e_test_subs.json"
        store = SubscriptionStore(path)
        sub = store.add(district="Lucknow", email="e2e@test.local")
        fake = {
            "region": "UP",
            "active_count": 1,
            "total_districts": 75,
            "active_districts": [
                {
                    "district": "Lucknow",
                    "location": "Lucknow",
                    "alert_level": "Yellow",
                    "rainfall_mm": 80,
                    "flood_depth_m": 0.1,
                    "flood_risk": 0,
                    "confidence": 0.7,
                }
            ],
        }
        assert "Lucknow" in format_summary_message(fake)
        assert len(store.find_matching(fake["active_districts"][0])) == 1
        store.remove(sub.id)
        path.unlink(missing_ok=True)
        NotificationService().send_console("e2e ok")

    def flood_validation_report():
        p = PROJECT_ROOT / "models/regions/uttar_pradesh/flood_validation_metrics.json"
        assert p.exists()
        data = json.loads(p.read_text())
        assert "aggregate" in data or "events" in data

    checks = [
        ("Training data present", training_data),
        ("Pipeline run", pipeline),
        ("NetCDF output variables", netcdf_vars),
        ("District alerts (75 districts)", district_alerts),
        ("ML model artefact", ml_model),
        ("Location forecast", location_forecast),
        ("0–6 h nowcast", nowcast),
        ("Map visualization", map_viz),
        ("Alert monitor script", alert_monitor),
        ("Subscriptions + notifications", subscriptions),
        ("Flood validation report", flood_validation_report),
    ]

    for name, fn in checks:
        check(name, fn)

    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"\n{'='*50}")
    print(f"Result: {passed}/{total} checks passed")
    if passed < total:
        print("\nFailed:")
        for name, ok, err in results:
            if not ok:
                print(f"  - {name}: {err}")
        return 1
    print(f"{PASS} All systems operational.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
