"""Flood inundation validation against satellite-observed and IMD references on holdout events."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from config.regions import get_active_region
from data_utils.imd_ingest import load_imd_rainfall
from flood.flood_estimation import classify_risk
from forecasting.model_inference import apply_all_source_corrections, load_correction_bundle, predict_ensemble_grid
from harmonisation.harmonisation import define_target_grid, harmonize_all
from hydrology.runoff_conversion import scs_cn_runoff
from pipeline.pipeline import CONFIG, load_real_data
from validation.contingency_metrics import binary_contingency_metrics
from validation.flood_events import FloodEvent, list_flood_events
from validation.observed_inundation import load_satellite_observed_mask, observations_available

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# IMD 24 h colour thresholds (mm) — used for independent extreme-rain proxy.
IMD_ORANGE_MM = 65.0
IMD_RED_MM = 115.0

# Flood / runoff detection thresholds (coarse 0.25° UP grid).
RUNOFF_THRESHOLD_MM = 10.0
DEPTH_THRESHOLD_M = 0.01  # 10 mm pluvial depth proxy
RISK_THRESHOLD = 1


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _event_data_cfg(event: FloodEvent, antecedent_pad_days: int = 7) -> dict:
    base = dict(CONFIG["data"])
    base["year"] = int(event.peak_date[:4])
    padded_start = pd.Timestamp(event.window_start) - pd.Timedelta(days=antecedent_pad_days)
    base["start_date"] = str(padded_start.date())
    base["end_date"] = event.window_end
    base["antecedent_pad_days"] = antecedent_pad_days
    return base


def _twi_weight(twi: np.ndarray) -> np.ndarray:
    finite = twi[np.isfinite(twi)]
    if finite.size == 0:
        return np.ones_like(twi, dtype=np.float32)
    lo, hi = np.nanpercentile(twi, 5), np.nanpercentile(twi, 95)
    if hi <= lo:
        return np.ones_like(twi, dtype=np.float32)
    norm = np.clip((twi - lo) / (hi - lo), 0.0, 1.0)
    return (0.3 + 0.7 * norm).astype(np.float32)


def _compute_runoff_and_inundation(
    rainfall_mm: np.ndarray,
    terrain: dict[str, np.ndarray],
    cn_grid: np.ndarray,
    antecedent_mm: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert rainfall to runoff (SCS-CN) and local pluvial inundation depth.

    At 0.25° resolution, statewide fill-spill over a single basin dilutes runoff
    to near-zero depth.  We therefore use a cell-scale pluvial proxy:
    depth (m) = runoff (mm) / 1000 × TWI susceptibility weight.
    """
    runoff = scs_cn_runoff(
        rainfall_mm,
        cn_grid,
        ia_factor=CONFIG["scs"]["ia_factor"],
        antecedent_rainfall_mm=antecedent_mm,
    )
    depth = (runoff / 1000.0) * _twi_weight(terrain["twi"])
    risk = classify_risk(depth)
    return runoff, depth, risk


def _reference_proxy_mask(
    imd_day_mm: np.ndarray,
    antecedent_mm: np.ndarray,
    twi: np.ndarray,
    rain_threshold: float = IMD_ORANGE_MM,
    antecedent_threshold: float = 100.0,
    twi_percentile: float = 55.0,
) -> np.ndarray:
    """
    Independent inundation likelihood mask from observed IMD rainfall + terrain TWI.

    Does not use the project's fused rainfall or ML correction — suitable as an
    external reference for pixel-based CSI/F1 when satellite water masks are unavailable.
    """
    twi_cut = np.nanpercentile(twi[np.isfinite(twi)], twi_percentile)
    heavy = imd_day_mm >= rain_threshold
    wet_antecedent = antecedent_mm >= antecedent_threshold
    susceptible = twi >= twi_cut
    return heavy & (wet_antecedent | (imd_day_mm >= IMD_RED_MM)) & susceptible


def _load_rainfall_model_meta(region_id: str) -> dict:
    metrics_path = get_active_region(region_id).model_dir(PROJECT_ROOT) / "multi_source_corrector_metrics.json"
    if not metrics_path.exists():
        return {}
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    ensemble = payload.get("ensemble", {})
    return {
        "model_version": payload.get("model_version"),
        "holdout_r2": ensemble.get("r2"),
        "holdout_mae_mm": ensemble.get("mae_mm"),
        "wet_day_r2": ensemble.get("wet_day_r2"),
    }


def _harmonised_peak_index(harmonised: xr.Dataset, peak_date: str) -> int:
    peak_ts = np.datetime64(peak_date)
    times = harmonised.time.values.astype("datetime64[D]")
    if peak_ts in times:
        return int(np.where(times == peak_ts)[0][0])
    return int(np.argmin(np.abs(times - peak_ts)))


def _ensemble_rainfall_for_day(
    harmonised: xr.Dataset,
    terrain: dict[str, np.ndarray],
    bundle: dict,
    day_index: int,
) -> np.ndarray | None:
    return predict_ensemble_grid(harmonised, terrain, bundle, time_index=day_index)


def _imd_fields_for_event(
    event: FloodEvent,
    target_grid: xr.Dataset,
    data_cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    region = get_active_region()
    bbox = {
        "lat_min": float(target_grid.lat.min()) - data_cfg["bbox_pad_deg"],
        "lat_max": float(target_grid.lat.max()) + data_cfg["bbox_pad_deg"],
        "lon_min": float(target_grid.lon.min()) - data_cfg["bbox_pad_deg"],
        "lon_max": float(target_grid.lon.max()) + data_cfg["bbox_pad_deg"],
    }
    imd = load_imd_rainfall(
        data_cfg["year"],
        data_cfg["imd_dir"],
        bbox,
        event.window_start,
        event.window_end,
    )
    imd_on_grid = imd.interp(lat=target_grid.lat, lon=target_grid.lon, method="nearest")
    peak_ts = np.datetime64(event.peak_date)
    if peak_ts not in imd_on_grid.time.values:
        peak_idx = int(np.argmin(np.abs(imd_on_grid.time.values.astype("datetime64[D]") - peak_ts)))
    else:
        peak_idx = int(np.where(imd_on_grid.time.values == peak_ts)[0][0])

    imd_day = imd_on_grid["rainfall"].isel(time=peak_idx).values.astype(np.float32)
    if peak_idx >= 5:
        antecedent = imd_on_grid["rainfall"].isel(time=slice(peak_idx - 5, peak_idx)).sum("time").values
    else:
        antecedent = imd_on_grid["rainfall"].isel(time=slice(0, peak_idx)).sum("time").values
    return imd_day, antecedent.astype(np.float32), peak_idx


def validate_flood_event(
    event: FloodEvent,
    region_id: str | None = None,
    depth_threshold_m: float = DEPTH_THRESHOLD_M,
    runoff_threshold_mm: float = RUNOFF_THRESHOLD_MM,
) -> dict[str, Any]:
    """
    Validate predicted inundation for one holdout event.

    References
    ----------
    1. **satellite_observed** — JRC Global Surface Water (Landsat 30 m monthly water
       detection), permanent water excluded, resampled to the UP 0.25° grid.
    2. **imd_hydrology** — pluvial inundation from IMD observed daily rainfall
       through SCS-CN + TWI-weighted local depth (process reference).
    3. **imd_runoff** — SCS-CN runoff exceedance (≥ threshold mm) from IMD rainfall.
    4. **imd_extreme_proxy** — independent mask from IMD orange/red thresholds,
       antecedent wetness, and TWI susceptibility.
    """
    region = get_active_region(region_id)
    cfg = region.target_grid
    target_grid = define_target_grid(
        cfg["lat_min"], cfg["lat_max"], cfg["lon_min"], cfg["lon_max"],
        cfg["resolution_km"], grid_step_deg=cfg.get("grid_step_deg"),
    )
    data_cfg = _event_data_cfg(event)
    data_cfg["imd_dir"] = str(region.imd_dir(PROJECT_ROOT))
    data_cfg["chirps_file"] = str(region.chirps_file(data_cfg["year"], PROJECT_ROOT))
    data_cfg["nwp_file"] = str(region.era5_file(data_cfg["year"], PROJECT_ROOT))
    data_cfg["radar_file"] = str(region.merra2_file(data_cfg["year"], PROJECT_ROOT))
    data_cfg["region_name"] = region.name

    saved_cfg = dict(CONFIG["data"])
    CONFIG["data"].update(data_cfg)
    try:
        data = load_real_data(target_grid)
    finally:
        CONFIG["data"].clear()
        CONFIG["data"].update(saved_cfg)
    terrain = data["terrain"]
    cn_grid = terrain["cn"]

    harmonised = harmonize_all(
        satellite_ds=data["satellite"],
        radar_ds=data["radar"],
        nwp_ds=data["nwp"],
        station_df=data["station_df"],
        target_grid=target_grid,
        target_freq="1D",
    )

    bundle = load_correction_bundle(region_id=region.id)
    imd_day, antecedent, peak_idx = _imd_fields_for_event(event, target_grid, data_cfg)

    # Reference: IMD-observed rainfall through identical hydrology
    ref_runoff, ref_depth, ref_risk = _compute_runoff_and_inundation(
        imd_day, terrain, cn_grid, antecedent_mm=antecedent,
    )
    ref_mask_hydro = ref_depth >= depth_threshold_m
    ref_mask_runoff = ref_runoff >= runoff_threshold_mm

    # Reference B: independent extreme-rain + terrain proxy
    ref_mask_proxy = _reference_proxy_mask(imd_day, antecedent, terrain["twi"])

    # Predicted: multi-source ML ensemble rainfall for peak day
    harm_peak_idx = _harmonised_peak_index(harmonised, event.peak_date)
    fused = _ensemble_rainfall_for_day(harmonised, terrain, bundle, harm_peak_idx) if bundle else None
    if fused is None:
        fused = harmonised["satellite_rainfall"].isel(time=harm_peak_idx).values.astype(np.float32)

    rainfall_mae_mm = float(np.nanmean(np.abs(fused - imd_day)))
    rainfall_rmse_mm = float(np.sqrt(np.nanmean((fused - imd_day) ** 2)))

    pred_runoff, pred_depth, pred_risk = _compute_runoff_and_inundation(
        fused, terrain, cn_grid, antecedent_mm=antecedent,
    )
    pred_mask = pred_depth >= depth_threshold_m
    pred_mask_runoff = pred_runoff >= runoff_threshold_mm

    valid = np.isfinite(imd_day) & np.isfinite(pred_depth) & np.isfinite(ref_depth)

    lat = target_grid.lat.values
    lon = target_grid.lon.values
    satellite_meta = None
    metrics_satellite = None
    if observations_available(region.id, PROJECT_ROOT):
        ref_mask_satellite, satellite_meta = load_satellite_observed_mask(
            event.peak_date,
            event.window_start,
            event.window_end,
            lat,
            lon,
            region_id=region.id,
            project_root=PROJECT_ROOT,
        )
        valid_sat = valid & np.isfinite(ref_mask_satellite.astype(float))
        metrics_satellite = binary_contingency_metrics(pred_mask, ref_mask_satellite, valid_sat)
    else:
        logger.warning(
            "Satellite observed inundation not downloaded; run scripts/download_flood_observations.py"
        )

    metrics_hydro = binary_contingency_metrics(pred_mask, ref_mask_hydro, valid)
    metrics_runoff = binary_contingency_metrics(pred_mask_runoff, ref_mask_runoff, valid)
    metrics_proxy = binary_contingency_metrics(pred_mask, ref_mask_proxy, valid)
    metrics_risk = binary_contingency_metrics(
        pred_risk >= RISK_THRESHOLD,
        ref_risk >= RISK_THRESHOLD,
        valid,
    )

    depth_rmse = float(np.sqrt(np.nanmean((pred_depth[valid] - ref_depth[valid]) ** 2)))
    runoff_rmse = float(np.sqrt(np.nanmean((pred_runoff[valid] - ref_runoff[valid]) ** 2)))

    return {
        "event_id": event.id,
        "event_name": event.name,
        "peak_date": event.peak_date,
        "window": [event.window_start, event.window_end],
        "region": region.id,
        "reference": {
            "satellite_observed": (
                "JRC GSW v1.4 Landsat monthly surface water (30 m), "
                "permanent water excluded, 0.25° grid"
            ),
            "imd_hydrology": (
                "SCS-CN runoff + TWI-weighted pluvial depth forced with IMD RF25 daily rainfall"
            ),
            "imd_runoff": f"SCS-CN runoff >= {runoff_threshold_mm} mm from IMD RF25 rainfall",
            "imd_extreme_proxy": (
                f"IMD >= {IMD_ORANGE_MM} mm (or >= {IMD_RED_MM} mm) + antecedent wetness + TWI susceptibility"
            ),
        },
        "thresholds": {
            "inundation_depth_m": depth_threshold_m,
            "runoff_mm": runoff_threshold_mm,
            "risk_class_min": RISK_THRESHOLD,
        },
        "peak_day_imd_mean_mm": float(np.nanmean(imd_day)),
        "peak_day_fused_mean_mm": float(np.nanmean(fused)),
        "peak_day_rainfall_mae_mm": rainfall_mae_mm,
        "peak_day_rainfall_rmse_mm": rainfall_rmse_mm,
        "depth_rmse_m": depth_rmse,
        "runoff_rmse_mm": runoff_rmse,
        "metrics_vs_satellite_observed": metrics_satellite,
        "satellite_reference_meta": satellite_meta,
        "metrics_vs_imd_hydrology": metrics_hydro,
        "metrics_runoff_vs_imd": metrics_runoff,
        "metrics_vs_imd_extreme_proxy": metrics_proxy,
        "metrics_risk_vs_imd_hydrology": metrics_risk,
    }


def run_all_event_validations(
    region_id: str | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    region = get_active_region(region_id)
    events = list_flood_events(region.id)
    results = []
    for event in events:
        logger.info("Validating flood event: %s", event.name)
        try:
            results.append(validate_flood_event(event, region_id=region.id))
        except Exception as exc:
            logger.exception("Event %s failed: %s", event.id, exc)
            results.append({"event_id": event.id, "event_name": event.name, "error": str(exc)})

    summary_rows = []
    for row in results:
        if "error" in row:
            continue
        hydro = row["metrics_vs_imd_hydrology"]
        runoff = row["metrics_runoff_vs_imd"]
        proxy = row["metrics_vs_imd_extreme_proxy"]
        sat = row.get("metrics_vs_satellite_observed") or {}
        summary_rows.append(
            {
                "event": row["event_name"],
                "peak_date": row["peak_date"],
                "csi_satellite": sat.get("csi"),
                "f1_satellite": sat.get("f1"),
                "csi_hydrology": hydro["csi"],
                "f1_hydrology": hydro["f1"],
                "csi_runoff": runoff["csi"],
                "f1_runoff": runoff["f1"],
                "csi_proxy": proxy["csi"],
                "f1_proxy": proxy["f1"],
                "depth_rmse_m": row["depth_rmse_m"],
                "runoff_rmse_mm": row["runoff_rmse_mm"],
                "rainfall_mae_mm": row.get("peak_day_rainfall_mae_mm"),
            }
        )

    aggregate = {}
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        aggregate = {
            "mean_csi_satellite": float(df["csi_satellite"].mean()) if df["csi_satellite"].notna().any() else None,
            "mean_f1_satellite": float(df["f1_satellite"].mean()) if df["f1_satellite"].notna().any() else None,
            "mean_csi_hydrology": float(df["csi_hydrology"].mean()),
            "mean_f1_hydrology": float(df["f1_hydrology"].mean()),
            "mean_csi_runoff": float(df["csi_runoff"].mean()),
            "mean_f1_runoff": float(df["f1_runoff"].mean()),
            "mean_csi_proxy": float(df["csi_proxy"].mean()),
            "mean_f1_proxy": float(df["f1_proxy"].mean()),
            "mean_depth_rmse_m": float(df["depth_rmse_m"].mean()),
            "mean_runoff_rmse_mm": float(df["runoff_rmse_mm"].mean()),
            "mean_peak_rainfall_mae_mm": float(df["rainfall_mae_mm"].mean()) if "rainfall_mae_mm" in df else None,
            "n_events": len(summary_rows),
        }

    report = {
        "region": region.id,
        "region_name": region.name,
        "holdout_year": region.holdout_year,
        "rainfall_model": _load_rainfall_model_meta(region.id),
        "events": results,
        "aggregate": aggregate,
    }

    if output_path is None:
        output_path = region.model_dir(PROJECT_ROOT) / "flood_validation_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_json_safe(report), indent=2),
        encoding="utf-8",
    )
    logger.info("Wrote flood validation report -> %s", output_path)
    return report
