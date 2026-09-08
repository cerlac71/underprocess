"""
SIH26071 integrated rainfall + inundation forecast engine.

Aligns with the problem statement:
  - Satellite (CHIRPS)
  - Radar (MERRA-2 / IMD Doppler for nowcast)
  - Observational weather (IMD 0.25° gridded + GHCN stations)
  - Numerical weather prediction (ERA5)
  - AI/ML multi-source bias correction + ensemble fusion
  - Heavy rainfall early warning (IMD colour scale)
  - Inundation prediction (SCS-CN runoff + terrain pluvial model)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from pathlib import Path

from alerts.alert_thresholds import combined_alert_level
from config.regions import get_active_region
from data_utils.imd_ingest import load_imd_rainfall
from forecasting.model_inference import apply_all_source_corrections, load_correction_bundle, predict_ensemble_grid
from fusion.multi_source_fusion import compute_confidence, inverse_variance_fusion, source_agreement_diagnostic
from harmonisation.harmonisation import define_target_grid, harmonize_all, normalize_rainfall_units, regrid_source
from hydrology.runoff_conversion import scs_cn_runoff
from flood.flood_estimation import classify_risk

logger = logging.getLogger(__name__)
_region = get_active_region()
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# SIH26071 lead-time horizons (hours)
LEAD_HORIZONS = {
    "nowcast": (0, 6),       # radar optical-flow
    "short_range": (6, 72),  # NWP + ML daily (1–3 days)
    "medium_range": (72, 168),  # NWP + ML daily (4–7 days)
}


@dataclass
class IntegratedForecastMeta:
    region: str
    issue_time: pd.Timestamp
    n_days: int
    sources: list[str]
    ml_model: str
    period: str


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
    ia_factor: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    runoff = scs_cn_runoff(
        rainfall_mm, cn_grid, ia_factor=ia_factor, antecedent_rainfall_mm=antecedent_mm,
    )
    depth = (runoff / 1000.0) * _twi_weight(terrain["twi"])
    risk = classify_risk(depth)
    return runoff, depth, risk


def _load_imd_observational(target_grid: xr.Dataset, bbox: dict) -> xr.DataArray | None:
    """IMD 0.25° gridded rainfall — observational reference (MoES/IMD)."""
    from pipeline.pipeline import CONFIG

    data_cfg = CONFIG["data"]
    try:
        imd_ds = load_imd_rainfall(
            year=data_cfg["year"],
            data_dir=str(PROJECT_ROOT / data_cfg["imd_dir"]),
            bbox=bbox,
            start_date=data_cfg["start_date"],
            end_date=data_cfg["end_date"],
        )
        imd_da = regrid_source(imd_ds, target_grid, "rainfall", method="conservative")
        imd_da = normalize_rainfall_units(imd_da, "gauge")
        imd_da.attrs["source_type"] = "observational"
        imd_da.attrs["description"] = "IMD 0.25° daily gridded rainfall (observational reference)"
        return imd_da
    except Exception as exc:
        logger.warning("IMD observational load failed: %s", exc)
        return None


def _fuse_day(
    harmonised: xr.Dataset,
    t_idx: int,
    terrain: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-variance fusion across satellite, radar, NWP for one day."""
    estimates: dict[str, np.ndarray] = {}
    variances: dict[str, float] = {"satellite": 2.5, "radar": 1.5, "nwp": 3.0}
    masks: dict[str, np.ndarray] = {}

    for key, raw_var, corr_var in (
        ("satellite", "satellite_rainfall", "satellite_rainfall_corrected"),
        ("radar", "radar_rainfall", "radar_rainfall_corrected"),
        ("nwp", "nwp_rainfall", "nwp_rainfall_corrected"),
    ):
        var = corr_var if corr_var in harmonised else raw_var
        if var not in harmonised:
            continue
        field = harmonised[var].isel(time=t_idx).values
        estimates[key] = field
        masks[key] = np.isfinite(field)

    if not estimates:
        shape = harmonised["satellite_rainfall"].isel(time=0).shape if "satellite_rainfall" in harmonised else (1, 1)
        return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)

    fused, fused_var, _ = inverse_variance_fusion(
        estimates,
        {k: np.full_like(v, variances[k]) for k, v in estimates.items()},
        masks,
    )
    avail_count, inter_var = source_agreement_diagnostic(estimates, masks)
    confidence = compute_confidence(fused_var, avail_count, inter_var, max_variance=80.0)
    return fused.astype(np.float32), confidence.astype(np.float32)


def _antecedent_grid(harmonised: xr.Dataset, t_idx: int, var: str = "observational_rainfall") -> np.ndarray:
    if var not in harmonised or t_idx < 1:
        template = harmonised[list(harmonised.data_vars)[0]].isel(time=0).values
        return np.zeros_like(template, dtype=np.float32)
    start = max(0, t_idx - 5)
    return harmonised[var].isel(time=slice(start, t_idx)).sum("time").values.astype(np.float32)


def build_integrated_forecast(forecast_outlook_days: int = 7) -> tuple[xr.Dataset, IntegratedForecastMeta]:
    """
    Build the SIH26071 integrated daily forecast product on the regional grid.

    Returns per-day grids for all four input sources, AI/ML fused rainfall,
    inundation depth/risk, and IMD early-warning alert level.
    """
    from pipeline.pipeline import CONFIG, load_real_data

    cfg = CONFIG["target_grid"]
    target_grid = define_target_grid(
        cfg["lat_min"], cfg["lat_max"], cfg["lon_min"], cfg["lon_max"],
        cfg["resolution_km"], grid_step_deg=cfg.get("grid_step_deg"),
    )
    data = load_real_data(target_grid)
    tlat, tlon = target_grid["lat"].values, target_grid["lon"].values
    bbox = {
        "lat_min": float(tlat.min()) - CONFIG["data"]["bbox_pad_deg"],
        "lat_max": float(tlat.max()) + CONFIG["data"]["bbox_pad_deg"],
        "lon_min": float(tlon.min()) - CONFIG["data"]["bbox_pad_deg"],
        "lon_max": float(tlon.max()) + CONFIG["data"]["bbox_pad_deg"],
    }

    harmonised = harmonize_all(
        satellite_ds=data["satellite"],
        radar_ds=data["radar"],
        nwp_ds=data["nwp"],
        station_df=data["station_df"],
        target_grid=target_grid,
        target_freq="1D",
    )

    imd_obs = _load_imd_observational(target_grid, bbox)
    if imd_obs is not None:
        harmonised["observational_rainfall"] = imd_obs

    bundle = load_correction_bundle()
    if bundle is not None:
        harmonised = apply_all_source_corrections(harmonised, data["terrain"], bundle)

    times = harmonised.time.values
    n_time = len(times)
    issue_idx = max(0, n_time - forecast_outlook_days - 1)
    issue_time = pd.Timestamp(times[issue_idx])
    lat, lon = harmonised.lat.values, harmonised.lon.values
    shape = (n_time, len(lat), len(lon))

    def _alloc() -> np.ndarray:
        return np.full(shape, np.nan, dtype=np.float32)

    satellite = _alloc()
    radar = _alloc()
    observational = _alloc()
    nwp = _alloc()
    ml_fused = _alloc()
    iv_fused = _alloc()
    runoff = _alloc()
    flood_depth = _alloc()
    flood_risk = _alloc()
    confidence = _alloc()
    phase = np.array(["verification"] * n_time, dtype=object)

    cn_grid = data["terrain"]["cn"]
    terrain = data["terrain"]

    for t_idx in range(n_time):
        if t_idx > issue_idx:
            phase[t_idx] = "forecast"

        def _grab(var: str, out: np.ndarray) -> None:
            if var in harmonised:
                out[t_idx] = harmonised[var].isel(time=t_idx).values.astype(np.float32)

        _grab("satellite_rainfall_corrected", satellite)
        if not np.isfinite(satellite[t_idx]).any():
            _grab("satellite_rainfall", satellite)
        _grab("radar_rainfall_corrected", radar)
        if not np.isfinite(radar[t_idx]).any():
            _grab("radar_rainfall", radar)
        _grab("observational_rainfall", observational)
        _grab("nwp_rainfall_corrected", nwp)
        if not np.isfinite(nwp[t_idx]).any():
            _grab("nwp_rainfall", nwp)

        iv, conf = _fuse_day(harmonised, t_idx, terrain)
        iv_fused[t_idx] = iv
        confidence[t_idx] = conf

        if bundle is not None:
            grid = predict_ensemble_grid(harmonised, terrain, bundle, time_index=t_idx)
            if grid is not None:
                ml_fused[t_idx] = grid
            else:
                ml_fused[t_idx] = iv
        else:
            ml_fused[t_idx] = iv

        ant = _antecedent_grid(harmonised, t_idx)
        ro, depth, risk = _compute_runoff_and_inundation(
            ml_fused[t_idx], terrain, cn_grid, ant,
            ia_factor=CONFIG["scs"]["ia_factor"],
        )
        runoff[t_idx] = ro
        flood_depth[t_idx] = depth
        flood_risk[t_idx] = risk.astype(np.float32)

    ds = xr.Dataset(
        {
            "satellite_rainfall": (("time", "lat", "lon"), satellite),
            "radar_rainfall": (("time", "lat", "lon"), radar),
            "observational_rainfall": (("time", "lat", "lon"), observational),
            "nwp_rainfall": (("time", "lat", "lon"), nwp),
            "ml_fused_rainfall": (("time", "lat", "lon"), ml_fused),
            "iv_fused_rainfall": (("time", "lat", "lon"), iv_fused),
            "runoff_mm": (("time", "lat", "lon"), runoff),
            "flood_depth_m": (("time", "lat", "lon"), flood_depth),
            "flood_risk": (("time", "lat", "lon"), flood_risk),
            "confidence": (("time", "lat", "lon"), confidence),
        },
        coords={"time": times, "lat": lat, "lon": lon, "forecast_phase": ("time", phase)},
        attrs={
            "title": "SIH26071 Integrated Heavy Rainfall & Inundation Forecast",
            "issue_time": str(issue_time),
            "issue_index": issue_idx,
            "forecast_outlook_days": forecast_outlook_days,
            "sources": "satellite,radar,observational,nwp",
            "ml_model": "multi_source_corrector.joblib",
            "period": f"{CONFIG['data']['start_date']} → {CONFIG['data']['end_date']}",
        },
    )

    meta = IntegratedForecastMeta(
        region=_region.name,
        issue_time=issue_time,
        n_days=n_time,
        sources=["Satellite (CHIRPS)", "Radar (MERRA-2)", "Observational (IMD 0.25°)", "NWP (ERA5)"],
        ml_model="Multi-source ML ensemble + inverse-variance fusion",
        period=ds.attrs["period"],
    )
    return ds, meta


def forecast_series_at_point(ds: xr.Dataset, lat: float, lon: float) -> pd.DataFrame:
    """Per-day integrated forecast at one location — all SIH26071 data layers."""
    li = int(np.argmin(np.abs(ds.lat.values - lat)))
    lj = int(np.argmin(np.abs(ds.lon.values - lon)))
    issue_idx = int(ds.attrs.get("issue_index", 0))

    rows: list[dict[str, Any]] = []
    for t_idx, t in enumerate(ds.time.values):
        rain = float(ds["ml_fused_rainfall"].isel(time=t_idx, lat=li, lon=lj).values)
        depth = float(ds["flood_depth_m"].isel(time=t_idx, lat=li, lon=lj).values)
        risk = int(round(float(ds["flood_risk"].isel(time=t_idx, lat=li, lon=lj).values)))
        phase = str(ds["forecast_phase"].values[t_idx])
        lead_days = max(0, t_idx - issue_idx) if phase == "forecast" else 0
        rows.append(
            {
                "date": pd.Timestamp(t),
                "phase": phase,
                "lead_days": lead_days,
                "satellite_mm": float(ds["satellite_rainfall"].isel(time=t_idx, lat=li, lon=lj).values),
                "radar_mm": float(ds["radar_rainfall"].isel(time=t_idx, lat=li, lon=lj).values),
                "observational_mm": float(ds["observational_rainfall"].isel(time=t_idx, lat=li, lon=lj).values),
                "nwp_mm": float(ds["nwp_rainfall"].isel(time=t_idx, lat=li, lon=lj).values),
                "ml_fused_mm": rain,
                "iv_fused_mm": float(ds["iv_fused_rainfall"].isel(time=t_idx, lat=li, lon=lj).values),
                "runoff_mm": float(ds["runoff_mm"].isel(time=t_idx, lat=li, lon=lj).values),
                "flood_depth_m": depth,
                "flood_risk": risk,
                "confidence": float(ds["confidence"].isel(time=t_idx, lat=li, lon=lj).values),
                "alert_level": combined_alert_level(rain, risk, depth),
            }
        )
    return pd.DataFrame(rows)


def outlook_summary(df: pd.DataFrame, days: int = 7, phase: str | None = "forecast") -> dict:
    """Summarise integrated outlook for forecast phase days."""
    if phase:
        subset = df[df["phase"] == phase].tail(days)
        if subset.empty:
            subset = df.tail(days)
    else:
        subset = df.tail(days)

    return {
        "days": len(subset),
        "phase": phase or "all",
        "total_ml_mm": float(subset["ml_fused_mm"].sum()),
        "total_nwp_mm": float(subset["nwp_mm"].sum()),
        "total_observed_mm": float(subset["observational_mm"].sum()),
        "peak_rainfall_mm": float(subset["ml_fused_mm"].max()) if not subset.empty else 0.0,
        "peak_flood_depth_m": float(subset["flood_depth_m"].max()) if not subset.empty else 0.0,
        "peak_alert": subset.loc[subset["ml_fused_mm"].idxmax(), "alert_level"] if not subset.empty else "Green",
        "peak_flood_risk": int(subset["flood_risk"].max()) if not subset.empty else 0,
        "daily": subset.to_dict("records"),
    }
