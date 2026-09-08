#!/usr/bin/env python3
"""
pipeline.py – End‑to‑end flood warning pipeline for SIH26071.

Orchestrates all modules on real regional data when ``CONFIG["data"]["mode"]`` is
``"real"`` (IMD + CHIRPS + ERA5 + MERRA-2 + terrain + trained bias correctors).
Set ``SIH_REGION=uttar_pradesh`` for the UP production path.
"""

import os
import os
import sys
from pathlib import Path

# Put src/ on the import path so this script runs from any directory without
# needing PYTHONPATH (project modules import each other as top-level packages).
_SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC_ROOT))

from config.bootstrap import bootstrap_project

bootstrap_project()

from config.regions import get_active_region

_region = get_active_region()

import logging
import datetime as dt
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Import project modules (assume they are in the same directory)
from data_utils.data_utils import (
    generate_synthetic_rainfall_series,
    generate_synthetic_grid,
    get_districts_for_region,
)
from data_utils.imd_ingest import load_imd_rainfall, imd_to_station_df
from data_utils.ghcn_ingest import load_ghcn_daily_precipitation
from harmonisation.harmonisation import (
    define_target_grid, harmonize_all, validate_alignment,
    regrid_source, normalize_rainfall_units,
)
from nowcasting.nowcasting import OpticalFlowNowcaster
from forecasting.model_inference import apply_all_source_corrections, load_correction_bundle, predict_ensemble_grid
from fusion.multi_source_fusion import inverse_variance_fusion, source_agreement_diagnostic, compute_confidence
from hydrology.runoff_conversion import scs_cn_runoff, get_curve_number
from flood.flood_estimation import compute_flood_depth_fill_spill, classify_risk
from alerts.alert_scanner import build_alert_summary, save_alert_summary
from alerts.alert_thresholds import AlertThresholdModule
from alerts.lead_time_tagging import AlertWithLeadTime, get_lead_time_bucket
from forecasting.integrated_forecast import build_integrated_forecast
from uncertainty_estimation.uncertainty_estimation import QuantileForecast, nowcast_uncertainty_empirical

def _imd_sample_enabled() -> bool:
    return os.environ.get("SIH_USE_IMD_RADAR_SAMPLE", "1").strip().lower() not in ("0", "false", "no")


def load_imd_doppler_nowcast_radar():
    """Load processed IMD Doppler sample volumes for optical-flow nowcasting."""
    from integerations.imd_radar_ingest import imd_sample_available, load_imd_sample_rainfall

    if not _imd_sample_enabled() or not imd_sample_available():
        return None
    ds = load_imd_sample_rainfall()
    logger.info(
        "Loaded IMD Doppler sample for nowcast (%d frames, %s).",
        ds.sizes.get("time", 0),
        ds.attrs.get("source", "IMD"),
    )
    return ds

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# 1. Configuration
# ----------------------------------------------------------------------
_holdout = _region.holdout_year
if _region.id == "belagavi":
    _start_date, _end_date = "2019-07-20", "2019-08-08"
else:
    # Short monsoon window for state-scale pipeline runs (full season for training)
    _start_date, _end_date = f"{_holdout}-08-01", f"{_holdout}-08-14"

CONFIG = {
    "region_id": _region.id,
    "target_grid": _region.target_grid,
    "data": {
        "mode": "real",  # "real" (IMD gridded rainfall) or "synthetic" (demo)
        "imd_dir": str(_region.imd_dir(PROJECT_ROOT)),
        "chirps_file": str(_region.chirps_file(_holdout, PROJECT_ROOT)),
        "nwp_file": str(_region.era5_file(_holdout, PROJECT_ROOT)),
        "radar_file": str(_region.merra2_file(_holdout, PROJECT_ROOT)),
        "cn_file": str(_region.data_root(PROJECT_ROOT) / "soil" / f"cn_{_region.id}.tif"),
        "year": _holdout,
        "start_date": _start_date,
        "end_date": _end_date,
        "bbox_pad_deg": _region.bbox_pad_deg,
        "region_name": _region.name,
        "model_path": str(_region.model_dir(PROJECT_ROOT) / "multi_source_corrector.joblib"),
        "metrics_path": str(_region.model_dir(PROJECT_ROOT) / "multi_source_corrector_metrics.json"),
    },
    # 0–6 hour optical-flow nowcast (hourly radar frames; see load_hourly_nowcast_radar)
    "nowcasting": {
        "lead_times_min": [60, 120, 180, 240, 300, 360],
        "timestep_min": 60,
        "lookback_hours": 6,
    },
    "fusion": {"max_variance": 10.0},
    "scs": {"ia_factor": 0.2, "default_cn": 79},
    "alerts": {"hysteresis_window": 2, "cooldown_period": 1},
    "output": {"save_plots": True, "save_netcdf": True}
}

# ----------------------------------------------------------------------
# 2. Data loading
# ----------------------------------------------------------------------
def make_terrain(target_grid):
    """
    Synthetic DEM + terrain derivatives on the target grid.
    Replace with real SRTM/Copernicus DEM and soil/land-use layers when available.
    """
    tlat = target_grid["lat"].values
    tlon = target_grid["lon"].values
    lon2d, lat2d = np.meshgrid(tlon, tlat)
    dem = 100 + 20 * np.sin(lon2d * 5) + 10 * np.cos(lat2d * 8)
    return {
        "dem": dem,
        "slope": np.random.uniform(0, 30, (tlat.size, tlon.size)),
        "flow_accum": np.random.randint(1, 1000, (tlat.size, tlon.size)),
        "twi": np.random.uniform(2, 12, (tlat.size, tlon.size)),
        "cn": np.full((tlat.size, tlon.size), CONFIG["scs"]["default_cn"])
    }


def _read_terrain_layer_pil(path: Path, target_shape: tuple[int, int]) -> np.ndarray:
    """Read a pre-gridded GeoTIFF when rasterio is unavailable (same shape as target)."""
    from PIL import Image

    with Image.open(path) as img:
        layer = np.array(img, dtype=np.float32)
    if layer.shape != target_shape:
        raise ValueError(
            f"Terrain layer {path.name} shape {layer.shape} != target {target_shape}; "
            "install rasterio for resampling."
        )
    return np.flipud(layer)


def load_real_terrain(target_grid):
    """Resample regional DEM derivatives onto the model grid."""
    region = get_active_region()
    dem_path = region.dem_file(PROJECT_ROOT)
    if not dem_path.exists():
        logger.warning("Terrain rasters not found for %s; using synthetic terrain.", region.id)
        return make_terrain(target_grid)

    target_shape = (target_grid.sizes["lat"], target_grid.sizes["lon"])
    bounds = (
        float(target_grid.lon.min()), float(target_grid.lat.min()),
        float(target_grid.lon.max()), float(target_grid.lat.max()),
    )

    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.windows import from_bounds
    except ImportError:
        logger.warning("rasterio not installed; reading pre-gridded terrain via PIL.")
        dem = _read_terrain_layer_pil(dem_path, target_shape)
        slope = _read_terrain_layer_pil(region.slope_file(PROJECT_ROOT), target_shape)
        twi = _read_terrain_layer_pil(region.twi_file(PROJECT_ROOT), target_shape)
        flow_path = region.terrain_dir(PROJECT_ROOT) / f"flow_accum_{region.id}.tif"
        flow_accum = (
            _read_terrain_layer_pil(flow_path, target_shape)
            if flow_path.exists()
            else np.full(target_shape, 100.0, dtype=np.float32)
        )
        cn_path = PROJECT_ROOT / CONFIG["data"]["cn_file"]
        if cn_path.exists():
            cn = _read_terrain_layer_pil(cn_path, target_shape)
            cn = np.where(np.isfinite(cn), cn, CONFIG["scs"]["default_cn"])
        else:
            cn = np.full(target_shape, CONFIG["scs"]["default_cn"], dtype=np.float32)
        return {"dem": dem, "slope": slope, "twi": twi, "flow_accum": flow_accum, "cn": cn}

    def read_layer(path: Path, resampling=Resampling.bilinear):
        with rasterio.open(path) as source:
            # The supplied layers and target grid are both WGS84.  Windowed
            # resampling avoids an unnecessary CRS conversion and works in
            # environments where rasterio's PROJ database is unavailable.
            window = from_bounds(*bounds, transform=source.transform)
            layer = source.read(
                1, window=window, out_shape=target_shape, resampling=resampling,
                masked=True,
            ).filled(np.nan).astype(np.float32)
        # GeoTIFF rows are north-to-south; model latitude is south-to-north.
        return np.flipud(layer)

    dem = read_layer(dem_path)
    slope = read_layer(region.slope_file(PROJECT_ROOT))
    twi = read_layer(region.twi_file(PROJECT_ROOT))
    flow_path = region.terrain_dir(PROJECT_ROOT) / f"flow_accum_{region.id}.tif"
    flow_accum = (
        read_layer(flow_path, resampling=Resampling.nearest)
        if flow_path.exists()
        else np.full(target_shape, 100.0, dtype=np.float32)
    )
    cn_path = PROJECT_ROOT / CONFIG["data"]["cn_file"]
    if cn_path.exists():
        cn = read_layer(cn_path, resampling=Resampling.nearest)
        cn = np.where(np.isfinite(cn), cn, CONFIG["scs"]["default_cn"])
    else:
        cn = np.full(target_shape, CONFIG["scs"]["default_cn"], dtype=np.float32)
    return {"dem": dem, "slope": slope, "twi": twi, "flow_accum": flow_accum, "cn": cn}


def load_synthetic_data(target_grid):
    """
    Generate synthetic data for all sources.
    Returns dict with keys: satellite, radar, nwp, station_df, terrain.
    Terrain is generated on target_grid so it aligns with the harmonised fields.
    """
    logger.info("Generating synthetic data...")
    # Create a common time range
    now = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    times = pd.date_range(now - dt.timedelta(hours=6), now, freq='10min')

    # Define a small domain for demonstration
    lat = np.linspace(10, 20, 20)
    lon = np.linspace(70, 80, 20)

    # Synthetic rainfall fields for each source (different biases/noise)
    np.random.seed(42)
    base = np.zeros((len(times), len(lat), len(lon)))
    # Add a moving storm cell
    for t, ts in enumerate(times):
        center_lat = 15 + 0.1 * (t / 10)
        center_lon = 75 + 0.05 * (t / 10)
        for i, la in enumerate(lat):
            for j, lo in enumerate(lon):
                dist = (la - center_lat)**2 + (lo - center_lon)**2
                base[t, i, j] = 30 * np.exp(-dist / 0.3) + np.random.normal(0, 1)

    # Create xarray Datasets for each source
    ds_satellite = xr.Dataset(
        {"rainfall": (("time", "lat", "lon"), base * 1.1 + 1.0)},
        coords={"time": times, "lat": lat, "lon": lon}
    )
    ds_radar = xr.Dataset(
        {"rainfall": (("time", "lat", "lon"), base * 0.9 + 0.5)},
        coords={"time": times, "lat": lat, "lon": lon}
    )
    # NWP on valid times with init_time as an attribute, matching nwp_ingest.py.
    ds_nwp = xr.Dataset(
        {"tp": (("time", "lat", "lon"), base * 0.8 + 2.0)},
        coords={"time": times, "lat": lat, "lon": lon},
        attrs={"init_time": str(times[0])}
    )
    # Station data: sample at a few points
    station_df = pd.DataFrame({
        "timestamp": times,
        "lat": np.random.choice(lat, len(times)),
        "lon": np.random.choice(lon, len(times)),
        "rainfall_mm": base[:, 10, 10] + np.random.normal(0, 2, len(times))
    })

    # Terrain (synthetic DEM) on the target grid
    terrain = make_terrain(target_grid)

    return {
        "satellite": ds_satellite,
        "radar": ds_radar,
        "nwp": ds_nwp,
        "station_df": station_df,
        "terrain": terrain,
        "lat": lat,
        "lon": lon,
        "times": times
    }


def _lat_slice(da_or_ds, lat_min: float, lat_max: float):
    lat = da_or_ds["lat"] if "lat" in da_or_ds.coords else da_or_ds.coords.get("latitude")
    if lat is not None and float(lat[0]) > float(lat[-1]):
        return slice(lat_max, lat_min)
    return slice(lat_min, lat_max)


def _load_optional_dataset(path: Path, default_var: str) -> xr.Dataset | None:
    """Load a NetCDF rainfall dataset if the file exists."""
    if not path.exists():
        logger.warning("Optional dataset not found: %s", path)
        return None
    with xr.open_dataset(path) as raw:
        ds = raw.load()
    if default_var not in ds.data_vars and len(ds.data_vars) == 1:
        ds = ds.rename({next(iter(ds.data_vars)): default_var})
    return ds


def load_real_data(target_grid):
    """
    Load real rainfall, terrain, and open-source NWP/radar products.
    Returns dict with keys: satellite, radar, nwp, station_df, terrain.
    """
    data_cfg = CONFIG["data"]
    tlat = target_grid["lat"].values
    tlon = target_grid["lon"].values
    bbox = {
        "lat_min": float(tlat.min()) - data_cfg["bbox_pad_deg"],
        "lat_max": float(tlat.max()) + data_cfg["bbox_pad_deg"],
        "lon_min": float(tlon.min()) - data_cfg["bbox_pad_deg"],
        "lon_max": float(tlon.max()) + data_cfg["bbox_pad_deg"],
    }
    imd_dir = str(PROJECT_ROOT / data_cfg["imd_dir"])
    logger.info("Loading IMD gridded rainfall (%s, %s to %s)...",
                data_cfg["year"], data_cfg["start_date"], data_cfg["end_date"])
    imd_ds = load_imd_rainfall(
        year=data_cfg["year"],
        data_dir=imd_dir,
        bbox=bbox,
        start_date=data_cfg["start_date"],
        end_date=data_cfg["end_date"],
    )
    satellite_path = PROJECT_ROOT / data_cfg["chirps_file"]
    if not satellite_path.exists():
        raise FileNotFoundError(f"CHIRPS satellite file not found: {satellite_path}")
    with xr.open_dataset(satellite_path) as raw_satellite:
        ds = raw_satellite.sel(
            time=slice(data_cfg["start_date"], data_cfg["end_date"]),
            lat=_lat_slice(raw_satellite, bbox["lat_min"], bbox["lat_max"]),
            lon=slice(bbox["lon_min"], bbox["lon_max"]),
        ).load()
    ds["rainfall"].attrs["units"] = "mm"
    ds.attrs["validation_reference"] = "IMD 0.25 degree daily gridded rainfall"
    ds.attrs["imd_reference_file"] = str(imd_ds.attrs.get("source", data_cfg["imd_dir"]))
    station_dir = PROJECT_ROOT / "data" / "stations" / "ghcn_raw"
    station_df = load_ghcn_daily_precipitation(
        station_dir, data_cfg["start_date"], data_cfg["end_date"]
    )
    if station_df.empty:
        logger.warning("No quality-controlled GHCN daily precipitation was available; "
                       "not substituting synthetic stations.")

    data_cfg = CONFIG["data"]
    nwp_ds = _load_optional_dataset(PROJECT_ROOT / data_cfg["nwp_file"], "tp")
    radar_ds = _load_optional_dataset(PROJECT_ROOT / data_cfg["radar_file"], "rainfall")
    for name, optional_ds in ("nwp", nwp_ds), ("radar", radar_ds):
        if optional_ds is not None and "time" in optional_ds.dims:
            trimmed = optional_ds.sel(
                time=slice(data_cfg["start_date"], data_cfg["end_date"]),
                lat=_lat_slice(optional_ds, bbox["lat_min"], bbox["lat_max"]),
                lon=slice(bbox["lon_min"], bbox["lon_max"]),
            )
            if _region.approx_grid_cells > 200 and trimmed.sizes.get("time", 0) > 48:
                var = "tp" if name == "nwp" else "rainfall"
                if var in trimmed.data_vars:
                    if name == "radar":
                        trimmed = trimmed[[var]].resample(time="1D").mean().rename({var: var})
                        trimmed[var].attrs["units"] = "mm/hr_mean_proxy"
                    else:
                        trimmed = trimmed[[var]].resample(time="1D").sum().rename({var: var})
                else:
                    trimmed = trimmed.resample(time="1D").mean()
            if name == "nwp":
                nwp_ds = trimmed
            else:
                radar_ds = trimmed

    return {
        "satellite": ds,
        "radar": radar_ds,
        "nwp": nwp_ds,
        "station_df": station_df,
        "terrain": load_real_terrain(target_grid),
        "lat": ds["lat"].values,
        "lon": ds["lon"].values,
        "times": ds["time"].values,
    }


def load_hourly_nowcast_radar(target_grid, data_cfg: dict, lookback_hours: int | None = None) -> xr.Dataset | None:
    """Load the last N hourly MERRA-2 frames on the regional grid for 0–6 h nowcasting.

    Fusion/training still uses daily-aggregated radar; this path keeps hourly
    resolution only for optical-flow extrapolation.
    """
    lookback = lookback_hours or CONFIG["nowcasting"].get("lookback_hours", 6)
    radar_path = PROJECT_ROOT / data_cfg["radar_file"]
    if not radar_path.exists():
        logger.warning("Hourly nowcast radar not found: %s", radar_path)
        return None

    tlat = target_grid["lat"].values
    tlon = target_grid["lon"].values
    bbox = {
        "lat_min": float(tlat.min()) - data_cfg["bbox_pad_deg"],
        "lat_max": float(tlat.max()) + data_cfg["bbox_pad_deg"],
        "lon_min": float(tlon.min()) - data_cfg["bbox_pad_deg"],
        "lon_max": float(tlon.max()) + data_cfg["bbox_pad_deg"],
    }

    with xr.open_dataset(radar_path) as raw:
        hourly = raw.sel(
            time=slice(data_cfg["start_date"], data_cfg["end_date"]),
            lat=_lat_slice(raw, bbox["lat_min"], bbox["lat_max"]),
            lon=slice(bbox["lon_min"], bbox["lon_max"]),
        ).load()

    if hourly.sizes.get("time", 0) < 2:
        logger.warning("Insufficient hourly radar frames for nowcast.")
        return None

    end_time = pd.Timestamp(data_cfg["end_date"]) + pd.Timedelta(hours=23)
    if end_time > pd.Timestamp(hourly.time.values[-1]):
        end_time = pd.Timestamp(hourly.time.values[-1])
    start_time = end_time - pd.Timedelta(hours=lookback - 1)
    window = hourly.sel(time=slice(start_time, end_time))
    if window.sizes.get("time", 0) < 2:
        window = hourly.isel(time=slice(-lookback, None))

    var = "rainfall" if "rainfall" in window.data_vars else next(iter(window.data_vars))
    if not (np.allclose(window.lat.values, tlat, rtol=0, atol=1e-6)
            and np.allclose(window.lon.values, tlon, rtol=0, atol=1e-6)):
        regridded = regrid_source(window, target_grid, var, method="bilinear")
        window = regridded.to_dataset(name="rainfall")
    elif var != "rainfall":
        window = window.rename({var: "rainfall"})

    window.attrs.update(
        {
            "source": "NASA POWER MERRA-2 hourly PRECTOTCORR (nowcast frames)",
            "units": "mm/hr",
            "nowcast_lookback_hours": lookback,
            "description": "Hourly rainfall frames for 0–6 h optical-flow nowcast on regional grid.",
        }
    )
    logger.info(
        "Loaded %d hourly nowcast frames (%s to %s).",
        window.sizes["time"],
        pd.Timestamp(window.time.values[0]),
        pd.Timestamp(window.time.values[-1]),
    )
    return window


def resolve_nowcast_source(target_grid, data: dict, harmonised: xr.Dataset, is_real: bool):
    """Pick the finest temporal radar source available for 0–6 h nowcasting."""
    nowcast_cfg = CONFIG["nowcasting"]
    prefer_imd_sample = os.environ.get("SIH_NOWCAST_IMD_SAMPLE", "0").strip().lower() in ("1", "true", "yes")

    if is_real and prefer_imd_sample:
        imd_nowcast = load_imd_doppler_nowcast_radar()
        if imd_nowcast is not None and imd_nowcast.sizes.get("time", 0) >= 2:
            return (
                imd_nowcast.fillna(0.0),
                "IMD Doppler radar (Jaipur sample volumes, official NetCDF Z→R)",
                {"lead_times_min": [10, 20, 30, 60, 120], "timestep_min": 10},
            )

    if is_real:
        hourly = load_hourly_nowcast_radar(target_grid, CONFIG["data"])
        if hourly is not None:
            region_name = CONFIG["data"].get("region_name", get_active_region().name)
            return (
                hourly.fillna(0.0),
                f"MERRA-2 hourly radar proxy on {region_name} grid (0–6 h nowcast)",
                nowcast_cfg,
            )

    if is_real and data.get("radar") is not None:
        radar_raw = data["radar"].fillna(0.0)
        radar_var = "rainfall" if "rainfall" in radar_raw.data_vars else next(iter(radar_raw.data_vars))
        if radar_raw.sizes.get("time", 0) >= 2:
            radar_on_grid = regrid_source(radar_raw, target_grid, radar_var, method="bilinear")
            radar_on_grid = normalize_rainfall_units(radar_on_grid, "radar")
            return (
                radar_on_grid.fillna(0.0).to_dataset(name="rainfall"),
                "radar (regridded)",
                nowcast_cfg,
            )

    nowcast_var = "radar_rainfall" if "radar_rainfall" in harmonised.data_vars else "satellite_rainfall"
    source_da = harmonised[nowcast_var]
    return (
        source_da.fillna(0.0).to_dataset(name="rainfall"),
        f"{nowcast_var} (harmonised fallback)",
        nowcast_cfg,
    )

# ----------------------------------------------------------------------
# 3. Main pipeline
# ----------------------------------------------------------------------
def run_pipeline():
    # Step 1: Define target grid
    cfg = CONFIG["target_grid"]
    target_grid = define_target_grid(
        cfg["lat_min"], cfg["lat_max"],
        cfg["lon_min"], cfg["lon_max"],
        cfg["resolution_km"],
        grid_step_deg=cfg.get("grid_step_deg"),
    )

    # Load data (terrain is generated on the target grid)
    is_real = CONFIG["data"]["mode"] == "real"
    data = load_real_data(target_grid) if is_real else load_synthetic_data(target_grid)

    # Step 2: Harmonise all sources
    logger.info("Harmonising data to common grid...")
    harmonised = harmonize_all(
        satellite_ds=data["satellite"],
        radar_ds=data["radar"],
        nwp_ds=data["nwp"],
        station_df=data["station_df"],
        target_grid=target_grid,
        target_freq='1D' if is_real else '10min'  # daily (real) vs 10-min (synthetic)
    )
    # Validate alignment
    validate_alignment(harmonised, timestep=harmonised.time.values[0])

    # Step 3: Nowcasting (optical flow on hourly radar frames, 0–6 h leads)
    logger.info("Running nowcasting (optical flow, 0–6 h)...")
    nowcaster = OpticalFlowNowcaster(use_decay=True, decay_factor=0.95)
    nowcast_ds, nowcast_var, nowcast_cfg = resolve_nowcast_source(
        target_grid, data, harmonised, is_real
    )
    logger.info("Nowcast input source: %s (%d frames)", nowcast_var, nowcast_ds.sizes.get("time", 0))
    nowcast_lead = nowcast_cfg["lead_times_min"]
    nowcast_maps = nowcaster.nowcast(
        nowcast_ds,
        lead_times_min=nowcast_lead,
        timestep_min=nowcast_cfg["timestep_min"],
    )
    # Convert to QuantileForecast objects (simple empirical uncertainty)
    # For demonstration, we use a simple spread proportional to lead time
    forecast_quantiles = {}
    for lead, fc in nowcast_maps.items():
        spread = 0.1 * lead / 60.0  # 10% per hour
        p10 = np.maximum(fc * (1 - spread), 0)
        p90 = fc * (1 + spread)
        qf = QuantileForecast(
            quantiles={0.1: p10, 0.5: fc, 0.9: p90},
            lead_time=lead,
            source='nowcasting'
        )
        forecast_quantiles[lead] = qf

    # Step 4: ML bias correction (trained multi-source models)
    correction_bundle = load_correction_bundle()
    if correction_bundle is not None and is_real:
        logger.info("Applying trained multi-source bias correction...")
        harmonised = apply_all_source_corrections(harmonised, data["terrain"], correction_bundle)
    elif 'nwp_rainfall' in harmonised.data_vars:
        logger.warning("No trained correction bundle found; using raw NWP rainfall.")
    else:
        logger.warning("No NWP data available for bias correction.")

    # Step 5: Multi‑source fusion
    logger.info("Fusing rainfall estimates...")
    estimates = {}
    variances = {}
    masks = {}

    ensemble_grid = None
    if correction_bundle is not None and is_real:
        ensemble_grid = predict_ensemble_grid(
            harmonised, data["terrain"], correction_bundle, time_index=-1
        )

    if ensemble_grid is not None:
        fused_mean = ensemble_grid
        metrics = correction_bundle.get("metrics", {})
        ensemble_rmse = metrics.get("ensemble", {}).get("rmse_mm", 13.0)
        ensemble_r2 = metrics.get("ensemble", {}).get("r2", 0.25)
        fused_var = np.full_like(fused_mean, ensemble_rmse ** 2)
        weights = {"ensemble_model": np.ones_like(fused_mean)}
        avail_count = np.full_like(fused_mean, 3.0)
        source_arrays = []
        for var in (
            "satellite_rainfall_corrected",
            "nwp_rainfall_corrected",
            "radar_rainfall_corrected",
        ):
            if var in harmonised.data_vars:
                source_arrays.append(harmonised[var].isel(time=-1).values)
        if len(source_arrays) >= 2:
            inter_var = np.nanvar(np.stack(source_arrays, axis=0), axis=0)
        else:
            inter_var = np.zeros_like(fused_mean)
        # Daily rainfall variance scale (mm/day)² — not the hourly default of 10
        daily_var_cap = max(float(ensemble_rmse) ** 2 * 4.0, 80.0)
        confidence = compute_confidence(
            fused_var, avail_count, inter_var, max_variance=daily_var_cap
        )
        model_floor = 0.35 + 0.5 * max(0.0, float(ensemble_r2))
        confidence = np.maximum(confidence, model_floor)
        logger.info("Using trained multi-source ensemble for fused rainfall.")
    else:
        if nowcast_maps:
            lead_min = min(nowcast_lead)
            nowcast_median = forecast_quantiles[lead_min].quantiles[0.5]
            estimates["nowcast"] = nowcast_median
            variances["nowcast"] = np.full_like(nowcast_median, 2.0)
            masks["nowcast"] = np.isfinite(nowcast_median)

        if 'nwp_rainfall_corrected' in harmonised.data_vars:
            nwp_corr = harmonised.nwp_rainfall_corrected.isel(time=-1).values
            estimates['nwp'] = nwp_corr
            variances['nwp'] = np.full_like(nwp_corr, 3.0)
            masks['nwp'] = np.isfinite(nwp_corr)

        if 'radar_rainfall_corrected' in harmonised.data_vars:
            radar_recent = harmonised.radar_rainfall_corrected.isel(time=-1).values
        elif 'radar_rainfall' in harmonised.data_vars:
            radar_recent = harmonised.radar_rainfall.isel(time=-1).values
        else:
            radar_recent = None
        if radar_recent is not None:
            estimates['radar'] = radar_recent
            variances['radar'] = np.full_like(radar_recent, 1.5)
            masks['radar'] = np.isfinite(radar_recent)

        if 'satellite_rainfall_corrected' in harmonised.data_vars:
            sat_recent = harmonised.satellite_rainfall_corrected.isel(time=-1).values
        else:
            sat_recent = harmonised.satellite_rainfall.isel(time=-1).values
        estimates['satellite'] = sat_recent
        variances['satellite'] = np.full_like(sat_recent, 2.5)
        masks['satellite'] = np.isfinite(sat_recent)

        fused_mean, fused_var, weights = inverse_variance_fusion(estimates, variances, masks)
        avail_count, inter_var = source_agreement_diagnostic(estimates, masks)
        daily_cap = max(float(np.nanmean(fused_var)) * 4.0, 80.0)
        confidence = compute_confidence(
            fused_var, avail_count, inter_var, max_variance=daily_cap
        )

    # Step 6: Rainfall → Runoff
    logger.info("Converting rainfall to runoff...")
    # Use SCS‑CN with terrain CN
    cn_grid = data["terrain"]["cn"]
    # For simplicity, use the fused rainfall as input
    runoff = scs_cn_runoff(fused_mean, cn_grid, ia_factor=CONFIG["scs"]["ia_factor"])

    # Step 7: Flood depth estimation (simplified bathtub)
    logger.info("Estimating flood depth...")
    # We need subbasin IDs; for demo, use a single basin.
    dem = data["terrain"]["dem"]
    subbasin_id = np.ones_like(dem, dtype=int)
    # Convert runoff (mm) to volume (m³); cell area follows the target grid
    # resolution (0.1 km -> 100 m x 100 m).
    cell_area = _region.cell_area_m2((cfg["lat_min"] + cfg["lat_max"]) / 2.0)
    runoff_volume = {1: np.sum(runoff * cell_area / 1000.0)}  # mm to m³
    flood_depth = compute_flood_depth_fill_spill(dem, subbasin_id, runoff_volume, cell_area)
    flood_risk = classify_risk(flood_depth)

    # Step 8: Alert generation
    logger.info("Generating alerts...")
    alert_module = AlertThresholdModule(
        terrain_adjustment=True,
        hysteresis_window=CONFIG["alerts"]["hysteresis_window"],
        cooldown_period=CONFIG["alerts"]["cooldown_period"]
    )
    # For a sample region (first district in synthetic mode, configured region in real mode)
    sample_region = get_districts_for_region(_region.id)[0]
    # We'll use the fused rainfall p90 over the region as the rainfall signal
    # Compute region‑specific thresholds using terrain
    terrain_features = {
        'slope': data["terrain"]["slope"].mean(),
        'flow_accum': data["terrain"]["flow_accum"].mean(),
        'twi': data["terrain"]["twi"].mean(),
        'cn': data["terrain"]["cn"].mean()
    }
    # p90 over the region matches the module's rainfall_forecast_p90 semantics;
    # the spatial mean would dilute a localised flood peak.
    region_rainfall = np.nanpercentile(fused_mean, 90)
    # Flood summary over actually flooded cells, not the whole (mostly dry) grid
    flooded_mask = flood_risk >= 1
    flood_frac = float(np.mean(flooded_mask))
    if flood_frac > 0:
        flood_depth_region = float(np.nanmean(flood_depth[flooded_mask]))
        # Highest risk category covering >= 0.5% of cells (avoids single-cell noise)
        region_risk = max((k for k in (3, 2, 1) if np.mean(flood_risk >= k) >= 0.005), default=0)
    else:
        flood_depth_region = 0.0
        region_risk = 0
    # Antecedent rainfall: real 5-day accumulation, or dummy in synthetic mode
    if is_real and 'satellite_rainfall' in harmonised.data_vars:
        antecedent_rainfall = float(
            harmonised.satellite_rainfall.isel(time=slice(-5, -1)).sum('time').mean().values
        )
    else:
        antecedent_rainfall = 20.0
    # Generate alert. Hysteresis needs consecutive timesteps to escalate, so in
    # real mode feed the module the daily rainfall sequence and apply the flood
    # escalation on the final (forecast) day.
    if is_real and 'satellite_rainfall' in harmonised.data_vars:
        hist = harmonised.satellite_rainfall
        times = hist.time.values
        alert_result = None
        for i, t in enumerate(times):
            ts = pd.Timestamp(t)
            ant = float(
                hist.sel(time=slice(ts - pd.Timedelta(days=5), ts - pd.Timedelta(days=1)))
                .sum('time').mean().values
            )
            is_last = i == len(times) - 1
            alert_result = alert_module.compute_alert(
                region_id=0,
                timestamp=ts,
                rainfall_forecast_p90=float(np.nanpercentile(hist.sel(time=t).values, 90)),
                antecedent_rainfall=ant,
                terrain_features=terrain_features,
                flood_depth=flood_depth_region if is_last else None,
                flood_risk=region_risk if is_last else None,
            )
    else:
        alert_result = alert_module.compute_alert(
            region_id=0,
            timestamp=pd.Timestamp.now(),
            rainfall_forecast_p90=region_rainfall,
            antecedent_rainfall=antecedent_rainfall,
            terrain_features=terrain_features,
            flood_depth=flood_depth_region,
            flood_risk=region_risk
        )
    logger.info(f"Alert level: {alert_result.alert_level} "
                f"(rainfall p90={region_rainfall:.1f} mm, flooded cells={100*flood_frac:.1f}%, "
                f"flood depth={flood_depth_region:.2f} m)")

    # Step 9: Lead‑time tagging
    logger.info("Adding lead‑time metadata...")
    # Create an AlertWithLeadTime object (using nowcast lead time)
    issue_time = dt.datetime.now(dt.timezone.utc)
    valid_time = issue_time + dt.timedelta(minutes=int(min(nowcast_lead)))
    alert_with_lead = AlertWithLeadTime(
        issue_time=issue_time,
        valid_time=valid_time,
        source_module='nowcasting',
        alert_level=alert_result.alert_level,
        confidence=float(np.nanmean(confidence)),
        risk_score=float(np.nanmean(flood_risk) * 100 / 3)  # scale to 0-100
    )
    logger.info(f"Lead time bucket: {alert_with_lead.lead_time_bucket}")
    logger.info(f"Alert message: {alert_with_lead.message}")

    # Step 10: Outputs
    logger.info("Generating outputs...")
    # Save plots
    if CONFIG["output"]["save_plots"]:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        # Rainfall
        im0 = axes[0, 0].imshow(fused_mean, origin='lower', cmap='Blues')
        axes[0, 0].set_title('Fused Rainfall (mm)')
        plt.colorbar(im0, ax=axes[0, 0])
        # Runoff
        im1 = axes[0, 1].imshow(runoff, origin='lower', cmap='Greens')
        axes[0, 1].set_title('Runoff (mm)')
        plt.colorbar(im1, ax=axes[0, 1])
        # Flood depth
        im2 = axes[1, 0].imshow(flood_depth, origin='lower', cmap='Reds')
        axes[1, 0].set_title('Flood Depth (m)')
        plt.colorbar(im2, ax=axes[1, 0])
        # Risk
        im3 = axes[1, 1].imshow(flood_risk, origin='lower', cmap='RdYlGn_r', vmin=0, vmax=3)
        axes[1, 1].set_title('Flood Risk (0-3)')
        plt.colorbar(im3, ax=axes[1, 1])
        plt.tight_layout()
        plot_path = PROJECT_ROOT / "pipeline_output.png"
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        logger.info("Plot saved to %s", plot_path)

    if CONFIG["output"]["save_netcdf"]:
        # Save fused rainfall, runoff, flood depth as NetCDF
        ds_out = xr.Dataset(
            {
                'fused_rainfall': (('lat', 'lon'), fused_mean),
                'runoff': (('lat', 'lon'), runoff),
                'flood_depth': (('lat', 'lon'), flood_depth),
                'flood_risk': (('lat', 'lon'), flood_risk),
                'confidence': (('lat', 'lon'), confidence)
            },
            coords={'lat': target_grid['lat'].values, 'lon': target_grid['lon'].values}
        )
        results_path = PROJECT_ROOT / "pipeline_results.nc"
        ds_out.to_netcdf(results_path)
        logger.info("Results saved to %s", results_path)

        # District-level alert summary for dashboard and notification service
        grid_results = {
            "lat": target_grid["lat"].values,
            "lon": target_grid["lon"].values,
            "fused_rainfall": fused_mean,
            "runoff": runoff,
            "flood_depth": flood_depth,
            "flood_risk": flood_risk,
            "confidence": confidence,
        }
        alert_summary = build_alert_summary(
            grid_results,
            region_name=CONFIG["data"]["region_name"] if is_real else sample_region["name"],
        )
        alert_path = PROJECT_ROOT / "alerts" / "district_alerts.json"
        save_alert_summary(alert_summary, alert_path)
        logger.info(
            "District alerts saved to %s (%d active / %d districts)",
            alert_path,
            alert_summary["active_count"],
            alert_summary["total_districts"],
        )

        # SIH26071 integrated forecast NetCDF (all 4 sources + ML + inundation)
        try:
            forecast_ds, forecast_meta = build_integrated_forecast(forecast_outlook_days=7)
            forecast_path = PROJECT_ROOT / "integrated_forecast.nc"
            forecast_ds.to_netcdf(forecast_path)
            logger.info(
                "Integrated forecast saved to %s (issue=%s, %d days)",
                forecast_path,
                forecast_meta.issue_time,
                forecast_meta.n_days,
            )
        except Exception as exc:
            logger.warning("Integrated forecast export failed: %s", exc)

    # Print final alert summary
    print("\n" + "="*60)
    print("FLOOD ALERT SUMMARY")
    print("="*60)
    region_name = CONFIG["data"]["region_name"] if is_real else sample_region['name']
    print(f"Region: {region_name}")
    print(f"Alert Level: {alert_result.alert_level}")
    print(f"Lead Time: {alert_with_lead.lead_time_hours:.1f} hours ({alert_with_lead.lead_time_bucket})")
    print(f"Confidence: {alert_with_lead.confidence:.2f}")
    print(f"Message: {alert_with_lead.message}")
    print("="*60)

    return alert_result, alert_with_lead

if __name__ == "__main__":
    run_pipeline()