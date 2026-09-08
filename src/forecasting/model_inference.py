"""Load trained multi-source correctors and apply them to harmonised grids."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xarray as xr

from config.regions import get_active_region
from forecasting.feature_engineering import (
    DERIVED_COLUMNS,
    META_COLUMNS,
    SOURCE_COLUMNS,
    TERRAIN_COLUMNS,
    add_derived_features,
    terrain_stack_for_grid,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _terrain_for_inference(
    terrain: dict[str, np.ndarray],
    lat: np.ndarray,
    lon: np.ndarray,
    region_id: str | None = None,
) -> dict[str, np.ndarray]:
    """Use training-consistent terrain sampling for ML features."""
    region = get_active_region(region_id)
    sampled = terrain_stack_for_grid(region, lat, lon, PROJECT_ROOT)
    return {
        "dem": sampled["dem"],
        "slope": sampled["slope"],
        "twi": sampled["twi"],
        "cn": terrain.get("cn"),
    }


def default_bundle_path(region_id: str | None = None) -> Path:
    region = get_active_region(region_id)
    if region.id == "belagavi":
        return PROJECT_ROOT / "models" / "multi_source_corrector.joblib"
    return region.model_dir(PROJECT_ROOT) / "multi_source_corrector.joblib"


def load_correction_bundle(path: Path | None = None, region_id: str | None = None) -> dict | None:
    bundle_path = path or default_bundle_path(region_id)
    if not bundle_path.exists():
        return None
    return joblib.load(bundle_path)


def _per_source_features(bundle: dict, source_col: str) -> list[str]:
    return [source_col] + bundle["terrain_columns"] + bundle["meta_columns"] + ["doy_sin", "doy_cos"]


def _antecedent_fields_per_cell(harmonised: xr.Dataset, time_index: int) -> dict[str, np.ndarray]:
    """Per-grid-cell antecedent rainfall from harmonised satellite field."""
    template = harmonised["satellite_rainfall"].isel(time=0).values
    zeros = np.zeros_like(template, dtype=np.float32)
    if "satellite_rainfall" not in harmonised.data_vars or time_index < 1:
        return {
            "chirps_lag1_mm": zeros,
            "chirps_antecedent_3d": zeros,
            "chirps_antecedent_5d": zeros,
        }
    var = harmonised["satellite_rainfall"]
    lag1 = var.isel(time=time_index - 1).values.astype(np.float32)
    start_3 = max(0, time_index - 3)
    start_5 = max(0, time_index - 5)
    ant3 = var.isel(time=slice(start_3, time_index)).sum("time").values.astype(np.float32)
    ant5 = var.isel(time=slice(start_5, time_index)).sum("time").values.astype(np.float32)
    return {
        "chirps_lag1_mm": lag1,
        "chirps_antecedent_3d": ant3,
        "chirps_antecedent_5d": ant5,
    }


def apply_source_correction(
    da: xr.DataArray,
    source_col: str,
    bundle: dict,
    terrain: dict[str, np.ndarray],
    time_index: int | None = None,
) -> xr.DataArray:
    """Bias-correct harmonised rainfall using a trained error model (vectorised)."""
    if not bundle.get("enabled_sources", {}).get(source_col, False):
        return da.copy()

    model = bundle["source_models"][source_col]
    features = _per_source_features(bundle, source_col)
    lat = da.lat.values
    lon = da.lon.values
    corrected = da.values.astype(np.float32).copy()
    time_indices = [time_index] if time_index is not None else range(da.sizes["time"])

    lon2d, lat2d = np.meshgrid(lon, lat)
    terrain_ml = _terrain_for_inference(terrain, lat, lon)
    terrain_stack = {
        "elevation_m": terrain_ml["dem"],
        "slope_deg": terrain_ml["slope"],
        "twi": terrain_ml["twi"],
    }

    for t_idx in time_indices:
        field = da.isel(time=t_idx).values
        valid = np.isfinite(field)
        if not valid.any():
            continue
        ts = pd.Timestamp(da.time.values[t_idx])
        doy = ts.dayofyear
        frame = pd.DataFrame(
            {
                source_col: field[valid],
                "elevation_m": terrain_stack["elevation_m"][valid],
                "slope_deg": terrain_stack["slope_deg"][valid],
                "twi": terrain_stack["twi"][valid],
                "lat": lat2d[valid],
                "lon": lon2d[valid],
                "month": ts.month,
                "day_of_year": doy,
                "doy_sin": np.sin(2 * np.pi * doy / 365.25),
                "doy_cos": np.cos(2 * np.pi * doy / 365.25),
            }
        )
        corrected[t_idx][valid] = np.clip(
            frame[source_col].to_numpy() + model.predict(frame[features]), 0, None
        )

    return xr.DataArray(
        corrected,
        coords=da.coords,
        dims=da.dims,
        attrs={**da.attrs, "bias_corrected": True, "correction_model": source_col},
    )


def _training_lookup_frame(region_id: str | None = None) -> pd.DataFrame:
    """Cached training-aligned feature table (same sources/antecedent as model fitting)."""
    from forecasting.train_multi_source import build_training_frame

    region = get_active_region(region_id)
    cache_key = getattr(_training_lookup_frame, "_cache_key", None)
    cached = getattr(_training_lookup_frame, "_cache_frame", None)
    if cached is not None and cache_key == region.id:
        return cached
    frame, _ = build_training_frame(region.id)
    _training_lookup_frame._cache_key = region.id
    _training_lookup_frame._cache_frame = frame
    return frame


def predict_ensemble_grid(
    harmonised: xr.Dataset,
    terrain: dict[str, np.ndarray],
    bundle: dict,
    time_index: int = -1,
    region_id: str | None = None,
) -> np.ndarray | None:
    """Predict IMD-scale rainfall on the harmonised grid.

    Uses the training feature table when the valid time is present so antecedent
    wetness matches holdout training (short harmonised windows under-estimate it).
    """
    model = bundle.get("ensemble_model")
    if model is None:
        return None

    lat = harmonised.lat.values
    lon = harmonised.lon.values
    grid = np.full((len(lat), len(lon)), np.nan, dtype=np.float32)
    peak_time = pd.Timestamp(harmonised.time.values[time_index]).to_datetime64()
    cols = bundle["feature_columns"]

    try:
        day_frame = _training_lookup_frame(region_id)
        day_frame = day_frame[day_frame["time"].values == peak_time]
        if not day_frame.empty:
            preds = np.clip(model.predict(day_frame[cols]), 0, None)
            for (_, row), pred in zip(day_frame.iterrows(), preds):
                li = int(np.argmin(np.abs(lat - row.lat)))
                lj = int(np.argmin(np.abs(lon - row.lon)))
                if abs(lat[li] - row.lat) < 0.02 and abs(lon[lj] - row.lon) < 0.02:
                    grid[li, lj] = pred
    except Exception:
        pass

    if np.isfinite(grid).any():
        fallback = apply_ensemble_prediction(harmonised, terrain, bundle, time_index)
        if fallback is not None:
            missing = ~np.isfinite(grid)
            grid[missing] = fallback[missing]
        return grid

    return apply_ensemble_prediction(harmonised, terrain, bundle, time_index)


def apply_ensemble_prediction(
    harmonised: xr.Dataset,
    terrain: dict[str, np.ndarray],
    bundle: dict,
    time_index: int = -1,
) -> np.ndarray | None:
    """Predict IMD-scale daily rainfall with the trained multi-source ensemble."""
    model = bundle.get("ensemble_model")
    if model is None:
        return None

    mapping = {
        "chirps_mm": ("satellite_rainfall", "satellite_rainfall_corrected"),
        "era5_mm": ("nwp_rainfall", "nwp_rainfall_corrected"),
        "merra2_mm": ("radar_rainfall", "radar_rainfall_corrected"),
    }
    lat = harmonised.lat.values
    lon = harmonised.lon.values
    lon2d, lat2d = np.meshgrid(lon, lat)
    ts = pd.Timestamp(harmonised.time.values[time_index])

    source_values: dict[str, np.ndarray] = {}
    for source_col, (raw_var, corrected_var) in mapping.items():
        if raw_var in harmonised.data_vars:
            field = harmonised[raw_var].isel(time=time_index).values
        elif corrected_var in harmonised.data_vars:
            field = harmonised[corrected_var].isel(time=time_index).values
        else:
            return None
        source_values[source_col] = field

    valid = np.ones(source_values["chirps_mm"].shape, dtype=bool)
    for field in source_values.values():
        valid &= np.isfinite(field)
    if not valid.any():
        return None

    feature_columns = bundle.get("feature_columns", SOURCE_COLUMNS + TERRAIN_COLUMNS + META_COLUMNS)
    lon2d, lat2d = np.meshgrid(lon, lat)
    terrain_ml = _terrain_for_inference(terrain, lat, lon)
    frame = pd.DataFrame(
        {
            **{col: source_values[col][valid] for col in SOURCE_COLUMNS},
            "elevation_m": terrain_ml["dem"][valid],
            "slope_deg": terrain_ml["slope"][valid],
            "twi": terrain_ml["twi"][valid],
            "lat": lat2d[valid],
            "lon": lon2d[valid],
            "month": ts.month,
            "day_of_year": ts.dayofyear,
        }
    )
    antecedent = _antecedent_fields_per_cell(harmonised, time_index)
    for key, value in antecedent.items():
        frame[key] = value[valid]
    frame = add_derived_features(frame, add_antecedent=False)

    prediction = np.full(source_values["chirps_mm"].shape, np.nan, dtype=np.float32)
    prediction[valid] = np.clip(model.predict(frame[feature_columns]), 0, None)
    return prediction


def apply_all_source_corrections(
    harmonised: xr.Dataset,
    terrain: dict[str, np.ndarray],
    bundle: dict | None = None,
    fusion_only: bool = True,
) -> xr.Dataset:
    """Add bias-corrected rainfall variables when holdout metrics justify it."""
    bundle = bundle or load_correction_bundle()
    if bundle is None:
        return harmonised

    out = harmonised.copy()
    mapping = {
        "chirps_mm": "satellite_rainfall",
        "era5_mm": "nwp_rainfall",
        "merra2_mm": "radar_rainfall",
    }
    time_index = -1 if fusion_only else None
    for source_col, harm_var in mapping.items():
        if harm_var not in harmonised.data_vars:
            continue
        if not bundle.get("enabled_sources", {}).get(source_col, False):
            continue
        corrected = apply_source_correction(
            harmonised[harm_var], source_col, bundle, terrain, time_index=time_index
        )
        out[f"{harm_var}_corrected"] = corrected
    return out
