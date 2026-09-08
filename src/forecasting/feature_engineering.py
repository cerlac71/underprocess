"""Shared rainfall feature engineering for training and inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_COLUMNS = ["chirps_mm", "era5_mm", "merra2_mm"]
TERRAIN_COLUMNS = ["elevation_m", "slope_deg", "twi"]
META_COLUMNS = ["lat", "lon", "month", "day_of_year"]
DERIVED_COLUMNS = [
    "source_mean",
    "source_std",
    "source_max",
    "source_range",
    "doy_sin",
    "doy_cos",
    "chirps_lag1_mm",
    "chirps_antecedent_3d",
    "chirps_antecedent_5d",
]
FEATURE_COLUMNS = SOURCE_COLUMNS + TERRAIN_COLUMNS + META_COLUMNS + DERIVED_COLUMNS


def terrain_at_points(lats: np.ndarray, lons: np.ndarray, path: Path) -> np.ndarray:
    """Sample a terrain GeoTIFF at lon/lat points (matches training pipeline)."""
    if not path.exists():
        return np.zeros(len(lats), dtype=np.float64)
    import rasterio

    with rasterio.open(path) as source:
        values = np.asarray([value[0] for value in source.sample(zip(lons, lats))], dtype=float)
        if source.nodata is not None:
            values[np.isclose(values, source.nodata)] = np.nan
    return values


def terrain_stack_for_grid(region, lat: np.ndarray, lon: np.ndarray, project_root: Path) -> dict[str, np.ndarray]:
    """Return elevation/slope/twi on the model grid using point sampling."""
    lon2d, lat2d = np.meshgrid(lon, lat)
    flat_lat = lat2d.ravel()
    flat_lon = lon2d.ravel()
    dem = terrain_at_points(flat_lat, flat_lon, region.dem_file(project_root)).reshape(lat2d.shape)
    slope = terrain_at_points(flat_lat, flat_lon, region.slope_file(project_root)).reshape(lat2d.shape)
    twi = terrain_at_points(flat_lat, flat_lon, region.twi_file(project_root)).reshape(lat2d.shape)
    for arr in (dem, slope, twi):
        np.nan_to_num(arr, copy=False, nan=0.0)
    return {"dem": dem.astype(np.float32), "slope": slope.astype(np.float32), "twi": twi.astype(np.float32)}


def add_temporal_meta(frame: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    """Add month and day-of-year from a datetime column."""
    out = frame.copy()
    ts = pd.to_datetime(out[time_col])
    out["month"] = ts.dt.month
    out["day_of_year"] = ts.dt.dayofyear
    return out


def add_derived_features(
    frame: pd.DataFrame,
    source_columns: list[str] | None = None,
    group_cols: tuple[str, str] = ("lat", "lon"),
    time_col: str = "time",
    add_antecedent: bool = True,
) -> pd.DataFrame:
    """Spatial / multi-source / seasonal features used by the ensemble."""
    out = frame.copy()
    sources = source_columns or SOURCE_COLUMNS
    src = out[sources].to_numpy(dtype=np.float64)
    out["source_mean"] = np.nanmean(src, axis=1)
    out["source_std"] = np.nanstd(src, axis=1)
    out["source_max"] = np.nanmax(src, axis=1)
    out["source_range"] = out["source_max"] - np.nanmin(src, axis=1)

    doy = out["day_of_year"].to_numpy(dtype=np.float64)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    if add_antecedent and time_col in out.columns:
        primary = sources[0]
        out = out.sort_values([*group_cols, time_col]).reset_index(drop=True)
        grouped = out.groupby(list(group_cols), sort=False)[primary]
        out["chirps_lag1_mm"] = grouped.shift(1).fillna(0.0)
        out["chirps_antecedent_3d"] = grouped.transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).sum()
        ).fillna(0.0)
        out["chirps_antecedent_5d"] = grouped.transform(
            lambda s: s.shift(1).rolling(5, min_periods=1).sum()
        ).fillna(0.0)
    else:
        out["chirps_lag1_mm"] = 0.0
        out["chirps_antecedent_3d"] = 0.0
        out["chirps_antecedent_5d"] = 0.0

    return out


def wet_day_sample_weights(imd_mm: np.ndarray) -> np.ndarray:
    """Up-weight rainy days so the model learns extremes, not only dry zeros."""
    weights = np.ones_like(imd_mm, dtype=np.float64)
    weights = np.where(imd_mm >= 2.0, 2.5, weights)
    weights = np.where(imd_mm >= 10.0, 4.0, weights)
    weights = np.where(imd_mm >= 25.0, 6.0, weights)
    return weights


def build_inference_frame(
    source_values: dict[str, np.ndarray],
    terrain: dict[str, np.ndarray],
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    timestamp: pd.Timestamp,
    valid_mask: np.ndarray,
    antecedent: dict[str, float] | None = None,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Vectorised feature matrix for one forecast timestep."""
    antecedent = antecedent or {}
    frame = pd.DataFrame(
        {
            **{col: source_values[col][valid_mask] for col in SOURCE_COLUMNS},
            "elevation_m": terrain["dem"][valid_mask],
            "slope_deg": terrain["slope"][valid_mask],
            "twi": terrain["twi"][valid_mask],
            "lat": lat2d[valid_mask],
            "lon": lon2d[valid_mask],
            "month": timestamp.month,
            "day_of_year": timestamp.dayofyear,
            "chirps_lag1_mm": antecedent.get("chirps_lag1_mm", 0.0),
            "chirps_antecedent_3d": antecedent.get("chirps_antecedent_3d", 0.0),
            "chirps_antecedent_5d": antecedent.get("chirps_antecedent_5d", 0.0),
        }
    )
    frame = add_derived_features(frame, add_antecedent=False)
    cols = feature_columns or FEATURE_COLUMNS
    return frame[cols]
