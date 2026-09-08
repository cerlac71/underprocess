"""Train and persist multi-source rainfall bias-correction models.

Fits per-source correctors (CHIRPS, ERA5, MERRA-2) plus a fused ensemble
against IMD daily rainfall for the active region (SIH_REGION env var).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC_ROOT))
from config.bootstrap import bootstrap_project

bootstrap_project()

import joblib
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config.regions import get_active_region
from data_utils.imd_ingest import discover_training_years, load_imd_rainfall_multi_year
from forecasting.feature_engineering import (
    DERIVED_COLUMNS,
    META_COLUMNS,
    SOURCE_COLUMNS,
    TERRAIN_COLUMNS,
    add_derived_features,
    add_temporal_meta,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MONSOON_ONLY = True

ENSEMBLE_FEATURES = SOURCE_COLUMNS + TERRAIN_COLUMNS + META_COLUMNS + DERIVED_COLUMNS
PER_SOURCE_FEATURES = SOURCE_COLUMNS + TERRAIN_COLUMNS + META_COLUMNS + ["doy_sin", "doy_cos"]


def _model_paths(region_id: str | None = None) -> tuple[Path, Path]:
    region = get_active_region(region_id)
    model_dir = region.model_dir(PROJECT_ROOT)
    model_dir.mkdir(parents=True, exist_ok=True)
    if region.id == "belagavi":
        return (
            PROJECT_ROOT / "models" / "multi_source_corrector.joblib",
            PROJECT_ROOT / "models" / "multi_source_corrector_metrics.json",
        )
    return (
        model_dir / "multi_source_corrector.joblib",
        model_dir / "multi_source_corrector_metrics.json",
    )


def _terrain_at_points(lats: np.ndarray, lons: np.ndarray, path: Path) -> np.ndarray:
    if not path.exists():
        return np.full(len(lats), np.nan)
    with rasterio.open(path) as source:
        values = np.asarray([value[0] for value in source.sample(zip(lons, lats))], dtype=float)
        if source.nodata is not None:
            values[np.isclose(values, source.nodata)] = np.nan
    return values


def _daily_aggregate(ds: xr.Dataset, var: str, source: str) -> xr.DataArray:
    """Aggregate hourly fields to daily totals compatible with IMD (mm/day)."""
    da = ds[var]
    if source == "merra2":
        # NASA POWER hourly PRECTOTCORR: use daily mean as intensity proxy (see data/README).
        daily = da.resample(time="1D").mean()
        daily.attrs["units"] = "mm/hr_mean_proxy"
    else:
        daily = da.resample(time="1D").sum()
        daily.attrs["units"] = "mm/day"
    return daily


def _open_yearly_stack(paths: list[Path], var: str) -> xr.DataArray:
    existing = [p for p in paths if p.exists()]
    if not existing:
        raise FileNotFoundError(f"No files among: {paths}")
    if len(existing) == 1:
        with xr.open_dataset(existing[0]) as ds:
            return ds[var].load()
    arrays = []
    for path in existing:
        with xr.open_dataset(path) as ds:
            arrays.append(ds[var].load())
    return xr.concat(arrays, dim="time").sortby("time")


def build_training_frame(region_id: str | None = None) -> tuple[pd.DataFrame, list[int]]:
    """Pair all rainfall sources with IMD analysis on a common daily grid."""
    region = get_active_region(region_id)
    years = discover_training_years(region_id=region.id)
    if not years:
        raise FileNotFoundError(
            f"No complete training years for region '{region.id}'. "
            f"Run: python scripts/download_training_years.py --region {region.id}"
        )

    imd = load_imd_rainfall_multi_year(
        years, region.imd_dir(PROJECT_ROOT), region.bbox, monsoon_only=MONSOON_ONLY
    )
    target = imd["rainfall"]

    chirps_paths = [region.chirps_file(y, PROJECT_ROOT) for y in years]
    era5_paths = [region.era5_file(y, PROJECT_ROOT) for y in years]
    merra_paths = [region.merra2_file(y, PROJECT_ROOT) for y in years]

    chirps = _open_yearly_stack(chirps_paths, "rainfall").interp(lat=imd.lat, lon=imd.lon, method="linear")

    era5_hourly = _open_yearly_stack(era5_paths, "tp").to_dataset(name="tp")
    era5 = (
        _daily_aggregate(era5_hourly, "tp", "era5")
        .interp(lat=imd.lat, lon=imd.lon, method="linear")
        .sel(time=target.time, method="nearest")
    )

    merra_hourly = _open_yearly_stack(merra_paths, "rainfall").to_dataset(name="rainfall")
    merra2 = (
        _daily_aggregate(merra_hourly, "rainfall", "merra2")
        .interp(lat=imd.lat, lon=imd.lon, method="linear")
        .sel(time=target.time, method="nearest")
    )

    frame = xr.Dataset(
        {
            "imd_mm": target,
            "chirps_mm": chirps,
            "era5_mm": era5,
            "merra2_mm": merra2,
        }
    ).to_dataframe().reset_index()
    frame = frame.dropna(subset=["imd_mm", "chirps_mm", "era5_mm", "merra2_mm"]).copy()

    points = frame[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
    terrain_files = [
        ("elevation_m", region.dem_file(PROJECT_ROOT)),
        ("slope_deg", region.slope_file(PROJECT_ROOT)),
        ("twi", region.twi_file(PROJECT_ROOT)),
    ]
    for name, tpath in terrain_files:
        points[name] = _terrain_at_points(points.lat.to_numpy(), points.lon.to_numpy(), tpath)
        if not tpath.exists():
            points[name] = 0.0
    frame = frame.merge(points, on=["lat", "lon"], how="left")
    for col in TERRAIN_COLUMNS:
        frame[col] = frame[col].fillna(0.0)
    frame = add_temporal_meta(frame)
    frame = add_derived_features(frame)
    required = ENSEMBLE_FEATURES + ["imd_mm"]
    return (
        frame.dropna(subset=required).sort_values("time").reset_index(drop=True),
        years,
    )


def _fit_error_model(train_df: pd.DataFrame, source_col: str) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        max_iter=180,
        max_leaf_nodes=16,
        l2_regularization=1.0,
        learning_rate=0.08,
        random_state=42,
    )
    features = [source_col] + TERRAIN_COLUMNS + META_COLUMNS + ["doy_sin", "doy_cos"]
    target = train_df["imd_mm"] - train_df[source_col]
    model.fit(train_df[features], target)
    return model


def _evaluate(actual: np.ndarray, predicted: np.ndarray) -> dict:
    wet = actual >= 2.0
    metrics = {
        "mae_mm": float(mean_absolute_error(actual, predicted)),
        "rmse_mm": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
    }
    if wet.sum() >= 20:
        metrics["wet_day_r2"] = float(r2_score(actual[wet], predicted[wet]))
        metrics["wet_day_mae_mm"] = float(mean_absolute_error(actual[wet], predicted[wet]))
    return metrics


def train(region_id: str | None = None) -> dict:
    region = get_active_region(region_id)
    bundle_path, metrics_path = _model_paths(region.id)
    frame, years = build_training_frame(region.id)
    holdout_start = pd.Timestamp(f"{region.holdout_year}-06-01")
    train_df = frame[frame.time < holdout_start].copy()
    test_df = frame[frame.time >= holdout_start].copy()
    if len(train_df) < 100 or len(test_df) < 20:
        raise ValueError(
            f"Insufficient samples (train={len(train_df)}, test={len(test_df)}). "
            "Download more years via scripts/download_training_years.py"
        )
    actual = test_df["imd_mm"].to_numpy()

    source_models: dict[str, HistGradientBoostingRegressor] = {}
    source_metrics: dict[str, dict] = {}

    for source_col in SOURCE_COLUMNS:
        model = _fit_error_model(train_df, source_col)
        features = [source_col] + TERRAIN_COLUMNS + META_COLUMNS + ["doy_sin", "doy_cos"]
        error_pred = model.predict(test_df[features])
        corrected = np.clip(test_df[source_col].to_numpy() + error_pred, 0, None)
        source_models[source_col] = model
        source_metrics[source_col] = {
            "raw": _evaluate(actual, test_df[source_col].to_numpy()),
            "corrected": _evaluate(actual, corrected),
        }

    ensemble = HistGradientBoostingRegressor(
        max_iter=280,
        max_leaf_nodes=24,
        min_samples_leaf=35,
        l2_regularization=0.6,
        learning_rate=0.05,
        random_state=42,
    )
    ensemble.fit(train_df[ENSEMBLE_FEATURES], train_df["imd_mm"])
    ensemble_pred = np.clip(ensemble.predict(test_df[ENSEMBLE_FEATURES]), 0, None)

    enabled_sources = {
        source_col: source_metrics[source_col]["corrected"]["mae_mm"]
        < source_metrics[source_col]["raw"]["mae_mm"]
        for source_col in SOURCE_COLUMNS
    }

    metrics = {
        "region": region.id,
        "region_name": region.name,
        "training_samples": int(len(train_df)),
        "test_samples": int(len(test_df)),
        "training_years": [y for y in years if y < region.holdout_year],
        "holdout_year": region.holdout_year,
        "training_end_exclusive": str(holdout_start.date()),
        "test_period": [str(pd.Timestamp(test_df.time.min()).date()), str(pd.Timestamp(test_df.time.max()).date())],
        "per_source": source_metrics,
        "ensemble": _evaluate(actual, ensemble_pred),
        "best_single_source_corrected": min(
            source_metrics.items(), key=lambda item: item[1]["corrected"]["mae_mm"]
        )[0],
        "enabled_sources": enabled_sources,
        "feature_columns": ENSEMBLE_FEATURES,
        "model_version": 2,
        "caveat": (
            f"Trained on {len(years)} monsoon seasons ({years[0]}-{years[-1]}) "
            f"for {region.name}; holdout year {region.holdout_year}. "
            "Ensemble uses raw multi-source inputs with antecedent wetness and seasonal features."
        ),
    }

    bundle_path.parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(
        {
            "source_models": source_models,
            "ensemble_model": ensemble,
            "source_columns": SOURCE_COLUMNS,
            "terrain_columns": TERRAIN_COLUMNS,
            "meta_columns": META_COLUMNS,
            "derived_columns": DERIVED_COLUMNS,
            "feature_columns": ENSEMBLE_FEATURES,
            "per_source_features": PER_SOURCE_FEATURES,
            "enabled_sources": enabled_sources,
            "region": region.id,
            "model_version": 2,
            "source_to_harmonised": {
                "chirps_mm": "satellite_rainfall",
                "era5_mm": "nwp_rainfall",
                "merra2_mm": "radar_rainfall",
            },
            "metrics": metrics,
        },
        bundle_path,
    )
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    summary_path = region.model_dir(PROJECT_ROOT) / "training_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "multi_source": metrics,
                "chirps_legacy": {"skipped": "Belagavi-only trainer"},
                "convlstm": {"skipped": "Belagavi-only trainer"},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return metrics


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
