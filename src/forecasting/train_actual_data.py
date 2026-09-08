"""Train a reproducible CHIRPS-to-IMD rainfall bias-correction model.

This is deliberately a calibrated rainfall-estimation model, trained only on
the real data included in this repository.  It uses a chronological holdout,
so its reported performance is not inflated by randomly mixing future event
days into the training set.
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

from data_utils.imd_ingest import discover_training_years, load_imd_rainfall_multi_year


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "chirps_to_imd_bias_corrector.joblib"
REPORT_PATH = PROJECT_ROOT / "models" / "chirps_to_imd_bias_corrector_metrics.json"
BBOX = {"lat_min": 14.625, "lat_max": 15.375, "lon_min": 74.625, "lon_max": 75.375}
HOLDOUT_YEAR = 2019
FEATURES = ["chirps_mm", "elevation_m", "slope_deg", "twi", "lat", "lon", "month", "day_of_year"]


def _terrain_at_points(lats: np.ndarray, lons: np.ndarray, filename: str) -> np.ndarray:
    path = PROJECT_ROOT / "data" / "terrain" / filename
    with rasterio.open(path) as source:
        values = np.asarray([value[0] for value in source.sample(zip(lons, lats))], dtype=float)
        if source.nodata is not None:
            values[np.isclose(values, source.nodata)] = np.nan
    return values


def _open_yearly_stack(paths: list[Path], var: str) -> xr.DataArray:
    existing = [p for p in paths if p.exists()]
    if not existing:
        raise FileNotFoundError(f"No files among: {paths}")
    arrays = []
    for path in existing:
        with xr.open_dataset(path) as ds:
            arrays.append(ds[var].load())
    return xr.concat(arrays, dim="time").sortby("time")


def build_training_frame() -> tuple[pd.DataFrame, list[int]]:
    """Align CHIRPS satellite and IMD analysis grids over all available monsoon seasons."""
    data_dir = PROJECT_ROOT / "data" / "gridded_rainfall"
    years = discover_training_years(data_dir)
    if not years:
        raise FileNotFoundError("No complete training years found.")

    imd = load_imd_rainfall_multi_year(years, data_dir, BBOX, monsoon_only=True)
    chirps_paths = [PROJECT_ROOT / "data" / "satellite" / f"chirps_belagavi_{y}.nc" for y in years]
    chirps = _open_yearly_stack(chirps_paths, "rainfall")
    chirps_on_imd = chirps.interp(lat=imd.lat, lon=imd.lon, method="linear")
    target = imd["rainfall"]
    frame = xr.Dataset({"chirps_mm": chirps_on_imd, "imd_mm": target}).to_dataframe().reset_index()
    frame = frame.dropna(subset=["chirps_mm", "imd_mm"]).copy()
    points = frame[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
    for name, filename in [("elevation_m", "dem_belagavi.tif"), ("slope_deg", "slope_belagavi.tif"), ("twi", "twi_belagavi.tif")]:
        points[name] = _terrain_at_points(points.lat.to_numpy(), points.lon.to_numpy(), filename)
    frame = frame.merge(points, on=["lat", "lon"], how="left")
    frame["month"] = pd.to_datetime(frame["time"]).dt.month
    frame["day_of_year"] = pd.to_datetime(frame["time"]).dt.dayofyear
    return frame.dropna(subset=FEATURES + ["imd_mm"]).sort_values("time").reset_index(drop=True), years


def train() -> dict:
    frame, years = build_training_frame()
    holdout_start = pd.Timestamp(f"{HOLDOUT_YEAR}-06-01")
    train_df = frame[frame.time < holdout_start]
    test_df = frame[frame.time >= holdout_start]
    if len(train_df) < 100:
        raise ValueError(f"Need at least 100 training samples; got {len(train_df)}")
    model = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=10, l2_regularization=2.0,
                                          learning_rate=0.08, random_state=42)
    model.fit(train_df[FEATURES], train_df["imd_mm"])
    predicted = np.clip(model.predict(test_df[FEATURES]), 0, None)
    baseline = test_df["chirps_mm"].to_numpy()
    actual = test_df["imd_mm"].to_numpy()
    model_mae = float(mean_absolute_error(actual, predicted))
    baseline_mae = float(mean_absolute_error(actual, baseline))
    use_model = model_mae <= baseline_mae
    metrics = {
        "training_samples": int(len(train_df)), "test_samples": int(len(test_df)),
        "training_years": [y for y in years if y < HOLDOUT_YEAR],
        "holdout_year": HOLDOUT_YEAR,
        "training_end_exclusive": str(holdout_start.date()),
        "test_period": [str(pd.Timestamp(test_df.time.min()).date()), str(pd.Timestamp(test_df.time.max()).date())],
        "model_mae_mm": model_mae,
        "model_rmse_mm": float(mean_squared_error(actual, predicted) ** 0.5),
        "model_r2": float(r2_score(actual, predicted)),
        "chirps_baseline_mae_mm": baseline_mae,
        "chirps_baseline_rmse_mm": float(mean_squared_error(actual, baseline) ** 0.5),
        "selected_predictor": "hist_gradient_boosting" if use_model else "raw_chirps_baseline",
        "selection_rule": "Select the predictor with lower chronological holdout MAE.",
        "caveat": f"Trained on {len(years)} monsoon seasons ({years[0]}-{years[-1]}); holdout year {HOLDOUT_YEAR}.",
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({
        "model": model if use_model else None,
        "features": FEATURES,
        "metrics": metrics,
        "predict": "model.predict(features) if model is not None else features['chirps_mm']",
    }, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
