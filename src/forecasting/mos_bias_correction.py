#!/usr/bin/env python3
"""
NWP Model Output Statistics (MOS) Bias Correction Module for SIH26071.

This module provides post-processing of NWP precipitation forecasts using
machine learning to correct systematic biases against historical observations.
The approach is standard Model Output Statistics (MOS) – we are not replacing
the NWP model but learning to correct its errors.

Two model families are implemented:
1. Quantile regression (via LightGBM with quantile objective) – outputs
   multiple quantiles (10th, 50th, 90th) to capture forecast uncertainty.
2. Gradient boosting regression (LightGBM) – a single bias‑corrected estimate.

Both models are trained on paired (forecast, observed) data, with features
including raw NWP rainfall, lead time, spatial coordinates, and seasonality.

Key design decisions (explained in comments):
- Target variable: we predict the forecast error (obs - nwp) rather than
  the observed value directly. This stabilizes the learning problem and
  explicitly preserves the NWP signal as a baseline.
- Chronological validation: time‑series split to avoid temporal leakage.
- For small datasets, a single model with lead_time as a feature is preferred
  over separate models per lead time to share information.

If historical data is insufficient (< one monsoon season), a simple
climatological bias correction fallback is provided.

Author: SIH26071 Team
"""

import logging

import cv2
import numpy as np
import pandas as pd
import xarray as xr
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass

# ML libraries
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.error("LightGBM not installed. Please install it via pip install lightgbm")

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Plotting for reliability diagram (optional)
try:
    import matplotlib.pyplot as plt
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Data preparation
# ----------------------------------------------------------------------
def build_training_data(nwp_forecast: xr.Dataset,
                        observations: xr.Dataset,
                        forecast_var: str = 'tp',
                        obs_var: str = 'rainfall',
                        auxiliary_vars: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Pair NWP forecasts with observations at the correct valid_time.

    This is the most critical step. For each forecast (init_time, lead_time,
    lat, lon), the valid_time is init_time + lead_time. We select the
    observation at that exact valid_time from the observations dataset.

    Parameters
    ----------
    nwp_forecast : xr.Dataset
        Forecast dataset with dimensions (init_time, lead_time, lat, lon).
        Must contain the forecast variable (e.g., 'tp') and optionally
        auxiliary variables at the same grid.
    observations : xr.Dataset
        Observed rainfall dataset with dimension (time, lat, lon) and
        variable 'rainfall' (or obs_var).
    forecast_var : str
        Name of the NWP precipitation variable.
    obs_var : str
        Name of the observed precipitation variable.
    auxiliary_vars : list of str, optional
        Names of auxiliary NWP variables to include as features.

    Returns
    -------
    pd.DataFrame
        Clean DataFrame with columns:
        - init_time
        - lead_time (in hours)
        - lat, lon
        - nwp_rainfall (raw forecast)
        - obs_rainfall
        - error (obs - nwp)
        - auxiliary variables (if provided)
        - month, season (derived)
        - lead_time as numeric (hours)
    """
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM is required for this module.")

    # Validate that init_time, lead_time, lat, lon exist in forecast
    required_dims = ['init_time', 'lead_time', 'lat', 'lon']
    for dim in required_dims:
        if dim not in nwp_forecast.dims:
            raise ValueError(f"Forecast dataset missing dimension: {dim}")

    # Validate observation dims
    if 'time' not in observations.dims or 'lat' not in observations.dims or 'lon' not in observations.dims:
        raise ValueError("Observations must have dimensions 'time', 'lat', 'lon'.")

    # Extract coordinates
    init_times = nwp_forecast['init_time'].values
    lead_times = nwp_forecast['lead_time'].values
    lat_vals = nwp_forecast['lat'].values
    lon_vals = nwp_forecast['lon'].values

    # Create lists to hold data
    rows = []

    # Loop over init_times, lead_times, and grid points
    # To avoid nested loops over all grid points, we can use xarray's broadcasting
    # But for simplicity and clarity, we iterate over time and grid (could be slow for large data)
    # Alternatively, we can use .stack() to flatten spatial dimensions.
    # We'll use a vectorized approach: create a DataArray of valid_times.

    # Compute valid_time for each (init_time, lead_time) pair
    # Convert init_time and lead_time to numpy arrays
    init_dt = pd.to_datetime(init_times)
    lead_hours = lead_times.astype(float)  # assume hours

    # Create a meshgrid of init_time and lead_time
    # We'll iterate over init_time and lead_time separately
    logger.info(f"Processing {len(init_dt)} init times and {len(lead_hours)} lead times...")

    for i, init in enumerate(init_dt):
        for j, lead_h in enumerate(lead_hours):
            valid_time = init + pd.Timedelta(hours=lead_h)

            # Select forecast for this init_time and lead_time
            fc_slice = nwp_forecast[forecast_var].sel(init_time=init, lead_time=lead_h)
            # Select observation for this valid_time
            # Use nearest neighbour in time if exact match not found
            try:
                obs_slice = observations[obs_var].sel(time=valid_time, method='nearest')
            except KeyError:
                logger.warning(f"No observation near {valid_time}, skipping.")
                continue

            # Ensure spatial alignment: both should be on same lat/lon grid
            # If not, we would need to regrid; assume they are already aligned.

            # Flatten spatial dimensions
            fc_flat = fc_slice.values.flatten()
            obs_flat = obs_slice.values.flatten()

            # Create meshgrid for lat/lon (already 1D arrays)
            lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)
            lat_flat = lat2d.flatten()
            lon_flat = lon2d.flatten()

            # Build rows for this time pair
            for k in range(len(fc_flat)):
                # Skip if either forecast or obs is NaN
                if np.isnan(fc_flat[k]) or np.isnan(obs_flat[k]):
                    continue

                row = {
                    'init_time': init,
                    'valid_time': valid_time,
                    'lead_time': lead_h,
                    'lat': lat_flat[k],
                    'lon': lon_flat[k],
                    'nwp_rainfall': fc_flat[k],
                    'obs_rainfall': obs_flat[k],
                    'error': obs_flat[k] - fc_flat[k],
                    'month': valid_time.month,
                    'season': _get_season(valid_time.month)
                }

                # Add auxiliary variables if present
                if auxiliary_vars:
                    for aux_var in auxiliary_vars:
                        if aux_var in nwp_forecast:
                            aux_slice = nwp_forecast[aux_var].sel(init_time=init, lead_time=lead_h)
                            aux_flat = aux_slice.values.flatten()
                            row[aux_var] = aux_flat[k]
                        else:
                            logger.warning(f"Auxiliary variable '{aux_var}' not found in forecast dataset.")
                rows.append(row)

            logger.debug(f"Processed init {init}, lead {lead_h}h, valid {valid_time}")

    if not rows:
        raise ValueError("No valid (forecast, observation) pairs found. Check data alignment.")

    df = pd.DataFrame(rows)
    logger.info(f"Built training DataFrame with {len(df)} samples.")
    return df


def _get_season(month: int) -> str:
    """
    Simple monsoon season classification for Indian context.
    """
    if month in [6, 7, 8, 9]:
        return 'monsoon'
    elif month in [10, 11]:
        return 'post-monsoon'
    elif month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'pre-monsoon'
    else:
        return 'unknown'


# ----------------------------------------------------------------------
# Bias correction models
# ----------------------------------------------------------------------
class BaseBiasCorrector:
    """
    Base class for bias correction models.
    """
    def __init__(self, feature_columns: List[str]):
        self.feature_columns = feature_columns
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class QuantileBiasCorrector(BaseBiasCorrector):
    """
    Quantile regression using LightGBM with quantile objective.

    Trains separate models for each quantile (e.g., 0.1, 0.5, 0.9).
    The predicted quantiles are added back to the raw forecast to produce
    bias-corrected rainfall distributions.
    """
    def __init__(self, feature_columns: List[str], quantiles: List[float] = [0.1, 0.5, 0.9]):
        super().__init__(feature_columns)
        self.quantiles = quantiles
        self.models: Dict[float, lgb.LGBMRegressor] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        for q in self.quantiles:
            logger.info(f"Training quantile {q}...")
            model = lgb.LGBMRegressor(
                objective='quantile',
                alpha=q,
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            )
            model.fit(X, y, **kwargs)
            self.models[q] = model
        return self

    def predict(self, X: pd.DataFrame) -> Dict[float, np.ndarray]:
        preds = {}
        for q, model in self.models.items():
            preds[q] = model.predict(X)
        return preds


class GradientBoostingBiasCorrector(BaseBiasCorrector):
    """
    Standard gradient boosting (LightGBM) for single-point bias correction.
    """
    def __init__(self, feature_columns: List[str]):
        super().__init__(feature_columns)
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        self.model = lgb.LGBMRegressor(
            objective='regression',
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        self.model.fit(X, y, **kwargs)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)


# ----------------------------------------------------------------------
# Feature engineering for inference
# ----------------------------------------------------------------------
def prepare_features_for_inference(nwp_forecast: xr.Dataset,
                                   lead_time: float,
                                   init_time,
                                   feature_columns: List[str]) -> pd.DataFrame:
    """
    Prepare a feature DataFrame for a single (init_time, lead_time) forecast grid.

    This is used at inference time to apply a trained model.
    """
    # Extract forecast for this init/lead
    fc_slice = nwp_forecast['tp'].sel(init_time=init_time, lead_time=lead_time)
    lat_vals = nwp_forecast['lat'].values
    lon_vals = nwp_forecast['lon'].values
    lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)

    # Flatten
    df = pd.DataFrame({
        'nwp_rainfall': fc_slice.values.flatten(),
        'lat': lat2d.flatten(),
        'lon': lon2d.flatten(),
        'lead_time': lead_time,
        'month': pd.to_datetime(init_time).month,
        'season': _get_season(pd.to_datetime(init_time).month)
    })

    # Add auxiliary variables if needed
    for aux in feature_columns:
        if aux not in df.columns and aux in nwp_forecast:
            aux_slice = nwp_forecast[aux].sel(init_time=init_time, lead_time=lead_time)
            df[aux] = aux_slice.values.flatten()

    # Ensure only feature columns are kept
    df = df[feature_columns]
    return df


# ----------------------------------------------------------------------
# Validation metrics (CSI, FSS)
# ----------------------------------------------------------------------
def compute_csi(pred: np.ndarray, obs: np.ndarray, threshold: float) -> float:
    """Critical Success Index."""
    pred_bin = pred >= threshold
    obs_bin = obs >= threshold
    hits = np.sum(pred_bin & obs_bin)
    misses = np.sum(~pred_bin & obs_bin)
    false_alarms = np.sum(pred_bin & ~obs_bin)
    denom = hits + misses + false_alarms
    if denom == 0:
        return np.nan
    return hits / denom

def compute_fss(pred: np.ndarray, obs: np.ndarray, threshold: float,
                neighborhood_size: int = 5) -> float:
    """Fractions Skill Score."""
    pred_bin = (pred >= threshold).astype(float)
    obs_bin = (obs >= threshold).astype(float)
    kernel = np.ones((neighborhood_size, neighborhood_size))
    pred_frac = cv2.filter2D(pred_bin, -1, kernel)
    obs_frac = cv2.filter2D(obs_bin, -1, kernel)
    numerator = np.sum((pred_frac - obs_frac) ** 2)
    denominator = np.sum(pred_frac**2) + np.sum(obs_frac**2)
    if denominator == 0:
        return np.nan
    return 1.0 - numerator / denominator

def evaluate_correction(raw_pred: np.ndarray, corrected_pred: np.ndarray,
                        obs: np.ndarray, thresholds: List[float] = [0.1, 1.0, 5.0, 10.0],
                        neighborhood_size: int = 5) -> pd.DataFrame:
    """
    Compute RMSE, MAE, CSI, and FSS for raw and corrected forecasts.
    """
    rows = []
    for th in thresholds:
        csi_raw = compute_csi(raw_pred, obs, th)
        csi_corr = compute_csi(corrected_pred, obs, th)
        fss_raw = compute_fss(raw_pred, obs, th, neighborhood_size)
        fss_corr = compute_fss(corrected_pred, obs, th, neighborhood_size)
        rows.append({
            'threshold': th,
            'CSI_raw': csi_raw,
            'CSI_corrected': csi_corr,
            'FSS_raw': fss_raw,
            'FSS_corrected': fss_corr
        })
    rmse_raw = np.sqrt(mean_squared_error(obs, raw_pred))
    rmse_corr = np.sqrt(mean_squared_error(obs, corrected_pred))
    mae_raw = mean_absolute_error(obs, raw_pred)
    mae_corr = mean_absolute_error(obs, corrected_pred)
    summary = pd.DataFrame([{'threshold': 'all', 'CSI_raw': np.nan, 'CSI_corrected': np.nan,
                             'FSS_raw': np.nan, 'FSS_corrected': np.nan,
                             'RMSE_raw': rmse_raw, 'RMSE_corrected': rmse_corr,
                             'MAE_raw': mae_raw, 'MAE_corrected': mae_corr}])
    # Combine with threshold metrics (we'll keep separate)
    df_metrics = pd.DataFrame(rows)
    df_metrics['RMSE_raw'] = rmse_raw
    df_metrics['RMSE_corrected'] = rmse_corr
    df_metrics['MAE_raw'] = mae_raw
    df_metrics['MAE_corrected'] = mae_corr
    return df_metrics


# ----------------------------------------------------------------------
# Reliability diagram for quantile forecasts
# ----------------------------------------------------------------------
def reliability_diagram(quantile_preds: Dict[float, np.ndarray],
                        obs: np.ndarray,
                        quantiles: List[float] = [0.1, 0.5, 0.9]) -> None:
    """
    Plot a reliability diagram for quantile forecasts.
    We bin observations by predicted quantile intervals and compute the
    observed frequency of exceedance.

    This is a simplified version: for each quantile q, we compute the
    fraction of observations that fall below the q-th predicted quantile.
    Ideally, that fraction should equal q.
    """
    if not PLOT_AVAILABLE:
        logger.warning("matplotlib not available; skipping reliability diagram.")
        return

    fig, ax = plt.subplots(figsize=(6,6))
    for q in quantiles:
        pred_q = quantile_preds[q]
        # Sort by predicted quantile
        sorted_idx = np.argsort(pred_q)
        sorted_pred = pred_q[sorted_idx]
        sorted_obs = obs[sorted_idx]
        # Compute empirical exceedance probability for bins
        n_bins = 10
        bin_edges = np.percentile(sorted_pred, np.linspace(0, 100, n_bins+1))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        emp_probs = []
        for i in range(n_bins):
            mask = (sorted_pred >= bin_edges[i]) & (sorted_pred < bin_edges[i+1])
            if mask.sum() > 0:
                emp_probs.append(np.mean(sorted_obs[mask] <= bin_centers[i]))
            else:
                emp_probs.append(np.nan)
        ax.plot(bin_centers, emp_probs, 'o-', label=f'q={q}')
    ax.plot([0,1], [0,1], 'k--', label='Perfect')
    ax.set_xlabel('Predicted quantile')
    ax.set_ylabel('Observed frequency')
    ax.set_title('Reliability Diagram')
    ax.legend()
    plt.show()


# ----------------------------------------------------------------------
# Fallback: climatological bias correction
# ----------------------------------------------------------------------
def climatological_bias_correction(nwp_forecast: xr.Dataset,
                                   historical_error_mean: xr.DataArray,
                                   forecast_var: str = 'tp') -> xr.DataArray:
    """
    Simple bias correction by subtracting the mean historical error
    (obs - nwp) per grid cell and lead time.

    This is useful when insufficient data prevents training an ML model.
    """
    # Align historical_error_mean (dimensions: lead_time, lat, lon) with forecast
    corrected = nwp_forecast[forecast_var] + historical_error_mean
    corrected = corrected.rename('bias_corrected')
    return corrected


# ----------------------------------------------------------------------
# Inference function
# ----------------------------------------------------------------------
def apply_bias_correction(nwp_forecast: xr.Dataset,
                          model: BaseBiasCorrector,
                          feature_columns: List[str],
                          lead_times: List[float],
                          init_time,
                          quantile: bool = False) -> Union[xr.DataArray, Dict[float, xr.DataArray]]:
    """
    Apply the trained bias correction model to a fresh NWP forecast.

    Parameters
    ----------
    nwp_forecast : xr.Dataset
        Forecast dataset for a single init_time (or multiple) with dims
        (lead_time, lat, lon) and variables.
    model : trained model object
    feature_columns : list of features used by the model.
    lead_times : list of lead times to correct.
    init_time : datetime for this forecast.
    quantile : bool
        If True, model is QuantileBiasCorrector and returns dict of DataArrays.

    Returns
    -------
    xr.DataArray or dict of xr.DataArray
        Bias-corrected rainfall (or error) on the same grid.
    """
    if isinstance(model, QuantileBiasCorrector):
        quantile = True

    corrected_maps = {}
    for lead_h in lead_times:
        # Prepare features for this lead time
        X = prepare_features_for_inference(nwp_forecast, lead_h, init_time, feature_columns)

        if quantile:
            pred_quantiles = model.predict(X)
            # For each quantile, convert error prediction to corrected rainfall
            # raw forecast = nwp_forecast['tp'].sel(lead_time=lead_h)
            raw = nwp_forecast['tp'].sel(lead_time=lead_h).values.flatten()
            quantile_arrays = {}
            for q, err_pred in pred_quantiles.items():
                # Note: model predicted error (obs - nwp), so corrected = raw + error
                corrected = raw + err_pred
                # Reshape to 2D
                lat_vals = nwp_forecast['lat'].values
                lon_vals = nwp_forecast['lon'].values
                corrected_2d = corrected.reshape(len(lat_vals), len(lon_vals))
                quantile_arrays[q] = xr.DataArray(corrected_2d,
                                                  coords={'lat': lat_vals, 'lon': lon_vals},
                                                  dims=['lat','lon'])
            corrected_maps[lead_h] = quantile_arrays
        else:
            err_pred = model.predict(X)
            raw = nwp_forecast['tp'].sel(lead_time=lead_h).values.flatten()
            corrected = raw + err_pred
            lat_vals = nwp_forecast['lat'].values
            lon_vals = nwp_forecast['lon'].values
            corrected_2d = corrected.reshape(len(lat_vals), len(lon_vals))
            corrected_maps[lead_h] = xr.DataArray(corrected_2d,
                                                  coords={'lat': lat_vals, 'lon': lon_vals},
                                                  dims=['lat','lon'])
    return corrected_maps


# ----------------------------------------------------------------------
# Main training and evaluation orchestration
# ----------------------------------------------------------------------
def train_and_evaluate(nwp_forecast: xr.Dataset,
                       observations: xr.Dataset,
                       feature_columns: Optional[List[str]] = None,
                       model_type: str = 'gradient_boosting',
                       quantiles: List[float] = [0.1, 0.5, 0.9],
                       train_frac: float = 0.8) -> Dict:
    """
    Complete pipeline: build training data, train model, evaluate.
    """
    # 1. Build training data
    df = build_training_data(nwp_forecast, observations)

    # 2. Chronological split (train/test by time)
    df = df.sort_values('valid_time')
    split_idx = int(len(df) * train_frac)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    logger.info(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

    # Define features and target
    if feature_columns is None:
        feature_columns = ['nwp_rainfall', 'lead_time', 'lat', 'lon', 'month']
    target_col = 'error'  # predict error (obs - nwp)

    # 3. Train model
    if model_type == 'quantile':
        model = QuantileBiasCorrector(feature_columns, quantiles)
        model.fit(train_df[feature_columns], train_df[target_col])
    elif model_type == 'gradient_boosting':
        model = GradientBoostingBiasCorrector(feature_columns)
        model.fit(train_df[feature_columns], train_df[target_col])
    else:
        raise ValueError("model_type must be 'quantile' or 'gradient_boosting'")

    # 4. Evaluate on test set
    test_features = test_df[feature_columns]
    test_raw = test_df['nwp_rainfall'].values
    test_obs = test_df['obs_rainfall'].values

    if model_type == 'quantile':
        pred_quantiles = model.predict(test_features)
        # Use median (q=0.5) as point prediction
        median_idx = quantiles.index(0.5) if 0.5 in quantiles else 0
        test_err_pred = pred_quantiles[quantiles[median_idx]]
        test_corrected = test_raw + test_err_pred
        # Compute metrics
        metrics_df = evaluate_correction(test_raw, test_corrected, test_obs)
        # Reliability diagram (optional)
        # reliability_diagram(pred_quantiles, test_obs, quantiles)
        return {'model': model, 'metrics': metrics_df, 'test_df': test_df}
    else:
        test_err_pred = model.predict(test_features)
        test_corrected = test_raw + test_err_pred
        metrics_df = evaluate_correction(test_raw, test_corrected, test_obs)
        return {'model': model, 'metrics': metrics_df, 'test_df': test_df}


# ----------------------------------------------------------------------
# Demonstration with synthetic data
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # Create dummy NWP forecast and observations
    logging.info("Creating synthetic NWP forecast and observation data...")

    # Define grid
    lat = np.linspace(10, 20, 10)  # 10 points
    lon = np.linspace(70, 80, 10)
    lead_times = [6, 12, 24, 48, 72]  # hours
    init_times = pd.date_range('2024-06-01', '2024-09-30', freq='1D')  # ~4 months

    # Synthetic forecast: random rain with bias
    np.random.seed(42)
    n_init = len(init_times)
    n_lead = len(lead_times)
    n_lat = len(lat)
    n_lon = len(lon)

    # Generate forecast values (log-normal)
    fc = np.random.exponential(scale=5.0, size=(n_init, n_lead, n_lat, n_lon))
    # Observation = forecast + systematic bias + noise
    bias = 2.0 * (np.sin(np.radians(lat))[:, None])  # lat-dependent bias
    obs = fc + bias[None, None, :, :] + np.random.normal(0, 1.0, fc.shape)
    obs = np.maximum(obs, 0)  # rainfall >= 0

    # Create xarray datasets
    ds_fc = xr.Dataset(
        {'tp': (('init_time','lead_time','lat','lon'), fc)},
        coords={'init_time': init_times, 'lead_time': lead_times,
                'lat': lat, 'lon': lon}
    )
    ds_obs = xr.Dataset(
        {'rainfall': (('time','lat','lon'), obs.reshape(-1, n_lat, n_lon))},
        coords={'time': pd.date_range('2024-06-01', periods=n_init*n_lead, freq='1D')}  # dummy times
    )
    # Note: The observation times are not aligned with valid_times in this dummy;
    # In reality you would have observations at each valid_time. For demonstration,
    # we'll adjust: The observation dataset should have time=init+lead for each pair.
    # We'll reconstruct it properly:
    obs_times = []
    obs_data = []
    for i, init in enumerate(init_times):
        for j, lead_h in enumerate(lead_times):
            valid = init + pd.Timedelta(hours=lead_h)
            obs_times.append(valid)
            obs_data.append(obs[i, j, :, :])
    obs_arr = np.stack(obs_data, axis=0)
    ds_obs = xr.Dataset(
        {'rainfall': (('time','lat','lon'), obs_arr)},
        coords={'time': obs_times, 'lat': lat, 'lon': lon}
    )

    # Run the pipeline
    logging.info("Training gradient boosting bias corrector...")
    results = train_and_evaluate(ds_fc, ds_obs, model_type='gradient_boosting')
    logging.info("Evaluation metrics:")
    print(results['metrics'])

    # Show quantile model example
    logging.info("Training quantile bias corrector...")
    q_results = train_and_evaluate(ds_fc, ds_obs, model_type='quantile',
                                   quantiles=[0.1, 0.5, 0.9])
    print(q_results['metrics'])