#!/usr/bin/env python3
"""
Uncertainty Estimation Module for SIH26071.

This module adds confidence bands to rainfall forecasts from the nowcasting
and NWP bias‑correction modules, and propagates uncertainty to downstream
flood extent maps. It provides:

- Source‑specific uncertainty estimation:
  * Nowcasting: STEPS ensemble (pysteps) or empirical lead‑time‑dependent error model.
  * NWP bias correction: direct quantile extraction or conformal prediction.
  * Downstream inundation: ensemble run per rainfall quantile.
- A unified output format (`QuantileForecast`) with validation.
- Calibration checks (reliability diagram, coverage) and CRPS scoring.
- Visualization functions for time‑series and spatial maps.

Author: SIH26071 Team
"""

import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Union, Callable

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# Optional dependencies
try:
    import pysteps
    from pysteps import nowcasts, motion, verification
    PYSTEPS_AVAILABLE = True
except ImportError:
    PYSTEPS_AVAILABLE = False
    logging.warning("pysteps not installed. STEPS ensemble not available.")

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Unified output format
# ----------------------------------------------------------------------
@dataclass
class QuantileForecast:
    """
    Container for probabilistic forecasts.

    Attributes
    ----------
    quantiles : dict
        Maps quantile level (float) to 2D/3D array of same shape.
        Example: {0.1: array_like, 0.5: array_like, 0.9: array_like}
    lead_time : int or float
        Lead time in minutes (for nowcasting) or hours (for NWP).
        Use consistent unit as specified by the source.
    source : str
        Identifier ('nowcasting', 'nwp_bias_correction', etc.)
    """
    quantiles: Dict[float, np.ndarray]
    lead_time: float
    source: str

    def __post_init__(self):
        self.validate()

    def validate(self):
        """Check that quantile ordering is valid (p10 <= p50 <= p90)."""
        levels = sorted(self.quantiles.keys())
        for i in range(len(levels) - 1):
            lower = self.quantiles[levels[i]]
            upper = self.quantiles[levels[i+1]]
            if not np.all(lower <= upper):
                raise ValueError(f"Quantile crossing detected between {levels[i]} and {levels[i+1]}.")
        logger.debug("Quantile ordering OK.")


def check_quantile_ordering(quantile_dict: Dict[float, np.ndarray]) -> bool:
    """Return True if all quantile levels are ordered correctly."""
    levels = sorted(quantile_dict.keys())
    for i in range(len(levels) - 1):
        if not np.all(quantile_dict[levels[i]] <= quantile_dict[levels[i+1]]):
            return False
    return True


# ----------------------------------------------------------------------
# Nowcasting uncertainty
# ----------------------------------------------------------------------
def nowcast_uncertainty_steps(
    precip_frames: np.ndarray,
    lead_times_min: List[int],
    num_ensemble_members: int = 20,
    seed: int = 42,
    timestep_min: int = 10,
) -> Dict[int, QuantileForecast]:
    """
    Generate ensemble nowcasts using pysteps STEPS (Short‑Term Ensemble
    Prediction System) and derive quantiles.

    STEPS perturbs the motion field and adds stochastic noise to produce
    an ensemble of plausible extrapolations, from which percentile maps
    can be computed.

    Parameters
    ----------
    precip_frames : np.ndarray
        Recent rainfall fields with shape (time, y, x) in mm/h.
    lead_times_min : list of int
        Lead times in minutes for which to return quantile maps.
    num_ensemble_members : int
        Number of ensemble members.
    seed : int
        Random seed for reproducibility.
    timestep_min : int
        Time between input frames in minutes.

    Returns
    -------
    dict
        Maps lead time to a QuantileForecast object.
    """
    if not PYSTEPS_AVAILABLE:
        raise ImportError("pysteps is required for STEPS ensemble nowcasting.")

    # Determine motion field from the input sequence
    motion_method = motion.get_method("LK")
    V = motion_method(precip_frames)

    # Set up STEPS nowcaster
    nowcast_method = nowcasts.get_method("steps")
    # STEPS requires a precipitation object; we'll create a dummy one
    # For simplicity, assume input is a plain array; pysteps can work with it.
    # We'll use the steps nowcaster directly with the last frame and V.
    # In pysteps, steps requires a precipitation object with metadata;
    # we'll keep it simple: use the extrapolation with perturbation
    # for demonstration, or use the ensemble via pysteps.ensemble.
    # Since the full STEPS implementation requires specific data structures,
    # we provide a fallback that mimics the perturbation logic if pysteps
    # is not fully compatible. The actual pysteps usage would require
    # pysteps.timeseries and pysteps.rcparams. For brevity, we implement
    # a simplified perturbation of the motion field and advection.

    # Here we simulate ensemble spread by perturbing motion field and
    # adding noise. This is a simplified version of STEPS.
    np.random.seed(seed)
    last_frame = precip_frames[-1]
    forecasts = {}
    for lead in lead_times_min:
        n_steps = max(1, int(lead / timestep_min))
        ensemble = []
        for _ in range(num_ensemble_members):
            # Perturb motion vector slightly
            V_pert = V + np.random.normal(0, 0.2, V.shape)
            # Advect using pysteps extrapolation (or simple advection)
            # We'll use pysteps.nowcasts.extrapolation if available.
            if nowcasts.get_method("extrapolation"):
                extrap = nowcasts.get_method("extrapolation")
                forecast = extrap(last_frame, V_pert, n_steps)
            else:
                # Fallback simple advection (not fully accurate)
                from scipy.ndimage import map_coordinates
                h, w = last_frame.shape
                y, x = np.mgrid[0:h, 0:w].astype(np.float32)
                map_x = x - V_pert[..., 0] * n_steps
                map_y = y - V_pert[..., 1] * n_steps
                forecast = map_coordinates(last_frame, [map_y, map_x], order=1, mode='nearest')
            # Add stochastic noise, increasing with lead time
            noise_scale = 0.1 * n_steps
            forecast = forecast + np.random.normal(0, noise_scale, forecast.shape)
            forecast = np.maximum(forecast, 0)
            ensemble.append(forecast)
        ensemble = np.array(ensemble)
        # Compute quantiles
        q10 = np.percentile(ensemble, 10, axis=0)
        q50 = np.percentile(ensemble, 50, axis=0)
        q90 = np.percentile(ensemble, 90, axis=0)
        forecasts[lead] = QuantileForecast(
            quantiles={0.1: q10, 0.5: q50, 0.9: q90},
            lead_time=lead,
            source='nowcasting_steps'
        )
    return forecasts


def nowcast_uncertainty_empirical(
    recent_frame: np.ndarray,
    lead_time_min: int,
    error_model: Dict[int, Dict[str, np.ndarray]],
) -> QuantileForecast:
    """
    Empirical lead‑time‑dependent uncertainty for a single deterministic
    nowcast output (fallback when STEPS is unavailable).

    Parameters
    ----------
    recent_frame : np.ndarray
        The deterministic nowcast rainfall map for the given lead time.
    lead_time_min : int
        Lead time in minutes.
    error_model : dict
        Pre‑computed error quantiles for each lead time.
        Structure: {lead_time: {'q10': array, 'q50': array, 'q90': array}}
        where arrays are the same shape as the forecast grid and represent
        multiplicative or additive error factors.

    Returns
    -------
    QuantileForecast
    """
    if lead_time_min not in error_model:
        raise KeyError(f"No error model for lead time {lead_time_min} min.")
    err = error_model[lead_time_min]
    # Assume error model gives quantiles of the forecast error distribution
    # (e.g., multiplicative factors or additive adjustments)
    q10 = recent_frame + err['q10']
    q50 = recent_frame + err['q50']
    q90 = recent_frame + err['q90']
    # Clip negative values
    q10 = np.maximum(q10, 0)
    q50 = np.maximum(q50, 0)
    q90 = np.maximum(q90, 0)
    return QuantileForecast(
        quantiles={0.1: q10, 0.5: q50, 0.9: q90},
        lead_time=lead_time_min,
        source='nowcasting_empirical'
    )


# ----------------------------------------------------------------------
# NWP bias correction uncertainty
# ----------------------------------------------------------------------
def nwp_uncertainty_from_quantiles(
    quantile_dict: Dict[float, np.ndarray],
    lead_time_hours: float,
) -> QuantileForecast:
    """
    Wrap already‑computed quantiles from a quantile regression model.
    """
    return QuantileForecast(
        quantiles=quantile_dict,
        lead_time=lead_time_hours,
        source='nwp_quantile_regression'
    )


def conformal_prediction(
    calibration_scores: np.ndarray,
    significance: float = 0.2,
) -> float:
    """
    Compute conformal quantile threshold from calibration scores.

    Implements split conformal prediction: using absolute residuals from a
    held‑out calibration set, the (1‑significance) quantile of the scores
    is used as the half‑width of the prediction interval.

    Parameters
    ----------
    calibration_scores : np.ndarray
        Absolute errors |y - ŷ| on calibration data.
    significance : float
        Desired miscoverage rate (e.g., 0.2 for 80% interval).

    Returns
    -------
    float
        Threshold value q such that P(|y - ŷ| <= q) ≈ 1 - significance.
    """
    n = len(calibration_scores)
    if n < 1:
        raise ValueError("Calibration set is empty.")
    # Compute the (1 - significance) quantile with finite sample correction
    q_index = int(np.ceil((1 - significance) * (n + 1)))
    q = np.sort(calibration_scores)[min(q_index, n-1)]
    return q


def nwp_uncertainty_conformal(
    point_forecast: np.ndarray,
    calibration_data: Tuple[np.ndarray, np.ndarray],
    significance: float = 0.2,
) -> QuantileForecast:
    """
    Apply split conformal prediction to a point forecast.

    Parameters
    ----------
    point_forecast : np.ndarray
        Deterministic forecast (e.g., from gradient boosting regressor).
    calibration_data : tuple of arrays (X_cal, y_cal) or precomputed residuals.
        If tuple of (X, y), we'll compute residuals as |y - model.predict(X)|.
        However, we don't have the model here; we can accept residuals directly.
        For simplicity, calibration_data is assumed to be an array of absolute
        residuals from the calibration set.
    significance : float
        Desired miscoverage rate (0.2 gives 80% interval).

    Returns
    -------
    QuantileForecast
    """
    if isinstance(calibration_data, tuple):
        # Assume first element is residuals
        calibration_scores = calibration_data[0]
    else:
        calibration_scores = calibration_data

    q = conformal_prediction(calibration_scores, significance)
    lower = point_forecast - q
    upper = point_forecast + q
    median = point_forecast  # point forecast as median
    lower = np.maximum(lower, 0)
    upper = np.maximum(upper, 0)
    return QuantileForecast(
        quantiles={0.1: lower, 0.5: median, 0.9: upper},
        lead_time=0,  # To be set by caller
        source='nwp_conformal'
    )


# ----------------------------------------------------------------------
# Downstream inundation uncertainty
# ----------------------------------------------------------------------
def run_inundation_ensemble(
    rainfall_quantiles: Dict[float, np.ndarray],
    inundation_model: Callable[[np.ndarray], np.ndarray],
) -> Dict[float, np.ndarray]:
    """
    Propagate rainfall uncertainty to flood extent by running an inundation
    model (or surrogate) on each rainfall quantile map.

    Parameters
    ----------
    rainfall_quantiles : dict
        Maps quantile level to rainfall map.
    inundation_model : callable
        Function that takes a rainfall map and returns a flood extent map
        (e.g., depth or binary inundation). This can be a simple threshold
        or a trained ML surrogate.

    Returns
    -------
    dict
        Maps quantile level to flood extent map.
    """
    flood_maps = {}
    for q, rain in rainfall_quantiles.items():
        flood_maps[q] = inundation_model(rain)
    return flood_maps


# ----------------------------------------------------------------------
# Calibration and scoring
# ----------------------------------------------------------------------
def compute_interval_coverage(
    obs: np.ndarray,
    forecast_lower: np.ndarray,
    forecast_upper: np.ndarray,
) -> float:
    """
    Compute empirical coverage of a prediction interval (e.g., p10‑p90).

    Parameters
    ----------
    obs : np.ndarray
        Observations.
    forecast_lower, forecast_upper : np.ndarray
        Lower and upper bounds of the interval, same shape as obs.

    Returns
    -------
    float
        Fraction of observations falling within the interval.
    """
    mask = ~np.isnan(obs)
    obs = obs[mask]
    lower = forecast_lower[mask]
    upper = forecast_upper[mask]
    coverage = np.mean((obs >= lower) & (obs <= upper))
    return coverage


def plot_reliability_diagram(
    forecasts: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    obs: np.ndarray,
    n_bins: int = 10,
) -> None:
    """
    Plot a reliability diagram for a set of probabilistic forecasts.

    Each tuple in forecasts should be (lower, median, upper) arrays.
    The diagram shows whether the nominal interval coverage matches the
    empirical frequency across bins of predicted median.

    Parameters
    ----------
    forecasts : list of tuples
        Each tuple contains (lower, median, upper) for a forecast instance.
    obs : np.ndarray
        Corresponding observations (flattened).
    n_bins : int
        Number of bins along the predicted median axis.
    """
    if not SKLEARN_AVAILABLE:
        logger.warning("scikit-learn needed for isotonic regression, using simple binning.")
    # Flatten arrays
    all_lower = np.concatenate([f[0].ravel() for f in forecasts])
    all_median = np.concatenate([f[1].ravel() for f in forecasts])
    all_upper = np.concatenate([f[2].ravel() for f in forecasts])
    obs_flat = obs.ravel()

    # Bin by median forecast value
    bin_edges = np.percentile(all_median, np.linspace(0, 100, n_bins+1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    empirical_coverage = []
    for i in range(n_bins):
        mask = (all_median >= bin_edges[i]) & (all_median < bin_edges[i+1])
        if mask.sum() > 5:
            cov = np.mean((obs_flat[mask] >= all_lower[mask]) & (obs_flat[mask] <= all_upper[mask]))
            empirical_coverage.append(cov)
        else:
            empirical_coverage.append(np.nan)
    # Plot
    plt.figure(figsize=(6,6))
    plt.plot(bin_centers, empirical_coverage, 'o-', label='Empirical coverage')
    plt.plot([bin_centers[0], bin_centers[-1]], [0.8, 0.8], 'k--', label='Nominal 80%')
    plt.xlabel('Forecast median (mm)')
    plt.ylabel('Empirical coverage')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.grid(True)
    plt.show()


def crps_ensemble(ensemble: np.ndarray, obs: np.ndarray) -> float:
    """
    Compute the Continuous Ranked Probability Score (CRPS) for an ensemble
    forecast against observations.

    Parameters
    ----------
    ensemble : np.ndarray
        Ensemble members with shape (n_members, ...).
    obs : np.ndarray
        Observations matching the spatial/temporal dimensions.

    Returns
    -------
    float
        Mean CRPS over all points.
    """
    # Flatten all but member dimension
    n_members = ensemble.shape[0]
    obs_flat = obs.ravel()
    ens_flat = ensemble.reshape(n_members, -1)
    # CRPS = mean(|X - y|) - 0.5 * mean(|X - X'|)
    # For each observation point, compute average over pairs.
    # Vectorized implementation using broadcasting.
    # For each point i, crps_i = mean_j |x_j - y_i| - 0.5 * mean_{j,k} |x_j - x_k|
    # We'll compute using loops for simplicity (small size) or use scipy.
    # Here we use a simple loop over points.
    crps_sum = 0.0
    n_points = obs_flat.shape[0]
    for i in range(n_points):
        x = ens_flat[:, i]
        y = obs_flat[i]
        term1 = np.mean(np.abs(x - y))
        # term2: average over all pairs of ensemble members
        if n_members > 1:
            # compute pairwise differences
            diff = np.abs(x[:, None] - x[None, :])
            term2 = 0.5 * np.mean(diff)
        else:
            term2 = 0.0
        crps_sum += term1 - term2
    return crps_sum / n_points


# ----------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------
def plot_cell_time_series(
    quantile_forecasts: List[QuantileForecast],
    lead_times: List[float],
    cell_index: Tuple[int, int],
    source_label: str = 'Forecast',
) -> None:
    """
    Plot a time series for a single grid cell showing median and 10‑90%
    band across lead times.

    Parameters
    ----------
    quantile_forecasts : list of QuantileForecast
        Forecasts for different lead times (each must have p10, p50, p90).
    lead_times : list of float
        Corresponding lead times.
    cell_index : tuple
        (row, col) index of the cell in the spatial grid.
    source_label : str
        Label for the title.
    """
    p10 = [np.atleast_2d(qf.quantiles[0.1])[cell_index] for qf in quantile_forecasts]
    p50 = [np.atleast_2d(qf.quantiles[0.5])[cell_index] for qf in quantile_forecasts]
    p90 = [np.atleast_2d(qf.quantiles[0.9])[cell_index] for qf in quantile_forecasts]

    plt.figure(figsize=(8,5))
    plt.fill_between(lead_times, p10, p90, alpha=0.3, label='80% interval (p10-p90)')
    plt.plot(lead_times, p50, 'b-', label='Median (p50)')
    plt.xlabel('Lead time')
    plt.ylabel('Rainfall (mm)')
    plt.title(f'{source_label} forecast at cell {cell_index}')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_spatial_quantiles(
    quantile_forecast: QuantileForecast,
    lead_time: float,
    variable_name: str = 'Rainfall (mm)',
) -> None:
    """
    Plot three side‑by‑side maps (p10, p50, p90) for a single lead time.

    Parameters
    ----------
    quantile_forecast : QuantileForecast
        Forecast for one lead time.
    lead_time : float
        Lead time (used in title).
    variable_name : str
        Label for colorbar.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15,4))
    for ax, q, title in zip(axes, [0.1, 0.5, 0.9], ['10th percentile', 'Median (50th)', '90th percentile']):
        data = quantile_forecast.quantiles[q]
        im = ax.imshow(data, origin='lower', cmap='viridis')
        ax.set_title(f'{title} (lead {lead_time})')
        fig.colorbar(im, ax=ax, label=variable_name)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# Demonstration (synthetic data)
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # Create dummy rainfall fields
    np.random.seed(0)
    lat = np.linspace(10, 20, 20)
    lon = np.linspace(70, 80, 20)
    lon2d, lat2d = np.meshgrid(lon, lat)
    base_field = 10 * np.exp(-((lat2d-15)**2 + (lon2d-75)**2)/2)

    # Simulate a few nowcast lead times with increasing spread
    lead_times_min = [10, 30, 60, 120]
    forecasts = []
    for lead in lead_times_min:
        # Deterministic forecast (median)
        median = base_field * (1 - 0.01*lead)  # decay
        # Add spread proportional to lead
        spread = 0.5 + 0.05*lead
        p10 = median - spread * np.random.rand(*median.shape)
        p90 = median + spread * np.random.rand(*median.shape)
        p10 = np.maximum(p10, 0)
        p90 = np.maximum(p90, 0)
        qf = QuantileForecast(
            quantiles={0.1: p10, 0.5: median, 0.9: p90},
            lead_time=lead,
            source='demo'
        )
        forecasts.append(qf)

    # Plot cell time series (pick a cell near center)
    cell = (10, 10)
    plot_cell_time_series(forecasts, lead_times_min, cell, source_label='Demo Nowcast')

    # Plot spatial quantiles for one lead time
    plot_spatial_quantiles(forecasts[2], lead_times_min[2])

    # Demonstrate calibration check with synthetic obs
    # Assume observations equal median plus noise
    obs = np.array([f.quantiles[0.5] + np.random.normal(0, 1.0, f.quantiles[0.5].shape) for f in forecasts])
    # Compute coverage for the last forecast
    coverage = compute_interval_coverage(
        obs[-1].ravel(),
        forecasts[-1].quantiles[0.1].ravel(),
        forecasts[-1].quantiles[0.9].ravel()
    )
    print(f"Empirical coverage for last forecast: {coverage:.2f}")

    # CRPS for the ensemble (using p10/p50/p90 as a crude three-member ensemble)
    # Not a true ensemble, but illustrative
    print("CRPS (approximate) not computed in demo.")