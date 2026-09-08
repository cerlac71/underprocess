#!/usr/bin/env python3
"""
Multi‑Source Data Fusion Module for SIH26071.

This module fuses rainfall estimates from multiple sources (satellite, radar,
gauge, NWP) into a single risk score with associated confidence. Two fusion
approaches are implemented:

1. Inverse‑variance weighting (recommended for hackathon) – weights each
   source by the reciprocal of its error variance, producing a minimum‑
   variance unbiased estimate under independent errors.

2. Bayesian sequential updating (Kalman‑filter style) – treats NWP as a
   prior and updates it with radar/gauge/satellite where available,
   producing a posterior mean and variance. More rigorous but also more
   sensitive to error correlation assumptions.

The module also provides:
- Source availability masks and disagreement diagnostics.
- Composition of a final interpretable risk score (0‑100) and a separate
  confidence level (0‑1) reflecting both source agreement and coverage.
- Validation helpers comparing fused accuracy to individual sources.
- Visualization of risk and confidence as a dual‑encoded map.

Default error variances for each source type are taken from published
literature (see comments), but can be overridden with user‑provided values.

Author: SIH26071 Team
"""

import logging
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Default error variances (in (mm/hr)^2 for rainfall rate) and sources
# References:
# Satellite (INSAT-3D/3DR Hydro-Estimator):
#   Mishra et al. (2016) found RMSE ~2-5 mm/hr in convective events.
#   We adopt a moderate variance of (2.0)^2 = 4.0 (mm/hr)^2.
# Radar (Doppler weather radar):
#   Typical QPE errors: 30-50% of intensity, but for moderate rain (<10 mm/hr)
#   variance is often ~1-2 (mm/hr)^2. We use 1.5^2 = 2.25.
# Rain gauge (point measurement, assumed representative):
#   Very low error at the gauge itself, but spatial representativeness error.
#   For grid cell average, we assign 1.0 (mm/hr)^2.
# NWP (bias‑corrected):
#   After quantile mapping, variance can be estimated from the inter‑quantile
#   range. We use a default of 3.0^2 = 9.0, but will be overridden by user
#   if quantile bands are provided.
DEFAULT_VARIANCE = {
    'satellite': 4.0,
    'radar': 2.25,
    'gauge': 1.0,
    'nwp': 9.0,
}

# Source names as used in the module
SOURCE_NAMES = ['satellite', 'radar', 'gauge', 'nwp']


# ----------------------------------------------------------------------
# 1. Inverse Variance Weighting
# ----------------------------------------------------------------------
def inverse_variance_fusion(
    estimates: Dict[str, np.ndarray],
    variances: Dict[str, np.ndarray],
    mask: Optional[Dict[str, np.ndarray]] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Fuse multiple rainfall estimates using inverse‑variance weighting.

    For each grid cell, sources with finite variance and valid data are
    combined as:

        fused = Σ (x_i / σ_i²) / Σ (1 / σ_i²)
        fused_variance = 1 / Σ (1 / σ_i²)

    Parameters
    ----------
    estimates : dict
        Maps source name to 2D numpy array of rainfall estimates (same shape).
    variances : dict
        Maps source name to 2D array of error variances (same shape as estimates).
        Can be scalar (broadcast) or array.
    mask : dict, optional
        Maps source name to boolean array (True where data is available).
        If not provided, all non‑NaN values in estimates are considered available.

    Returns
    -------
    fused_mean : np.ndarray
        Fused rainfall estimate.
    fused_variance : np.ndarray
        Variance of the fused estimate.
    source_weights : dict
        Normalized weight maps for each source (for diagnostics).
    """
    shape = next(iter(estimates.values())).shape
    if mask is None:
        mask = {name: ~np.isnan(arr) for name, arr in estimates.items()}

    # Initialize sums
    sum_weights = np.zeros(shape, dtype=np.float64)
    sum_weighted_values = np.zeros(shape, dtype=np.float64)
    source_weights = {name: np.zeros(shape, dtype=np.float64) for name in estimates}

    for name in estimates:
        if name not in variances:
            raise ValueError(f"Variance not provided for source '{name}'")
        var = variances[name]
        # Ensure variance is array-like
        var = np.full(shape, var) if np.isscalar(var) else var
        # Use only where mask is True and variance > 0
        valid = mask[name] & (var > 0)
        weights = np.where(valid, 1.0 / var, 0.0)
        sum_weights += weights
        sum_weighted_values += weights * estimates[name]
        source_weights[name] = weights

    # Avoid division by zero
    fused_mean = np.divide(sum_weighted_values, sum_weights,
                           out=np.full(shape, np.nan), where=sum_weights > 0)
    fused_variance = np.divide(1.0, sum_weights,
                               out=np.full(shape, np.nan), where=sum_weights > 0)

    # Normalize weights for diagnostics (per cell)
    for name in source_weights:
        source_weights[name] = np.divide(source_weights[name], sum_weights,
                                         out=np.zeros(shape), where=sum_weights > 0)

    return fused_mean, fused_variance, source_weights


# ----------------------------------------------------------------------
# 2. Bayesian Sequential Fusion (Kalman filter style)
# ----------------------------------------------------------------------
def bayesian_sequential_fusion(
    prior_mean: np.ndarray,
    prior_variance: np.ndarray,
    updates: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Bayesian updating of a prior (e.g., NWP) with additional
    observations (radar, gauge, satellite) treated as independent likelihoods.

    Each update is a tuple (likelihood_mean, likelihood_variance, mask).
    The posterior after each update becomes the prior for the next.

    For Gaussian distributions, the update equations are:

        posterior_variance = 1 / (1/prior_variance + 1/likelihood_variance)
        posterior_mean = posterior_variance * (prior_mean / prior_variance +
                                                likelihood_mean / likelihood_variance)

    Parameters
    ----------
    prior_mean : np.ndarray
        Prior mean estimate (2D).
    prior_variance : np.ndarray
        Prior variance (2D).
    updates : list of tuples
        Each tuple: (likelihood_mean, likelihood_variance, mask)
        where mask is boolean array (True where observation available).

    Returns
    -------
    posterior_mean : np.ndarray
        Updated mean.
    posterior_variance : np.ndarray
        Updated variance.
    """
    post_mean = prior_mean.copy()
    post_var = prior_variance.copy()

    for lik_mean, lik_var, mask in updates:
        # Only update where mask True and variances finite
        update_mask = mask & (post_var > 0) & (lik_var > 0)
        # Compute new variance
        new_var = np.divide(1.0,
                            1.0/post_var + 1.0/lik_var,
                            out=np.full_like(post_var, np.nan),
                            where=update_mask)
        # Compute new mean
        new_mean = np.divide(
            post_mean/post_var + lik_mean/lik_var,
            1.0/post_var + 1.0/lik_var,
            out=np.full_like(post_mean, np.nan),
            where=update_mask
        )
        # Apply updates only where mask True
        post_mean = np.where(update_mask, new_mean, post_mean)
        post_var = np.where(update_mask, new_var, post_var)

    return post_mean, post_var


# ----------------------------------------------------------------------
# 3. Source Availability and Disagreement
# ----------------------------------------------------------------------
def source_agreement_diagnostic(
    estimates: Dict[str, np.ndarray],
    mask: Optional[Dict[str, np.ndarray]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per‑cell source availability count and inter‑source variance.

    The inter‑source variance measures how much the available sources
    disagree. High variance indicates active conflict among sources,
    which may undermine confidence even if the fused mean is well‑constrained.

    Parameters
    ----------
    estimates : dict
        Maps source name to array.
    mask : dict, optional
        Availability masks. If None, uses non‑NaN values.

    Returns
    -------
    availability_count : np.ndarray
        Number of sources available at each cell.
    inter_source_variance : np.ndarray
        Variance of the available estimates (unweighted).
    """
    if mask is None:
        mask = {name: ~np.isnan(arr) for name, arr in estimates.items()}

    shape = next(iter(estimates.values())).shape
    stack = []
    for name in estimates:
        arr = np.where(mask[name], estimates[name], np.nan)
        stack.append(arr)
    stack = np.stack(stack, axis=0)  # (n_sources, H, W)
    availability_count = np.sum(~np.isnan(stack), axis=0)
    inter_source_variance = np.nanvar(stack, axis=0)
    # If only one source, variance is 0
    inter_source_variance = np.where(availability_count > 1, inter_source_variance, np.nan)
    return availability_count.astype(int), inter_source_variance


# ----------------------------------------------------------------------
# 4. Confidence Level
# ----------------------------------------------------------------------
def compute_confidence(
    fused_variance: np.ndarray,
    availability_count: np.ndarray,
    inter_source_variance: np.ndarray,
    max_variance: float = 10.0,
) -> np.ndarray:
    """
    Convert fused variance, source count, and disagreement into a confidence
    score in [0,1].

    A high confidence requires low fused variance, multiple sources, and low
    disagreement. The exact formula is heuristic but interpretable.

    Parameters
    ----------
    fused_variance : np.ndarray
        Variance of fused estimate.
    availability_count : np.ndarray
        Number of sources used.
    inter_source_variance : np.ndarray
        Variance among the sources (disagreement).
    max_variance : float
        Variance above which confidence is considered zero.

    Returns
    -------
    np.ndarray
        Confidence level (0‑1).
    """
    # Confidence from variance (lower variance -> higher confidence)
    conf_var = np.clip(1.0 - fused_variance / max_variance, 0, 1)
    # Confidence from source count (more sources -> higher confidence)
    conf_count = np.clip((availability_count - 1) / 3.0, 0, 1)  # 1..4 -> 0..1
    # Confidence from agreement (low disagreement -> high confidence)
    # Use a logistic-style mapping; max expected variance set to 25 (mm/hr)^2
    max_disagree = 2500.0  # (mm/day)² — daily rainfall spread among sources
    conf_agree = np.clip(1.0 - inter_source_variance / max_disagree, 0, 1)
    # Combine (geometric mean gives balanced penalty)
    confidence = np.power(conf_var * conf_count * conf_agree, 1/3)
    return confidence


# ----------------------------------------------------------------------
# 5. Risk Score Composition
# ----------------------------------------------------------------------
def compose_risk_score(
    fused_rainfall: np.ndarray,
    confidence: np.ndarray,
    flood_risk_category: Optional[np.ndarray] = None,
    alert_tier: Optional[np.ndarray] = None,
    weights: Dict[str, float] = None,
) -> np.ndarray:
    """
    Combine fused rainfall (normalized to 0‑100), flood risk category,
    and alert tier into a single interpretable risk score.

    Parameters
    ----------
    fused_rainfall : np.ndarray
        Fused rainfall (mm). Will be normalized relative to a typical
        heavy rain threshold (e.g., 100 mm).
    confidence : np.ndarray
        Confidence level (0‑1). Used to modulate the final risk: lower
        confidence reduces the score to reflect uncertainty.
    flood_risk_category : np.ndarray, optional
        Categorical risk 0‑3 from inundation module.
    alert_tier : np.ndarray, optional
        Alert level as numeric 0‑3 (Green=0, Yellow=1, Orange=2, Red=3).
    weights : dict, optional
        Weights for the components: {'rainfall', 'flood', 'alert'}.
        Defaults: rainfall=0.5, flood=0.3, alert=0.2.

    Returns
    -------
    np.ndarray
        Risk score (0‑100). Confidence is already factored in.
    """
    if weights is None:
        weights = {'rainfall': 0.5, 'flood': 0.3, 'alert': 0.2}

    # Normalize rainfall to 0‑100 using a threshold of 100 mm
    rainfall_score = np.clip(fused_rainfall / 100.0 * 100, 0, 100)

    # Flood risk score: map 0‑3 to 0‑100
    if flood_risk_category is not None:
        flood_score = flood_risk_category * (100 / 3)
    else:
        flood_score = np.zeros_like(rainfall_score)

    # Alert tier score: map 0‑3 to 0‑100
    if alert_tier is not None:
        alert_score = alert_tier * (100 / 3)
    else:
        alert_score = np.zeros_like(rainfall_score)

    # Weighted sum
    raw_risk = (weights['rainfall'] * rainfall_score +
                weights['flood'] * flood_score +
                weights['alert'] * alert_score)

    # Modulate by confidence: lower confidence reduces risk score
    risk_score = raw_risk * confidence

    return np.clip(risk_score, 0, 100)


# ----------------------------------------------------------------------
# 6. Validation: compare fused vs individual sources
# ----------------------------------------------------------------------
def validate_fusion(
    fused_estimate: np.ndarray,
    individual_estimates: Dict[str, np.ndarray],
    observed: np.ndarray,
    mask: Optional[Dict[str, np.ndarray]] = None,
) -> pd.DataFrame:
    """
    Compute RMSE for the fused estimate and each individual source.

    Parameters
    ----------
    fused_estimate : np.ndarray
        Fused rainfall estimate.
    individual_estimates : dict
        Source name -> array (same shape as observed).
    observed : np.ndarray
        Ground truth observations.
    mask : dict, optional
        Availability masks. Only cells where a source is available are
        included in that source's RMSE calculation.

    Returns
    -------
    pd.DataFrame
        RMSE for each source plus the fused estimate.
    """
    rmse = {}
    # Fused RMSE: all cells where fused is not NaN
    valid = ~np.isnan(fused_estimate) & ~np.isnan(observed)
    rmse['fused'] = np.sqrt(np.mean((fused_estimate[valid] - observed[valid])**2))

    for name, est in individual_estimates.items():
        if mask is not None:
            valid = mask[name] & ~np.isnan(est) & ~np.isnan(observed)
        else:
            valid = ~np.isnan(est) & ~np.isnan(observed)
        if valid.sum() > 0:
            rmse[name] = np.sqrt(np.mean((est[valid] - observed[valid])**2))
        else:
            rmse[name] = np.nan
    return pd.DataFrame.from_dict(rmse, orient='index', columns=['RMSE'])


# ----------------------------------------------------------------------
# 7. Visualization
# ----------------------------------------------------------------------
def plot_risk_confidence(risk: np.ndarray, confidence: np.ndarray,
                         title: str = 'Fused Risk and Confidence'):
    """
    Plot risk score with confidence overlay as hatching/transparency.

    Parameters
    ----------
    risk : np.ndarray
        2D array of risk score (0‑100).
    confidence : np.ndarray
        2D array of confidence (0‑1).
    """
    fig, ax = plt.subplots(figsize=(8,6))
    # Plot risk as color
    im = ax.imshow(risk, origin='lower', cmap='YlOrRd', vmin=0, vmax=100)
    ax.set_title(title)
    # Overlay confidence as hatching: low confidence = sparse dots
    # Create a mask where confidence < 0.5
    low_conf = confidence < 0.5
    if low_conf.any():
        # Use a white hatch to indicate lower confidence
        ax.contourf(low_conf.astype(float), levels=[0.5, 1.5],
                    colors='none', hatches=['..'], alpha=0.5)
    fig.colorbar(im, ax=ax, label='Risk Score')
    # Add a note about hatching
    ax.text(0.02, 0.98, 'Hatched = low confidence', transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# Main demonstration with synthetic data
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a simple 10x10 grid
    np.random.seed(42)
    shape = (10, 10)
    # Simulate NWP, radar, satellite, gauge estimates
    nwp = np.random.normal(5, 3, shape)
    radar = np.random.normal(6, 1.5, shape)  # more accurate
    satellite = np.random.normal(4, 2.5, shape)
    gauge = np.random.normal(5.5, 1.0, shape)  # most accurate where present

    # Introduce missing data: gauge only at a few points, radar no coverage beyond range
    mask_gauge = np.zeros(shape, dtype=bool)
    mask_gauge[2, 2] = True
    mask_gauge[7, 3] = True
    mask_gauge[5, 8] = True
    mask_radar = (np.indices(shape)[1] < 7)  # radar covers left side

    # Replace missing with NaN
    radar[~mask_radar] = np.nan
    gauge[~mask_gauge] = np.nan

    # Build dictionaries
    estimates = {
        'nwp': nwp,
        'radar': radar,
        'satellite': satellite,
        'gauge': gauge,
    }
    variances = {
        'nwp': 9.0,
        'radar': 2.25,
        'satellite': 4.0,
        'gauge': 1.0,
    }
    mask = {
        'nwp': np.ones(shape, dtype=bool),
        'radar': mask_radar,
        'satellite': np.ones(shape, dtype=bool),
        'gauge': mask_gauge,
    }

    # Inverse variance fusion
    fused_mean, fused_var, weights = inverse_variance_fusion(estimates, variances, mask)
    avail_count, inter_var = source_agreement_diagnostic(estimates, mask)
    confidence = compute_confidence(fused_var, avail_count, inter_var)

    # Risk score (synthetic flood risk and alert tier)
    flood_risk = np.random.randint(0, 4, shape)  # 0-3
    alert_tier = np.random.randint(0, 4, shape)
    risk = compose_risk_score(fused_mean, confidence, flood_risk, alert_tier)

    # Plot
    plot_risk_confidence(risk, confidence, title='Demo Fused Risk and Confidence')

    # Validation (synthetic truth)
    truth = np.random.normal(5.5, 1.0, shape)
    rmse_df = validate_fusion(fused_mean, estimates, truth, mask)
    print("RMSE comparison:")
    print(rmse_df)

    # Source disagreement example: print max disagreement cells
    max_disagreement = np.unravel_index(np.nanargmax(inter_var), shape)
    print(f"Cell with highest disagreement: {max_disagreement}, "
          f"inter-source variance = {inter_var[max_disagreement]:.2f}")