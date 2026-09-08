#!/usr/bin/env python3
"""
Rainfall‑to‑Runoff Conversion Module for SIH26071.

This module converts rainfall intensity (mm) into surface runoff (mm) using
two approaches:

1. **SCS Curve Number (CN)** – an industry‑standard empirical method that
   accounts for land use and soil type, with optional antecedent moisture
   condition (AMC) adjustment for multi‑day events.

2. **Machine Learning** – a gradient boosting model (LightGBM) that maps
   terrain and rainfall features to runoff. In the absence of observed
   runoff data, the ML model can be trained to emulate CN outputs (a
   "learned correction"), which is honest and avoids inventing ground truth.

Both methods produce runoff depth grids at the same spatial/temporal
resolution as the input rainfall, and propagate uncertainty by processing
each rainfall quantile separately.

Author: SIH26071 Team
"""

import logging
import numpy as np
import pandas as pd
import xarray as xr
from typing import Optional, Tuple, Dict, List, Union

# Optional ML library
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not installed. ML approach unavailable.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# 1. SCS Curve Number (CN) method
# ----------------------------------------------------------------------
# Standard CN lookup table (simplified: hydrologic soil group A/B/C/D vs
# land use classes). Full NRCS tables are more detailed; this is a
# representative subset for common Indian land covers.
CN_LOOKUP = {
    # Land use: {soil_group: CN}
    'forest':        {'A': 30, 'B': 55, 'C': 70, 'D': 77},
    'grassland':     {'A': 39, 'B': 61, 'C': 74, 'D': 80},
    'agriculture':   {'A': 59, 'B': 71, 'C': 79, 'D': 83},
    'urban_residential': {'A': 61, 'B': 75, 'C': 83, 'D': 87},
    'urban_commercial':  {'A': 81, 'B': 88, 'C': 91, 'D': 93},
    'barren':        {'A': 77, 'B': 86, 'C': 91, 'D': 94},
    'water':         {'A': 100, 'B': 100, 'C': 100, 'D': 100},  # fully impervious
}

def get_curve_number(lulc: np.ndarray,
                     soil_group: Optional[np.ndarray] = None,
                     fallback_soil_group: str = 'C') -> np.ndarray:
    """
    Derive a Curve Number grid from land use/land cover and hydrologic
    soil group (HSG). If soil_group is missing, a uniform fallback
    (default 'C') is used, which is moderately well‑drained – this is
    acceptable for a hackathon demo but should be flagged.

    Parameters
    ----------
    lulc : np.ndarray
        Array of land use classes (strings or integer codes). If strings,
        they must match keys in CN_LOOKUP. If integers, they are mapped
        via LULC_CODE_TO_CLASS (not shown; user should adapt).
    soil_group : np.ndarray, optional
        Array of soil groups ('A','B','C','D') same shape as lulc. If
        None, fallback_soil_group is used for all cells.
    fallback_soil_group : str
        Soil group to use if soil_group not provided.

    Returns
    -------
    np.ndarray
        Curve Number grid (0-100).
    """
    # Ensure lulc is array of strings
    if lulc.dtype.kind in ['U', 'S']:
        lulc_str = lulc.astype(str)
    else:
        # Assume integer codes mapping to LULC classes; user must provide mapping
        raise ValueError("LULC array must be strings or provide a mapping to class names.")

    if soil_group is None:
        soil_group_arr = np.full(lulc_str.shape, fallback_soil_group, dtype='<U1')
    else:
        soil_group_arr = soil_group.astype(str)

    cn = np.zeros_like(lulc_str, dtype=np.float32)
    for i in range(lulc_str.shape[0]):
        for j in range(lulc_str.shape[1]):
            lc = lulc_str[i, j]
            sg = soil_group_arr[i, j]
            if lc in CN_LOOKUP and sg in CN_LOOKUP[lc]:
                cn[i, j] = CN_LOOKUP[lc][sg]
            else:
                # Fallback: use 'agriculture' with soil group C
                cn[i, j] = CN_LOOKUP['agriculture']['C']
                logger.debug(f"Unknown LULC '{lc}' or soil '{sg}', using default CN 79")
    return cn


def adjust_cn_for_amc(cn: np.ndarray,
                      antecedent_rainfall_mm: float,
                      amc_window_days: int = 5) -> np.ndarray:
    """
    Adjust Curve Number for Antecedent Moisture Condition (AMC).

    The CN method's standard CN is for average moisture (AMC II). For
    multi‑day events, the soil may already be wet, increasing runoff. The
    adjustment uses the total rainfall over the preceding `amc_window_days`
    to classify into AMC I (dry), II (average), or III (wet) and apply
    standard adjustment factors.

    Parameters
    ----------
    cn : np.ndarray
        Curve Number array (AMC II).
    antecedent_rainfall_mm : float
        Total rainfall (mm) over the preceding window.
    amc_window_days : int
        Number of days used to compute antecedent rainfall.

    Returns
    -------
    np.ndarray
        Adjusted CN.
    """
    # Standard AMC classes based on 5‑day antecedent rainfall (mm)
    if antecedent_rainfall_mm < 35:
        amc_class = 1  # dry
    elif antecedent_rainfall_mm > 53:
        amc_class = 3  # wet
    else:
        amc_class = 2  # average

    if amc_class == 1:
        # CN_I = CN_II / (2.281 - 0.01281 * CN_II)
        return cn / (2.281 - 0.01281 * cn)
    elif amc_class == 3:
        # CN_III = CN_II / (0.427 + 0.00573 * CN_II)
        return cn / (0.427 + 0.00573 * cn)
    else:
        return cn  # unchanged


def scs_cn_runoff(rainfall_mm: np.ndarray,
                  cn: np.ndarray,
                  ia_factor: float = 0.2,
                  amc_window_days: int = 5,
                  antecedent_rainfall_mm: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute surface runoff depth using the SCS‑CN method.

    Q = (P - Ia)^2 / (P - Ia + S) for P > Ia, else Q = 0.
    S = (25400 / CN) - 254 (in mm)
    Ia = ia_factor * S (traditional: 0.2, revised: 0.05)

    Parameters
    ----------
    rainfall_mm : np.ndarray
        Rainfall depth (mm) for the current time step.
    cn : np.ndarray
        Curve Number grid (AMC II). May be adjusted before calling if
        antecedent moisture is known.
    ia_factor : float
        Initial abstraction fraction. 0.2 (traditional) or 0.05 (revised).
    amc_window_days : int
        Used only if antecedent_rainfall_mm is provided (for adjustment).
    antecedent_rainfall_mm : np.ndarray, optional
        Grid of antecedent rainfall (mm) over `amc_window_days`. If
        provided, CN is adjusted for AMC before computing runoff.

    Returns
    -------
    np.ndarray
        Runoff depth (mm), same shape as rainfall.
    """
    # Apply AMC adjustment if antecedent rainfall provided
    if antecedent_rainfall_mm is not None:
        # For each cell, adjust CN based on its antecedent rainfall
        cn_adj = np.zeros_like(cn, dtype=np.float32)
        for i in range(cn.shape[0]):
            for j in range(cn.shape[1]):
                cn_adj[i, j] = adjust_cn_for_amc(cn[i, j], antecedent_rainfall_mm[i, j],
                                                 amc_window_days)
    else:
        cn_adj = cn

    # Compute S and Ia
    S = (25400.0 / cn_adj) - 254.0
    S = np.where(S <= 0, 0.1, S)  # avoid division by zero or negative
    Ia = ia_factor * S

    # Runoff
    P = rainfall_mm.astype(np.float32)
    Q = np.where(P > Ia,
                 (P - Ia) ** 2 / (P - Ia + S),
                 0.0)
    return Q.astype(np.float32)


# ----------------------------------------------------------------------
# 2. ML‑based runoff mapping
# ----------------------------------------------------------------------
def train_ml_runoff(features: pd.DataFrame,
                    target: np.ndarray,
                    test_frac: float = 0.2,
                    use_cn_target: bool = True) -> 'lgb.LGBMRegressor':
    """
    Train a LightGBM model to predict runoff from terrain and rainfall features.

    Parameters
    ----------
    features : pd.DataFrame
        Feature columns: rainfall, antecedent_rainfall, slope, flow_accum,
        twi, distance_to_stream, lulc_code, soil_code, cn (optional).
    target : np.ndarray
        Runoff depth (mm) per sample. If `use_cn_target` is True, this is
        CN‑derived runoff; otherwise, observed runoff (if available).
    test_frac : float
        Fraction of data reserved for testing (chronological split is
        assumed if data is sorted by time; we simply take last portion).
    use_cn_target : bool
        If True, target is from CN method (simulated); if False, target is
        observed runoff.

    Returns
    -------
    lgb.LGBMRegressor
        Trained model.
    """
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM is required for ML approach.")

    # Chronological split (assumes features are sorted by time)
    n = len(features)
    split_idx = int(n * (1 - test_frac))
    X_train = features.iloc[:split_idx]
    y_train = target[:split_idx]

    model = lgb.LGBMRegressor(
        objective='regression',
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)
    logger.info(f"ML runoff model trained on {len(X_train)} samples.")
    return model


def predict_ml_runoff(model: 'lgb.LGBMRegressor',
                      features: pd.DataFrame) -> np.ndarray:
    """
    Predict runoff using the trained ML model.
    """
    return model.predict(features)


# ----------------------------------------------------------------------
# 3. Uncertainty propagation: run conversion on each quantile
# ----------------------------------------------------------------------
def runoff_with_uncertainty(rainfall_quantiles: Dict[float, np.ndarray],
                            cn: np.ndarray,
                            method: str = 'scs',
                            ml_model: Optional['lgb.LGBMRegressor'] = None,
                            feature_prep_func: Optional[callable] = None,
                            **kwargs) -> Dict[float, np.ndarray]:
    """
    Apply runoff conversion to each rainfall quantile (p10, p50, p90, etc.).

    Parameters
    ----------
    rainfall_quantiles : dict
        Maps quantile level to rainfall grid.
    cn : np.ndarray
        Curve Number grid (for SCS method).
    method : str
        'scs' or 'ml'.
    ml_model : trained ML model (if method='ml').
    feature_prep_func : callable
        Function that takes rainfall grid and returns a DataFrame of features
        for ML inference. Required if method='ml'.
    **kwargs : additional args for scs_cn_runoff (e.g. ia_factor, antecedent_rainfall).

    Returns
    -------
    dict
        Maps quantile level to runoff grid.
    """
    runoff_q = {}
    for q, rain in rainfall_quantiles.items():
        if method == 'scs':
            runoff = scs_cn_runoff(rain, cn, **kwargs)
        elif method == 'ml':
            if ml_model is None or feature_prep_func is None:
                raise ValueError("ML method requires ml_model and feature_prep_func.")
            features = feature_prep_func(rain)
            runoff = predict_ml_runoff(ml_model, features).reshape(rain.shape)
        else:
            raise ValueError("method must be 'scs' or 'ml'")
        runoff_q[q] = runoff
    return runoff_q


# ----------------------------------------------------------------------
# 4. Sanity checks
# ----------------------------------------------------------------------
def sanity_check_runoff(runoff: np.ndarray, rainfall: np.ndarray) -> bool:
    """
    Check for physically impossible runoff values.

    Returns True if all cells satisfy 0 <= runoff <= rainfall, else False.
    """
    mask = ~np.isnan(runoff) & ~np.isnan(rainfall)
    if not np.allclose(runoff[mask], np.clip(runoff[mask], 0, rainfall[mask])):
        logger.error("Runoff exceeds rainfall in some cells – check CN or ML model.")
        return False
    return True


# ----------------------------------------------------------------------
# Demonstration with synthetic data
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # Create synthetic grid
    np.random.seed(42)
    lat = np.linspace(10, 11, 20)
    lon = np.linspace(75, 76, 20)
    lon2d, lat2d = np.meshgrid(lon, lat)

    # Synthetic LULC: mix of classes
    lulc = np.random.choice(['forest', 'agriculture', 'urban_residential',
                             'grassland', 'barren'], size=lat2d.shape).astype(str)
    # Synthetic soil group: assume 'C' for all
    soil_group = np.full(lat2d.shape, 'C', dtype='<U1')

    # Compute CN
    cn = get_curve_number(lulc, soil_group)
    print("Curve Number range:", np.nanmin(cn), "-", np.nanmax(cn))

    # Synthetic rainfall event (e.g., 50 mm over 2 hours)
    rainfall = np.random.uniform(0, 80, size=lat2d.shape)
    rainfall = np.maximum(rainfall, 0)
    # Compute runoff with CN
    runoff = scs_cn_runoff(rainfall, cn, ia_factor=0.2)
    print("Runoff range:", np.nanmin(runoff), "-", np.nanmax(runoff))

    # Sanity check
    print("Sanity check passed:", sanity_check_runoff(runoff, rainfall))

    # Example with AMC adjustment (antecedent rainfall 60 mm)
    antecedent = np.full(lat2d.shape, 60.0)
    runoff_amc = scs_cn_runoff(rainfall, cn, antecedent_rainfall_mm=antecedent)
    print("AMC-adjusted runoff range:", np.nanmin(runoff_amc), "-", np.nanmax(runoff_amc))

    # Show uncertainty propagation (using p10, p50, p90 of rainfall)
    rain_quantiles = {
        0.1: rainfall * 0.8,
        0.5: rainfall,
        0.9: rainfall * 1.2
    }
    runoff_quantiles = runoff_with_uncertainty(rain_quantiles, cn, method='scs')
    for q in runoff_quantiles:
        print(f"Runoff p{int(q*100)} range:", np.nanmin(runoff_quantiles[q]),
              "-", np.nanmax(runoff_quantiles[q]))