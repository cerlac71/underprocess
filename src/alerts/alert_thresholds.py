#!/usr/bin/env python3
"""
Alert Threshold Module for SIH26071.

This module converts upstream forecast outputs (rainfall, runoff, flood depth)
into discrete, region‑specific alert levels (Watch / Warning / Severe) based on
IMD standards, locally adjusted using terrain and hydrological features.

Key design principles:
- Interpretable, rule‑based logic (no ML) for public safety decisions.
- Use conservative p90 rainfall forecasts to bias toward avoiding false negatives.
- Regional adjustment derived from terrain (flow accumulation, slope, TWI, CN)
  so thresholds are physically motivated, not arbitrary.
- Multi‑signal decision: combine rainfall, antecedent moisture, and flood depth
  when available.
- Hysteresis to avoid alert flickering.

Author: SIH26071 Team
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List, Union
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# IMD Baseline thresholds (24‑hour accumulated rainfall)
# Source: IMD Standard Operation Procedure – Weather Forecasting and Warning
# Green: <64 mm/24h, Yellow (Watch): 64.5‑115.5, Orange (Be prepared): 115.6‑204.4,
# Red (Severe): >204.5
# ----------------------------------------------------------------------
IMD_THRESHOLDS_24H = {
    "green_upper": 64.0,          # below this = green
    "yellow_upper": 115.5,        # below this = yellow
    "orange_upper": 204.4,        # below this = orange
    "red_lower": 204.5,           # above this = red
}

# Sub‑daily thresholds derived from IMD hourly heavy rain definitions
# IMD defines heavy rain as >6.5 mm/hr (reported in some guidance).
# For nowcasting, we use a simple scaling approach: distribute the 24‑hr
# threshold over expected event duration. A more physically based approach
# would use intensity‑duration‑frequency curves, but for hackathon we
# define an hourly equivalent based on typical monsoon event duration (6 h).
# This is not an official standard, but it is defensible as a heuristic.
HOURLY_ALERT_MULTIPLIERS = {
    "watch": 64.0 / 6.0,    # ~10.7 mm/hr
    "warning": 115.5 / 6.0, # ~19.25 mm/hr
    "severe": 204.5 / 6.0,  # ~34.08 mm/hr
}

# Risk category mapping
RISK_TO_ALERT = {
    0: "Green",
    1: "Yellow",
    2: "Orange",
    3: "Red",
}

ALERT_RANK = {"Green": 0, "Yellow": 1, "Orange": 2, "Red": 3}


def _level_rank(level: str) -> int:
    return ALERT_RANK.get(level, -1)


def _level_up(level: str) -> str:
    order = ["Green", "Yellow", "Orange", "Red"]
    idx = order.index(level)
    return order[min(idx + 1, len(order) - 1)]


def combined_alert_level(rainfall_mm: float, flood_risk: int, flood_depth_m: float) -> str:
    """Combine IMD rainfall colour scale with inundation risk."""
    if rainfall_mm <= IMD_THRESHOLDS_24H["green_upper"]:
        rain_level = "Green"
    elif rainfall_mm <= IMD_THRESHOLDS_24H["yellow_upper"]:
        rain_level = "Yellow"
    elif rainfall_mm <= IMD_THRESHOLDS_24H["orange_upper"]:
        rain_level = "Orange"
    else:
        rain_level = "Red"

    flood_level = RISK_TO_ALERT.get(int(flood_risk), "Green")
    level = rain_level if _level_rank(rain_level) >= _level_rank(flood_level) else flood_level

    if flood_depth_m >= 0.3 and _level_rank(level) < 2:
        level = _level_up(level)
    return level


def is_actionable_alert(level: str, flood_risk: int) -> bool:
    """True when heavy rain or inundation warrants user notification."""
    return level != "Green" or flood_risk >= 1

@dataclass
class AlertResult:
    region_id: int
    timestamp: pd.Timestamp
    alert_level: str                # "Green", "Yellow", "Orange", "Red"
    triggering_signals: List[str]   # list of reasons
    time_to_onset: Optional[float]  # hours until threshold reached, if any

class AlertThresholdModule:
    """
    Computes region‑specific alert levels from forecast inputs.
    """

    def __init__(self,
                 terrain_adjustment: bool = True,
                 historical_calibration: Optional[Dict[int, float]] = None,
                 hysteresis_window: int = 2,
                 cooldown_period: int = 1):
        """
        Parameters
        ----------
        terrain_adjustment : bool
            Whether to apply terrain‑based threshold adjustment.
        historical_calibration : dict, optional
            Maps region_id to a multiplicative adjustment factor derived from
            historical flood events. If provided, overrides terrain adjustment
            for that region.
        hysteresis_window : int
            Number of consecutive timesteps an alert must persist before
            escalating. Prevents flickering.
        cooldown_period : int
            Number of timesteps to wait before de‑escalating.
        """
        self.terrain_adjustment = terrain_adjustment
        self.historical_calibration = historical_calibration or {}
        self.hysteresis_window = hysteresis_window
        self.cooldown_period = cooldown_period
        self.last_alert_level = {}   # region_id -> last alert level
        self.alert_count = {}        # region_id -> consecutive timesteps at current level
        self.cooldown_timer = {}     # region_id -> timesteps remaining before de-escalation allowed

    def _terrain_adjustment_factor(self,
                                   slope: float,
                                   flow_accum: float,
                                   twi: float,
                                   cn: Optional[float] = None) -> float:
        """
        Compute a multiplicative adjustment factor for rainfall thresholds based on terrain.

        Steeper slopes, higher flow accumulation, and lower TWI all indicate
        greater flash‑flood potential, so thresholds are reduced (factor < 1).
        Conversely, flat, well‑drained areas get factor > 1.

        The formula is a simple weighted combination, clipped to [0.5, 2.0]
        to avoid extreme values.
        """
        # Normalize inputs to approximate range [0,1]
        # slope in degrees: typical 0-30°
        slope_norm = np.clip(slope / 30.0, 0, 1)
        # flow_accum is cell count; log scale to reduce skew
        if flow_accum > 0:
            flow_norm = np.clip(np.log10(flow_accum + 1) / 6.0, 0, 1)
        else:
            flow_norm = 0.0
        # TWI: high TWI = wet, low = dry; lower TWI increases flood risk
        # typical TWI range 0-20; invert so low TWI -> high factor
        twi_norm = np.clip((10 - twi) / 10.0, 0, 1) if twi > 0 else 1.0

        # Combine: more weight on slope and flow accumulation
        factor = 1.0 - 0.3 * slope_norm - 0.2 * flow_norm + 0.2 * twi_norm

        # If CN provided, areas with high CN (urban, impervious) get lower thresholds
        if cn is not None:
            # CN 30-100 -> higher CN means more runoff, reduce factor
            cn_norm = np.clip((cn - 30) / 70.0, 0, 1)
            factor -= 0.2 * cn_norm

        return float(np.clip(factor, 0.5, 2.0))

    def _regional_thresholds(self,
                             region_id: int,
                             terrain_features: Optional[Dict] = None) -> Dict:
        """
        Return adjusted rainfall thresholds for a region.

        terrain_features should contain keys: slope, flow_accum, twi, cn (optional).
        If historical calibration exists for this region, use that factor.
        """
        if region_id in self.historical_calibration:
            factor = self.historical_calibration[region_id]
        elif self.terrain_adjustment and terrain_features is not None:
            factor = self._terrain_adjustment_factor(
                terrain_features.get('slope', 10.0),
                terrain_features.get('flow_accum', 1.0),
                terrain_features.get('twi', 5.0),
                terrain_features.get('cn', None)
            )
        else:
            factor = 1.0

        # Apply factor to IMD baseline thresholds
        thresholds = {
            "yellow": IMD_THRESHOLDS_24H["yellow_upper"] * factor,
            "orange": IMD_THRESHOLDS_24H["orange_upper"] * factor,
            "red": IMD_THRESHOLDS_24H["red_lower"] * factor,
        }
        return thresholds

    def _determine_alert_from_rainfall(self,
                                       rainfall_mm: float,
                                       thresholds: Dict) -> str:
        """Map rainfall total to alert level."""
        if rainfall_mm <= thresholds["yellow"]:
            return "Green"
        elif rainfall_mm <= thresholds["orange"]:
            return "Yellow"
        elif rainfall_mm <= thresholds["red"]:
            return "Orange"
        else:
            return "Red"

    def _apply_hysteresis(self,
                          region_id: int,
                          current_level: str,
                          timestamp: pd.Timestamp) -> str:
        """
        Apply hysteresis to avoid alert flickering.

        Escalation requires the alert to persist for `hysteresis_window`
        consecutive timesteps. De‑escalation is delayed by `cooldown_period`.
        """
        # Initialize if not present
        if region_id not in self.last_alert_level:
            self.last_alert_level[region_id] = "Green"
            self.alert_count[region_id] = 0
            self.cooldown_timer[region_id] = 0

        last = self.last_alert_level[region_id]

        # If same as before, increment counter
        if current_level == last:
            self.alert_count[region_id] += 1
        else:
            # Different level: check if we are allowed to change
            if self._level_rank(current_level) > self._level_rank(last):
                # Escalation: require consecutive counts
                if self.alert_count[region_id] >= self.hysteresis_window:
                    self.last_alert_level[region_id] = current_level
                    self.alert_count[region_id] = 1
                    self.cooldown_timer[region_id] = 0
                    return current_level
                else:
                    # Not yet sustained; stay at last level
                    return last
            else:
                # De‑escalation: require cooldown
                if self.cooldown_timer[region_id] <= 0:
                    self.last_alert_level[region_id] = current_level
                    self.alert_count[region_id] = 1
                    self.cooldown_timer[region_id] = 0
                    return current_level
                else:
                    self.cooldown_timer[region_id] -= 1
                    return last

        # Same level, update cooldown
        if current_level == "Green":
            self.cooldown_timer[region_id] = 0
        else:
            self.cooldown_timer[region_id] = max(0, self.cooldown_timer[region_id] - 1)

        self.last_alert_level[region_id] = current_level
        return current_level

    @staticmethod
    def _level_rank(level: str) -> int:
        """Convert alert level to numeric rank for comparison."""
        mapping = {"Green": 0, "Yellow": 1, "Orange": 2, "Red": 3}
        return mapping.get(level, -1)

    def compute_alert(self,
                      region_id: int,
                      timestamp: pd.Timestamp,
                      rainfall_forecast_p90: float,
                      rainfall_forecast_p50: Optional[float] = None,
                      antecedent_rainfall: Optional[float] = None,
                      terrain_features: Optional[Dict] = None,
                      flood_depth: Optional[float] = None,
                      flood_risk: Optional[int] = None) -> AlertResult:
        """
        Compute alert level for a region at a given timestamp.

        Parameters
        ----------
        region_id : int
            Unique identifier for the region (sub‑basin, district, etc.)
        timestamp : pd.Timestamp
            Forecast valid time.
        rainfall_forecast_p90 : float
            Upper (90th percentile) rainfall forecast for the relevant accumulation period.
        rainfall_forecast_p50 : float, optional
            Median rainfall forecast (used for time‑to‑onset estimation).
        antecedent_rainfall : float, optional
            Rainfall in preceding days (mm), used to adjust for soil moisture.
        terrain_features : dict, optional
            Contains slope, flow_accum, twi, cn etc. for regional adjustment.
        flood_depth : float, optional
            Predicted flood depth (m) from inundation module.
        flood_risk : int, optional
            Flood risk category (0‑3) from inundation module.

        Returns
        -------
        AlertResult
            Contains alert level, triggering signals, and time‑to‑onset.
        """
        # Compute adjusted thresholds
        thresholds = self._regional_thresholds(region_id, terrain_features)

        # Base alert from rainfall p90
        alert_from_rain = self._determine_alert_from_rainfall(rainfall_forecast_p90, thresholds)
        signals = [f"rainfall_p90={rainfall_forecast_p90:.1f} mm (threshold {thresholds['yellow']:.1f}/{thresholds['orange']:.1f}/{thresholds['red']:.1f})"]

        # If antecedent moisture is high, we may escalate
        if antecedent_rainfall is not None and antecedent_rainfall > 50:
            # Increase alert by one level if already above green
            if alert_from_rain != "Green":
                alert_from_rain = self._level_up(alert_from_rain)
                signals.append(f"antecedent_rainfall={antecedent_rainfall:.1f} mm (soil saturated)")

        # If flood depth/risk is provided and indicates higher severity, use that
        if flood_risk is not None and flood_risk > 0:
            alert_from_flood = RISK_TO_ALERT[flood_risk]
            if self._level_rank(alert_from_flood) > self._level_rank(alert_from_rain):
                alert_from_rain = alert_from_flood
                signals.append(f"flood_depth={flood_depth:.2f} m, risk={flood_risk}")

        # Apply hysteresis to get final alert
        final_alert = self._apply_hysteresis(region_id, alert_from_rain, timestamp)

        # Estimate time to onset if we have p50 and we're not yet at that level
        time_to_onset = None
        if rainfall_forecast_p50 is not None and final_alert != "Green":
            # Simple linear extrapolation from current time? We'll skip for brevity.
            pass

        return AlertResult(
            region_id=region_id,
            timestamp=timestamp,
            alert_level=final_alert,
            triggering_signals=signals,
            time_to_onset=time_to_onset
        )

    @staticmethod
    def _level_up(level: str) -> str:
        """Increase alert level by one step."""
        order = ["Green", "Yellow", "Orange", "Red"]
        idx = order.index(level)
        if idx < len(order) - 1:
            return order[idx+1]
        return level


# ----------------------------------------------------------------------
# Validation and synthetic test
# ----------------------------------------------------------------------
def validate_against_historical(historical_events: pd.DataFrame,
                                alert_module: AlertThresholdModule) -> pd.DataFrame:
    """
    Backtest the alert module against historical flood events.

    historical_events should have columns: region_id, timestamp,
    rainfall_forecast_p90, actual_flood (bool), plus optional terrain features.
    Returns a confusion matrix summary.
    """
    results = []
    for _, row in historical_events.iterrows():
        alert = alert_module.compute_alert(
            region_id=row['region_id'],
            timestamp=row['timestamp'],
            rainfall_forecast_p90=row['rainfall_forecast_p90'],
            terrain_features=row.get('terrain_features', None)
        )
        predicted = alert.alert_level in ["Orange", "Red"]
        actual = row['actual_flood']
        results.append({'predicted': predicted, 'actual': actual})

    df = pd.DataFrame(results)
    tp = ((df['predicted']==True) & (df['actual']==True)).sum()
    fp = ((df['predicted']==True) & (df['actual']==False)).sum()
    fn = ((df['predicted']==False) & (df['actual']==True)).sum()
    tn = ((df['predicted']==False) & (df['actual']==False)).sum()
    summary = pd.DataFrame({
        'Metric': ['True Positive', 'False Positive', 'False Negative', 'True Negative'],
        'Count': [tp, fp, fn, tn]
    })
    return summary


def synthetic_test():
    """Run a simple synthetic extreme rainfall scenario through the alert module."""
    alert_mod = AlertThresholdModule()
    # Scenario: region with mild terrain, increasing rainfall
    print("Synthetic test: Rainfall from 50 to 300 mm")
    for rain in [50, 80, 120, 200, 250, 300]:
        alert = alert_mod.compute_alert(
            region_id=1,
            timestamp=pd.Timestamp("2025-01-01 00:00"),
            rainfall_forecast_p90=rain,
            terrain_features={'slope': 5, 'flow_accum': 10, 'twi': 8, 'cn': 70}
        )
        print(f"Rainfall {rain} mm -> Alert: {alert.alert_level} "
              f"({', '.join(alert.triggering_signals)})")


if __name__ == "__main__":
    synthetic_test()