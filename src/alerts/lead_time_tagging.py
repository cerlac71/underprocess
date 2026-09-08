#!/usr/bin/env python3
"""
Lead‑Time Tagging and Messaging Module for SIH26071.

This module explicitly surfaces lead‑time information on every alert object
produced by the pipeline. It defines a consistent schema, computes lead_time
and lead_time_bucket, generates human‑readable messages, and provides
visualization/tracking utilities.

Design notes:
- Lead time buckets are chosen to reflect common warning windows:
  0‑1 hr: Immediate, 1‑6 hr: Short‑term, 6‑24 hr: Medium‑term, 24‑72 hr:
  Extended. These align with operational planning horizons.
- The module does not perform any ML; it is purely plumbing and
  communication, ensuring that the early‑warning aspect of the system is
  clearly expressed to responders.
- Skill‑vs‑lead‑time data can be supplied from earlier validation modules
  to attach an expected accuracy to each bucket.

Author: SIH26071 Team
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Lead time bucket definitions
# ----------------------------------------------------------------------
LEAD_TIME_BUCKETS = {
    "immediate": (0, 1),      # 0 to <1 hour
    "short_term": (1, 6),     # 1 to <6 hours
    "medium_term": (6, 24),   # 6 to <24 hours
    "extended": (24, 72),     # 24 to <72 hours
}

# Bucket ordering for sorting
BUCKET_ORDER = ["immediate", "short_term", "medium_term", "extended"]

def get_lead_time_bucket(lead_time_hours: float) -> str:
    """
    Determine the lead time bucket from a numeric lead time in hours.

    Parameters
    ----------
    lead_time_hours : float
        Lead time in hours (may be fractional).

    Returns
    -------
    str
        Bucket name (one of 'immediate', 'short_term', 'medium_term',
        'extended').
    """
    for bucket, (lower, upper) in LEAD_TIME_BUCKETS.items():
        if lower <= lead_time_hours < upper:
            return bucket
    # If beyond 72 hours, we treat as 'extended' but with a warning
    logger.warning(f"Lead time {lead_time_hours}h exceeds 72h; treating as 'extended'.")
    return "extended"


# ----------------------------------------------------------------------
# Alert dataclass
# ----------------------------------------------------------------------
@dataclass
class AlertWithLeadTime:
    """
    Alert object with explicit lead time metadata.

    Attributes
    ----------
    issue_time : datetime
        When the forecast was generated.
    valid_time : datetime
        When the predicted event is expected to occur.
    lead_time : timedelta
        valid_time - issue_time (computed).
    lead_time_hours : float
        Lead time in hours (numeric).
    lead_time_bucket : str
        Categorical bucket (see LEAD_TIME_BUCKETS).
    source_module : str
        Which upstream module produced this alert
        (e.g., 'nowcasting', 'nwp_bias_correction').
    alert_level : str
        Alert level ('Green', 'Yellow', 'Orange', 'Red').
    confidence : float
        Confidence level from fusion (0-1).
    risk_score : Optional[float]
        Fused risk score (0-100).
    message : str
        Human‑readable alert message (generated after initialization).
    skill_estimate : Optional[float]
        Expected skill (e.g., CSI) for this lead time bucket, if available.
    """

    issue_time: datetime
    valid_time: datetime
    source_module: str
    alert_level: str
    confidence: float
    risk_score: Optional[float] = None
    lead_time: timedelta = field(init=False)
    lead_time_hours: float = field(init=False)
    lead_time_bucket: str = field(init=False)
    message: str = field(init=False)
    skill_estimate: Optional[float] = None

    def __post_init__(self):
        self.lead_time = self.valid_time - self.issue_time
        self.lead_time_hours = self.lead_time.total_seconds() / 3600.0
        self.lead_time_bucket = get_lead_time_bucket(self.lead_time_hours)
        self.message = generate_alert_message(self)


# ----------------------------------------------------------------------
# Message generation
# ----------------------------------------------------------------------
def generate_alert_message(alert: 'AlertWithLeadTime') -> str:
    """
    Generate a human‑readable alert message based on lead time bucket,
    confidence, and alert level.

    The message frames the lead time in actionable terms: urgency for
    immediate threats, preparation for short‑term, situational awareness
    for extended.

    Parameters
    ----------
    alert : AlertWithLeadTime
        The alert object.

    Returns
    -------
    str
        A string message suitable for SMS, dashboard, or voice alert.
    """
    # Determine confidence wording
    if alert.confidence >= 0.8:
        conf_text = "high confidence"
    elif alert.confidence >= 0.5:
        conf_text = "moderate confidence"
    else:
        conf_text = "preliminary, monitor for updates"

    # Lead time phrasing
    if alert.lead_time_hours < 1:
        time_phrase = f"in approximately {int(alert.lead_time_hours * 60)} minutes"
    elif alert.lead_time_hours < 24:
        time_phrase = f"in approximately {int(alert.lead_time_hours)} hours"
    else:
        days = alert.lead_time_hours / 24
        time_phrase = f"in approximately {days:.1f} days"

    # Bucket‑specific framing
    if alert.lead_time_bucket == "immediate":
        action = "Take immediate protective action."
    elif alert.lead_time_bucket == "short_term":
        action = "Prepare to take action shortly."
    elif alert.lead_time_bucket == "medium_term":
        action = "Review emergency plans and monitor updates."
    else:
        action = "Maintain situational awareness and continue monitoring."

    # Combine with alert level
    level_text = alert.alert_level.upper()
    if alert.alert_level in ["Red", "Severe"]:
        severity = "Severe"
    elif alert.alert_level in ["Orange", "Warning"]:
        severity = "Warning"
    elif alert.alert_level in ["Yellow", "Watch"]:
        severity = "Watch"
    else:
        severity = "Notice"

    # Construct message
    msg = (f"{severity} alert: {level_text} conditions expected {time_phrase}. "
           f"{action} This forecast is from {alert.source_module} with {conf_text}.")
    if alert.skill_estimate is not None:
        msg += f" Historical skill at this lead time is {alert.skill_estimate:.2f} (0‑1 scale)."
    return msg


# ----------------------------------------------------------------------
# Skill tracking
# ----------------------------------------------------------------------
class LeadTimeSkillTracker:
    """
    Stores and plots forecast skill (CSI, FSS, etc.) as a function of
    lead time bucket.

    The data is typically derived from earlier validation modules.
    """

    def __init__(self):
        self.skill_data: Dict[str, Dict[str, float]] = {}
        # Structure: {bucket_name: {'csi': value, 'fss': value, 'rmse': value}}

    def add_skill(self, bucket: str, csi: Optional[float] = None,
                  fss: Optional[float] = None, rmse: Optional[float] = None):
        """Add or update skill metrics for a bucket."""
        if bucket not in self.skill_data:
            self.skill_data[bucket] = {}
        if csi is not None:
            self.skill_data[bucket]['csi'] = csi
        if fss is not None:
            self.skill_data[bucket]['fss'] = fss
        if rmse is not None:
            self.skill_data[bucket]['rmse'] = rmse

    def get_skill(self, bucket: str, metric: str = 'csi') -> Optional[float]:
        """Retrieve a specific skill metric for a bucket."""
        return self.skill_data.get(bucket, {}).get(metric)

    def plot_skill_vs_lead_time(self, metric: str = 'csi',
                                title: str = 'Forecast Skill vs Lead Time'):
        """
        Plot the skill metric across lead time buckets.

        Parameters
        ----------
        metric : str
            'csi', 'fss', or 'rmse'.
        title : str
            Plot title.
        """
        buckets = BUCKET_ORDER
        values = []
        for b in buckets:
            val = self.get_skill(b, metric)
            values.append(val if val is not None else np.nan)

        plt.figure(figsize=(8, 4))
        plt.plot(range(len(buckets)), values, marker='o', linestyle='-',
                 label=metric.upper())
        plt.xticks(range(len(buckets)), [b.replace('_', ' ').title() for b in buckets])
        plt.xlabel('Lead Time Bucket')
        plt.ylabel(metric.upper())
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# ----------------------------------------------------------------------
# Timeline visualization
# ----------------------------------------------------------------------
def plot_alert_timeline(alerts: List[AlertWithLeadTime],
                        region_name: str = "Region"):
    """
    Plot a timeline of alerts for a single region over the forecast horizon.

    Each alert is represented as a colored marker at its valid_time,
    with color indicating alert level.

    Parameters
    ----------
    alerts : list of AlertWithLeadTime
        Alert objects for the same region, sorted by valid_time.
    region_name : str
        Name or identifier of the region.
    """
    if not alerts:
        logger.warning("No alerts provided for timeline.")
        return

    # Sort by valid_time
    alerts_sorted = sorted(alerts, key=lambda a: a.valid_time)

    # Define alert level colors
    level_colors = {
        'Green': 'green',
        'Yellow': 'yellow',
        'Orange': 'orange',
        'Red': 'red',
    }

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 3))
    for alert in alerts_sorted:
        color = level_colors.get(alert.alert_level, 'blue')
        ax.axvline(x=alert.valid_time, color=color, alpha=0.5, linewidth=2)
        ax.scatter(alert.valid_time, 1, color=color, s=100, zorder=5)
        # Annotate with lead time bucket and confidence
        label = f"{alert.lead_time_bucket}\nconf={alert.confidence:.2f}"
        ax.annotate(label, xy=(alert.valid_time, 1), xytext=(0, 20),
                    textcoords='offset points', ha='center', fontsize=8)

    ax.set_yticks([])
    ax.set_xlabel('Valid Time')
    ax.set_title(f'Alert Timeline for {region_name}')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# Filtering utilities
# ----------------------------------------------------------------------
def filter_alerts_by_bucket(alerts: List[AlertWithLeadTime],
                            bucket: str) -> List[AlertWithLeadTime]:
    """Return alerts that belong to a specific lead time bucket."""
    return [a for a in alerts if a.lead_time_bucket == bucket]


def sort_alerts_by_lead_time(alerts: List[AlertWithLeadTime]) -> List[AlertWithLeadTime]:
    """Sort alerts by lead time (ascending)."""
    return sorted(alerts, key=lambda a: a.lead_time_hours)


# ----------------------------------------------------------------------
# Example usage / demonstration
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a skill tracker with demo data
    tracker = LeadTimeSkillTracker()
    tracker.add_skill("immediate", csi=0.7, fss=0.9, rmse=2.0)
    tracker.add_skill("short_term", csi=0.5, fss=0.8, rmse=3.5)
    tracker.add_skill("medium_term", csi=0.3, fss=0.6, rmse=5.0)
    tracker.add_skill("extended", csi=0.2, fss=0.5, rmse=7.0)

    # Plot skill degradation
    tracker.plot_skill_vs_lead_time(metric='csi')

    # Create a few alerts
    now = datetime(2025, 1, 1, 0, 0)
    alerts = [
        AlertWithLeadTime(
            issue_time=now,
            valid_time=now + timedelta(minutes=30),
            source_module='nowcasting',
            alert_level='Red',
            confidence=0.9,
            risk_score=85
        ),
        AlertWithLeadTime(
            issue_time=now,
            valid_time=now + timedelta(hours=3),
            source_module='nowcasting',
            alert_level='Orange',
            confidence=0.8,
            risk_score=70
        ),
        AlertWithLeadTime(
            issue_time=now,
            valid_time=now + timedelta(hours=12),
            source_module='nwp_bias_correction',
            alert_level='Yellow',
            confidence=0.6,
            risk_score=50
        ),
        AlertWithLeadTime(
            issue_time=now,
            valid_time=now + timedelta(hours=48),
            source_module='nwp_bias_correction',
            alert_level='Yellow',
            confidence=0.4,
            risk_score=40
        ),
    ]

    # Attach skill estimates
    for alert in alerts:
        alert.skill_estimate = tracker.get_skill(alert.lead_time_bucket, 'csi')

    # Print messages
    for alert in alerts:
        print(f"Lead {alert.lead_time_hours:.1f}h, bucket={alert.lead_time_bucket}, "
              f"level={alert.alert_level}, conf={alert.confidence:.2f}")
        print(alert.message)
        print("---")

    # Filter immediate alerts
    immediate = filter_alerts_by_bucket(alerts, 'immediate')
    print(f"Number of immediate alerts: {len(immediate)}")

    # Plot timeline
    plot_alert_timeline(alerts, region_name="Demo Region")