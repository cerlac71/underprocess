from alerts.alert_thresholds import AlertThresholdModule, combined_alert_level
from alerts.notification_service import NotificationConfig, NotificationService, format_summary_message
from alerts.subscriptions import Subscription, SubscriptionStore

DEFAULT_THRESHOLDS = {"watch": 15.0, "warning": 30.0, "severe": 50.0}


def classify_alert_level(rainfall_mm: float, thresholds: dict | None = None) -> str:
    thresholds = thresholds or DEFAULT_THRESHOLDS
    if rainfall_mm >= thresholds["severe"]:
        return "Severe"
    if rainfall_mm >= thresholds["warning"]:
        return "Warning"
    if rainfall_mm >= thresholds["watch"]:
        return "Watch"
    return "Normal"


__all__ = [
    "AlertThresholdModule",
    "DEFAULT_THRESHOLDS",
    "NotificationConfig",
    "NotificationService",
    "Subscription",
    "SubscriptionStore",
    "classify_alert_level",
    "combined_alert_level",
    "format_summary_message",
]
