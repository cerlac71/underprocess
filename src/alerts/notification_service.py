"""Multi-channel alert delivery: console, email, SMS, webhook, desktop."""

from __future__ import annotations

import json
import logging
import os
import smtplib
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any
from urllib import error, request

logger = logging.getLogger(__name__)


@dataclass
class NotificationConfig:
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = ""
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from_number: str = ""
    webhook_url: str = ""
    desktop_enabled: bool = True

    @classmethod
    def from_env(cls) -> "NotificationConfig":
        return cls(
            smtp_host=os.environ.get("SIH_SMTP_HOST", ""),
            smtp_port=int(os.environ.get("SIH_SMTP_PORT", "587")),
            smtp_user=os.environ.get("SIH_SMTP_USER", ""),
            smtp_password=os.environ.get("SIH_SMTP_PASSWORD", ""),
            smtp_from=os.environ.get("SIH_SMTP_FROM", os.environ.get("SIH_SMTP_USER", "")),
            twilio_account_sid=os.environ.get("SIH_TWILIO_ACCOUNT_SID", ""),
            twilio_auth_token=os.environ.get("SIH_TWILIO_AUTH_TOKEN", ""),
            twilio_from_number=os.environ.get("SIH_TWILIO_FROM_NUMBER", ""),
            webhook_url=os.environ.get("SIH_ALERT_WEBHOOK_URL", ""),
            desktop_enabled=os.environ.get("SIH_DESKTOP_ALERTS", "1").strip().lower() not in ("0", "false"),
        )


def format_district_alert(row: dict[str, Any]) -> str:
    return (
        f"⚠️ {row['alert_level']} — {row['location']}\n"
        f"24h rainfall: {row['rainfall_mm']} mm\n"
        f"Inundation depth: {row['flood_depth_m']} m (risk {row['flood_risk']}/3)\n"
        f"Confidence: {row.get('confidence', 0):.0%}"
    )


def format_summary_message(summary: dict[str, Any]) -> str:
    active = summary.get("active_districts", [])
    if not active:
        return f"✅ {summary.get('region', 'Region')}: No heavy-rain or flood alerts. All districts Green."

    lines = [
        f"🚨 SIH26071 Alert — {summary.get('region', 'Region')}",
        f"Active warnings: {summary.get('active_count', len(active))} / {summary.get('total_districts', '?')}",
        "",
    ]
    for row in sorted(active, key=lambda r: r.get("alert_level", ""), reverse=True)[:15]:
        lines.append(format_district_alert(row))
        lines.append("")
    if len(active) > 15:
        lines.append(f"... and {len(active) - 15} more districts.")
    return "\n".join(lines)


class NotificationService:
    def __init__(self, config: NotificationConfig | None = None, log_path: Path | None = None):
        self.config = config or NotificationConfig.from_env()
        self.log_path = log_path

    def _append_log(self, entry: dict[str, Any]) -> None:
        if not self.log_path:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        history: list[dict[str, Any]] = []
        if self.log_path.exists():
            try:
                history = json.loads(self.log_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                history = []
        history.append(entry)
        history = history[-500:]
        self.log_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

    def send_console(self, message: str) -> bool:
        print(message, file=sys.stderr)
        return True

    def send_desktop(self, title: str, message: str) -> bool:
        if not self.config.desktop_enabled:
            return False
        try:
            if sys.platform == "darwin":
                script = f'display notification "{message[:200]}" with title "{title}" sound name "Glass"'
                subprocess.run(["osascript", "-e", script], check=True, capture_output=True)
                return True
            if sys.platform.startswith("linux"):
                subprocess.run(["notify-send", title, message[:200]], check=True, capture_output=True)
                return True
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            logger.debug("Desktop notification unavailable: %s", exc)
        return False

    def send_email(self, to: str, subject: str, body: str) -> bool:
        if not all([self.config.smtp_host, self.config.smtp_user, self.config.smtp_password, to]):
            logger.warning("Email not configured (set SIH_SMTP_* env vars).")
            return False
        msg = MIMEMultipart()
        msg["From"] = self.config.smtp_from or self.config.smtp_user
        msg["To"] = to
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain", "utf-8"))
        try:
            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port, timeout=30) as server:
                server.starttls()
                server.login(self.config.smtp_user, self.config.smtp_password)
                server.send_message(msg)
            return True
        except smtplib.SMTPException as exc:
            logger.error("Email send failed: %s", exc)
            return False

    def send_sms(self, to: str, body: str) -> bool:
        sid = self.config.twilio_account_sid
        token = self.config.twilio_auth_token
        from_num = self.config.twilio_from_number
        if not all([sid, token, from_num, to]):
            logger.warning("SMS not configured (set SIH_TWILIO_* env vars).")
            return False
        url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
        data = f"From={from_num}&To={to}&Body={body[:1500]}".encode("utf-8")
        req = request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
        try:
            with request.urlopen(req, timeout=30) as resp:
                return resp.status == 201
        except error.URLError as exc:
            # Twilio requires Basic auth — use requests-style auth via urllib
            logger.debug("Twilio urllib failed, retrying with auth: %s", exc)

        import base64

        auth = base64.b64encode(f"{sid}:{token}".encode()).decode()
        req = request.Request(url, data=data, method="POST")
        req.add_header("Authorization", f"Basic {auth}")
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
        try:
            with request.urlopen(req, timeout=30) as resp:
                return resp.status in (200, 201)
        except error.URLError as exc:
            logger.error("SMS send failed: %s", exc)
            return False

    def send_webhook(self, url: str, payload: dict[str, Any]) -> bool:
        if not url:
            return False
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        try:
            with request.urlopen(req, timeout=15) as resp:
                return 200 <= resp.status < 300
        except error.URLError as exc:
            logger.error("Webhook failed: %s", exc)
            return False

    def notify_summary(
        self,
        summary: dict[str, Any],
        channels: list[str] | None = None,
        email_to: str | None = None,
        sms_to: str | None = None,
        webhook_url: str | None = None,
    ) -> dict[str, bool]:
        """Send alert summary on requested channels. Returns success per channel."""
        channels = channels or ["console"]
        message = format_summary_message(summary)
        title = f"Rainfall EWS — {summary.get('active_count', 0)} active alerts"
        results: dict[str, bool] = {}

        if "console" in channels:
            results["console"] = self.send_console(message)

        if "desktop" in channels and summary.get("active_count", 0) > 0:
            results["desktop"] = self.send_desktop(title, message[:500])

        if "email" in channels and email_to:
            results["email"] = self.send_email(
                email_to,
                subject=title,
                body=message,
            )

        if "sms" in channels and sms_to:
            short = message[:300] if len(message) > 300 else message
            results["sms"] = self.send_sms(sms_to, short)

        hook = webhook_url or self.config.webhook_url
        if "webhook" in channels and hook:
            results["webhook"] = self.send_webhook(hook, summary)

        self._append_log(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "active_count": summary.get("active_count", 0),
                "channels": channels,
                "results": results,
            }
        )
        return results

    def notify_subscribers(
        self,
        summary: dict[str, Any],
        subscriptions: list[Any],
        channels: list[str] | None = None,
    ) -> dict[str, int]:
        """Notify each subscription matching active district alerts."""
        sent = {"email": 0, "sms": 0, "webhook": 0}
        active = summary.get("active_districts", [])
        if not active:
            return sent

        for row in active:
            matching = [s for s in subscriptions if s.matches(row)]
            msg = format_district_alert(row)
            for sub in matching:
                ch = channels or ["console"]
                if sub.email and "email" in ch:
                    if self.send_email(sub.email, f"Flood/Rain Alert — {row['district']}", msg):
                        sent["email"] += 1
                if sub.phone and "sms" in ch:
                    if self.send_sms(sub.phone, msg[:300]):
                        sent["sms"] += 1
                if sub.webhook_url and "webhook" in ch:
                    if self.send_webhook(sub.webhook_url, {"alert": row, "summary": summary}):
                        sent["webhook"] += 1
        return sent
