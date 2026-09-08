"""User alert subscriptions for district-level notifications."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Subscription:
    id: str
    district: str
    email: str | None = None
    phone: str | None = None
    webhook_url: str | None = None
    min_level: str = "Yellow"  # Green, Yellow, Orange, Red
    notify_flood: bool = True
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def matches(self, alert_row: dict[str, Any]) -> bool:
        if alert_row.get("district", "").lower() != self.district.lower():
            return False
        level_rank = {"Green": 0, "Yellow": 1, "Orange": 2, "Red": 3}
        if level_rank.get(alert_row.get("alert_level", "Green"), 0) < level_rank.get(self.min_level, 1):
            if not (self.notify_flood and alert_row.get("flood_risk", 0) >= 1):
                return False
        return bool(self.email or self.phone or self.webhook_url)


class SubscriptionStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, rows: list[dict[str, Any]]) -> None:
        self.path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    def list_all(self) -> list[Subscription]:
        return [Subscription(**row) for row in self._load()]

    def add(
        self,
        district: str,
        email: str | None = None,
        phone: str | None = None,
        webhook_url: str | None = None,
        min_level: str = "Yellow",
        notify_flood: bool = True,
    ) -> Subscription:
        sub = Subscription(
            id=str(uuid.uuid4())[:8],
            district=district,
            email=email or None,
            phone=phone or None,
            webhook_url=webhook_url or None,
            min_level=min_level,
            notify_flood=notify_flood,
        )
        rows = self._load()
        rows.append(asdict(sub))
        self._save(rows)
        return sub

    def remove(self, sub_id: str) -> bool:
        rows = self._load()
        new_rows = [r for r in rows if r.get("id") != sub_id]
        if len(new_rows) == len(rows):
            return False
        self._save(new_rows)
        return True

    def find_matching(self, alert_row: dict[str, Any]) -> list[Subscription]:
        return [s for s in self.list_all() if s.matches(alert_row)]
