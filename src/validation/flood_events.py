"""Catalogued Uttar Pradesh monsoon flood cases for holdout validation (2019)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FloodEvent:
    id: str
    name: str
    peak_date: str
    window_start: str
    window_end: str
    lat: float
    lon: float
    notes: str


# Dates chosen from IMD RF25 2019 monsoon maxima over UP + documented heavy-rain spells.
FLOOD_EVENTS: tuple[FloodEvent, ...] = (
    FloodEvent(
        id="jul_2019_central",
        name="Central UP heavy rainfall — 10 Jul 2019",
        peak_date="2019-07-10",
        window_start="2019-07-08",
        window_end="2019-07-11",
        lat=26.85,
        lon=80.95,
        notes="State-wide monsoon peak; Lucknow / central belt.",
    ),
    FloodEvent(
        id="jul_2019_west",
        name="Western UP spell — 11 Jul 2019",
        peak_date="2019-07-11",
        window_start="2019-07-09",
        window_end="2019-07-12",
        lat=26.45,
        lon=80.33,
        notes="Consecutive extreme rainfall day following 10 Jul peak.",
    ),
    FloodEvent(
        id="aug_2019_east",
        name="Eastern UP rainfall — 14 Aug 2019",
        peak_date="2019-08-14",
        window_start="2019-08-12",
        window_end="2019-08-15",
        lat=26.76,
        lon=83.37,
        notes="Gorakhpur / eastern districts; pipeline holdout window.",
    ),
    FloodEvent(
        id="sep_2019_east",
        name="Late monsoon eastern UP — 28 Sep 2019",
        peak_date="2019-09-28",
        window_start="2019-09-26",
        window_end="2019-09-29",
        lat=25.32,
        lon=82.97,
        notes="Varanasi belt late-season extreme rainfall.",
    ),
)


def list_flood_events(region_id: str = "uttar_pradesh") -> tuple[FloodEvent, ...]:
    if region_id == "belagavi":
        return (
            FloodEvent(
                id="bel_aug_2019",
                name="Belagavi monsoon — 2 Aug 2019",
                peak_date="2019-08-02",
                window_start="2019-07-28",
                window_end="2019-08-08",
                lat=15.30,
                lon=74.50,
                notes="Belagavi validation window.",
            ),
        )
    return FLOOD_EVENTS
