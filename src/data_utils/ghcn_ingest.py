"""Read verified daily precipitation observations from NOAA GHCN-Daily files.

The repository stores the original fixed-width ``.dly`` downloads under
``data/stations/ghcn_raw``.  This reader filters them to a requested period and
returns the standard station-data frame used by the pipeline.  GHCN stores
precipitation in tenths of a millimetre; quality-flagged and missing values are
excluded rather than imputed.
"""

from __future__ import annotations

import calendar
from pathlib import Path

import pandas as pd


STATIONS = {
    "IN009020400": (15.63, 74.52, "Khanpur"),
    "IN009021000": (15.85, 74.617, "Belgaum/Sambra"),
    "IN009021100": (15.85, 74.533, "Belgaum"),
    "IN009090101": (15.18, 74.97, "Kalghatgi"),
    "IN009090200": (15.33, 75.13, "Hubli"),
}


def load_ghcn_daily_precipitation(data_dir, start_date: str, end_date: str) -> pd.DataFrame:
    """Return good-quality daily GHCN PRCP observations for the local stations."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    rows = []
    for path in sorted(Path(data_dir).glob("*.dly")):
        station_id = path.stem
        if station_id not in STATIONS:
            continue
        lat, lon, name = STATIONS[station_id]
        for line in path.read_text(encoding="ascii").splitlines():
            if line[17:21] != "PRCP":
                continue
            year, month = int(line[11:15]), int(line[15:17])
            for day in range(1, calendar.monthrange(year, month)[1] + 1):
                offset = 21 + (day - 1) * 8
                value = int(line[offset:offset + 5])
                measurement_flag = line[offset + 5:offset + 6].strip()
                quality_flag = line[offset + 6:offset + 7].strip()
                timestamp = pd.Timestamp(year=year, month=month, day=day)
                if timestamp < start or timestamp > end or value == -9999 or quality_flag:
                    continue
                rows.append({
                    "timestamp": timestamp,
                    "station_id": station_id,
                    "station_name": name,
                    "lat": lat,
                    "lon": lon,
                    "rainfall_mm": value / 10.0,
                    "measurement_flag": measurement_flag or None,
                    "source": "NOAA GHCN-Daily",
                })
    columns = ["timestamp", "station_id", "station_name", "lat", "lon",
               "rainfall_mm", "measurement_flag", "source"]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["timestamp", "station_id"]
    ).reset_index(drop=True)
