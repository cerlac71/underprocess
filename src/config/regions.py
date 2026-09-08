"""Geographic region definitions for SIH26071.

Set the active region with the SIH_REGION environment variable:

    export SIH_REGION=uttar_pradesh
    python scripts/download_training_years.py --region uttar_pradesh
    python scripts/train_models.py

Regions
-------
- ``belagavi``      — original demo domain (~55 km box, Karnataka)
- ``uttar_pradesh`` — full UP state at 0.25° (~840 grid cells)
- ``india``         — all-India IMD 0.25° grid (~14k cells; large downloads)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Region:
    id: str
    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    grid_step: float = 0.25
    holdout_year: int = 2019
    resolution_km: float = 0.25
    bbox_pad_deg: float = 0.5

    @property
    def bbox(self) -> dict[str, float]:
        return {
            "lat_min": self.lat_min,
            "lat_max": self.lat_max,
            "lon_min": self.lon_min,
            "lon_max": self.lon_max,
        }

    @property
    def target_grid(self) -> dict:
        return {
            "lat_min": self.lat_min,
            "lat_max": self.lat_max,
            "lon_min": self.lon_min,
            "lon_max": self.lon_max,
            "resolution_km": self.resolution_km,
            "grid_step_deg": self.grid_step,
        }

    def cell_area_m2(self, lat_center: float | None = None) -> float:
        """Approximate grid-cell area in m² for runoff/flood volume conversion."""
        import math

        lat = lat_center if lat_center is not None else (self.lat_min + self.lat_max) / 2.0
        lat_m = self.grid_step * 111_320.0
        lon_m = self.grid_step * 111_320.0 * math.cos(math.radians(lat))
        return lat_m * lon_m

    @property
    def approx_grid_cells(self) -> int:
        n_lat = int((self.lat_max - self.lat_min) / self.grid_step) + 1
        n_lon = int((self.lon_max - self.lon_min) / self.grid_step) + 1
        return n_lat * n_lon

    def data_root(self, project_root: Path | None = None) -> Path:
        root = project_root or PROJECT_ROOT
        return root / "data" / "regions" / self.id

    def imd_dir(self, project_root: Path | None = None) -> Path:
        """IMD all-India files are shared; stored once under data/gridded_rainfall."""
        root = project_root or PROJECT_ROOT
        return root / "data" / "gridded_rainfall"

    def satellite_dir(self, project_root: Path | None = None) -> Path:
        return self.data_root(project_root) / "satellite"

    def nwp_dir(self, project_root: Path | None = None) -> Path:
        return self.data_root(project_root) / "nwp"

    def radar_dir(self, project_root: Path | None = None) -> Path:
        return self.data_root(project_root) / "radar"

    def terrain_dir(self, project_root: Path | None = None) -> Path:
        return self.data_root(project_root) / "terrain"

    def model_dir(self, project_root: Path | None = None) -> Path:
        return (project_root or PROJECT_ROOT) / "models" / "regions" / self.id

    def _resolve(self, primary: Path, legacy: Path | None = None) -> Path:
        if primary.exists():
            return primary
        if legacy and legacy.exists():
            return legacy
        return primary

    def chirps_file(self, year: int, project_root: Path | None = None) -> Path:
        root = project_root or PROJECT_ROOT
        primary = self.satellite_dir(root) / f"chirps_{self.id}_{year}.nc"
        legacy = root / "data" / "satellite" / f"chirps_belagavi_{year}.nc"
        return self._resolve(primary, legacy if self.id == "belagavi" else None)

    def era5_file(self, year: int, project_root: Path | None = None) -> Path:
        root = project_root or PROJECT_ROOT
        primary = self.nwp_dir(root) / f"era5_{self.id}_{year}.nc"
        legacy = root / "data" / "nwp" / f"era5_belagavi_{year}.nc"
        return self._resolve(primary, legacy if self.id == "belagavi" else None)

    def merra2_file(self, year: int, project_root: Path | None = None) -> Path:
        root = project_root or PROJECT_ROOT
        primary = self.radar_dir(root) / f"merra2_{self.id}_{year}.nc"
        legacy = root / "data" / "radar" / f"merra2_belagavi_{year}.nc"
        return self._resolve(primary, legacy if self.id == "belagavi" else None)

    def dem_file(self, project_root: Path | None = None) -> Path:
        root = project_root or PROJECT_ROOT
        primary = self.terrain_dir(root) / f"dem_{self.id}.tif"
        legacy = root / "data" / "terrain" / "dem_belagavi.tif"
        return self._resolve(primary, legacy if self.id == "belagavi" else None)

    def slope_file(self, project_root: Path | None = None) -> Path:
        root = project_root or PROJECT_ROOT
        primary = self.terrain_dir(root) / f"slope_{self.id}.tif"
        legacy = root / "data" / "terrain" / "slope_belagavi.tif"
        return self._resolve(primary, legacy if self.id == "belagavi" else None)

    def twi_file(self, project_root: Path | None = None) -> Path:
        root = project_root or PROJECT_ROOT
        primary = self.terrain_dir(root) / f"twi_{self.id}.tif"
        legacy = root / "data" / "terrain" / "twi_belagavi.tif"
        return self._resolve(primary, legacy if self.id == "belagavi" else None)


REGIONS: dict[str, Region] = {
    "belagavi": Region(
        id="belagavi",
        name="Belagavi / Goa border, Karnataka",
        lat_min=14.625,
        lat_max=15.375,
        lon_min=74.625,
        lon_max=75.375,
        grid_step=0.25,
        resolution_km=0.1,
        bbox_pad_deg=1.0,
    ),
    "uttar_pradesh": Region(
        id="uttar_pradesh",
        name="Uttar Pradesh",
        lat_min=23.5,
        lat_max=30.5,
        lon_min=77.0,
        lon_max=84.5,
        grid_step=0.25,
        resolution_km=28.0,
        bbox_pad_deg=0.5,
    ),
    "india": Region(
        id="india",
        name="India (IMD 0.25° national grid)",
        lat_min=6.5,
        lat_max=37.0,
        lon_min=68.0,
        lon_max=97.5,
        grid_step=0.25,
        resolution_km=28.0,
        bbox_pad_deg=0.0,
    ),
}


def get_active_region(region_id: str | None = None) -> Region:
    rid = (region_id or os.environ.get("SIH_REGION", "uttar_pradesh")).lower().strip()
    if rid not in REGIONS:
        known = ", ".join(REGIONS)
        raise ValueError(f"Unknown region '{rid}'. Choose from: {known}")
    return REGIONS[rid]


def get_data_root(region_id: str | None = None, project_root: Path | None = None) -> Path:
    return get_active_region(region_id).data_root(project_root)


def list_regions() -> list[dict]:
    return [
        {
            "id": r.id,
            "name": r.name,
            "bbox": r.bbox,
            "grid_cells": r.approx_grid_cells,
        }
        for r in REGIONS.values()
    ]
