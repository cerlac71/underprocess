# Data layout — Uttar Pradesh (SIH26071)

All **raw training and validation data for the UP model is present** under `data/regions/uttar_pradesh/`.  
Only the old **Belagavi demo folders** at the repo root (`data/satellite/`, `data/nwp/`, etc.) were removed — they are not used by the UP pipeline.

## Verify before judging / submission

```bash
cd SIH2026
python scripts/verify_training_data.py
```

Writes a full inventory; use `--json data/regions/uttar_pradesh/data_verification.json` for a file to show judges.

## Raw data map (what the model was trained on)

| Category | Path | Source | Years / notes |
|----------|------|--------|----------------|
| **Satellite** | `regions/uttar_pradesh/satellite/chirps_uttar_pradesh_*.nc` | CHIRPS v2.0 daily 0.25° | 2012, 2013, 2014, 2016, 2018, 2019 (Jun–Sep) |
| **NWP** | `regions/uttar_pradesh/nwp/era5_uttar_pradesh_*.nc` | ERA5 via Open-Meteo | Same years, hourly → daily |
| **Radar (nowcast)** | `radar/imd_doppler_sample/` | IMD Doppler (Jaipur sample volumes) | Format proof; use `SIH_NOWCAST_IMD_SAMPLE=1` |
| **Radar (0–6 h nowcast)** | `regions/uttar_pradesh/radar/merra2_*.nc` (hourly) | NASA POWER / MERRA-2 | Optical flow on UP grid |
| **Radar proxy (daily fusion)** | `regions/uttar_pradesh/radar/merra2_uttar_pradesh_*.nc` | NASA POWER / MERRA-2 | Daily ML fusion until Lucknow Doppler arrives |
| **Observations** | `gridded_rainfall/RF25_ind*_rfp25.nc` | IMD 0.25° daily gridded | 2012–2019 (training target) |
| **Terrain** | `regions/uttar_pradesh/terrain/*.tif` | Open-Meteo elevation + derived slope/TWI/flow | State bbox |
| **Manifest** | `regions/uttar_pradesh/training_manifest.json` | Download log | Lists every file above |

**Training:** 2012–2018 monsoon seasons · **Holdout:** 2019  
**Model:** `models/regions/uttar_pradesh/multi_source_corrector.joblib`

## Re-download if anything is missing

```bash
export SIH_REGION=uttar_pradesh

# IMD Doppler sample (official NetCDF, ~85 MB raw) — nowcast integration
python scripts/download_imd_radar_sample.py

# All CHIRPS + ERA5 + MERRA-2 training years + IMD gaps
python scripts/download_training_years.py --region uttar_pradesh

# DEM / slope / TWI / flow accumulation
python scripts/download_region_terrain.py --region uttar_pradesh

# Retrain (optional)
PYTHONPATH=src python src/forecasting/train_multi_source.py

# Satellite-observed inundation reference (JRC GSW Landsat, required for real CSI/F1)
python scripts/download_flood_observations.py
PYTHONPATH=src SIH_REGION=uttar_pradesh python scripts/run_flood_validation.py
```

## What was deleted (and why)

| Removed path | Was | UP replacement |
|--------------|-----|------------------|
| `data/satellite/chirps_belagavi_*` | Karnataka demo | `data/regions/uttar_pradesh/satellite/` |
| `data/nwp/era5_belagavi_*` | Karnataka demo | `data/regions/uttar_pradesh/nwp/` |
| `data/radar/merra2_belagavi_*` | Karnataka demo | `data/regions/uttar_pradesh/radar/` |
| `data/terrain/dem_belagavi.tif` | Karnataka demo | `data/regions/uttar_pradesh/terrain/` |

Belagavi data is **not** required for the UP submission. To restore Belagavi for a separate demo:

```bash
python scripts/download_open_data.py   # set SIH_REGION=belagavi first
```

## Institutional data (future production)

- IMD Doppler weather radar (production UP) — sample Jaipur volumes prove integration; request Lucknow data from [radarapi.imd.gov.in](https://radarapi.imd.gov.in) / `radarlab@gmail.com`  
- NCMRWF operational NWP — replace ERA5 archive  
- GPM IMERG — optional higher-res satellite  

Do not fabricate these; document provenance when added.
