# Scaling to Uttar Pradesh or All-India

The project originally targeted **Belagavi** (~25 grid cells). It now supports
three regions via `src/config/regions.py`.

## Supported regions

| Region ID | Coverage | Grid cells (0.25°) | Download time (est.) |
|-----------|----------|-------------------|----------------------|
| `belagavi` | ~55×55 km, Karnataka | ~25 | Minutes |
| `uttar_pradesh` | Full UP state | ~840 | Hours |
| `india` | All-India IMD grid | ~14,000 | Days |

## Quick start — Uttar Pradesh

```bash
# 1. Set region (or pass --region on each script)
export SIH_REGION=uttar_pradesh

# 2. Download training data (2010–2019 monsoon seasons)
python scripts/download_training_years.py --region uttar_pradesh

# 3. Download terrain / landcover for UP bbox
python scripts/download_open_data.py --region uttar_pradesh

# 4. Train models on UP data
python scripts/train_models.py

# 5. Run pipeline
python -m src.pipeline.pipeline
```

## What changes per region

- **Bounding box** — lat/lon limits for clipping all sources
- **Data paths** — `data/regions/{region_id}/satellite|nwp|radar|terrain/`
- **IMD** — shared all-India files in `data/gridded_rainfall/` (subset on load)
- **Models** — saved to `models/regions/{region_id}/` when using regional training

## Realistic expectations

### Uttar Pradesh (recommended next step)
- Feasible on a laptop with patience (~2–6 hours download, ~30 min training)
- Models learn UP-specific monsoon bias patterns
- Flood terrain needs SRTM tiles for UP (auto-downloaded by `download_open_data.py`)
- **Nowcast:** 0–6 h optical flow on hourly MERRA-2 frames over the full UP grid (daily aggregate still used for fusion/training)

### All-India
- **Rainfall bias correction**: feasible at IMD 0.25° resolution
- **ConvLSTM hourly nowcast**: impractical at full India resolution without GPU cluster
- **0.1 km flood routing**: not feasible nationally — use district-level tiles or 1 km DEM
- Recommended approach: train **state-wise models** (UP, Bihar, Assam, etc.) not one national model

## SIH presentation tip

> "Our system is architected for any Indian state. The demo runs on Belagavi;
> production deployment targets Uttar Pradesh with state-specific trained models
> on IMD + CHIRPS + ERA5, extensible to all-India at 0.25°."
