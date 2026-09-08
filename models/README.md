# Trained models — Uttar Pradesh (SIH26071)

## Active model (production path)

```
models/regions/uttar_pradesh/
├── multi_source_corrector.joblib       # Per-source + ensemble ML correctors
├── multi_source_corrector_metrics.json # Holdout metrics (2019)
└── training_summary.json
```

Train / retrain:

```bash
export SIH_REGION=uttar_pradesh
python scripts/train_models.py
# or
PYTHONPATH=src python src/forecasting/train_multi_source.py
```

## Raw data used (must be present for judges)

All under `data/regions/uttar_pradesh/` + shared `data/gridded_rainfall/` (IMD).  
Verify with:

```bash
python scripts/verify_training_data.py
```

## UP training setup

| Item | Value |
|------|-------|
| Training years | 2012, 2013, 2014, 2016, 2018 (monsoon Jun–Sep) |
| Holdout | 2019 monsoon |
| Training samples | 444,690 grid-day pairs |
| Holdout samples | 88,938 |
| Ensemble MAE | ~6.2 mm/day |
| Ensemble R² | ~0.39 |
| Wet-day R² (≥2 mm) | ~0.20 |

## Sources enabled after holdout validation

| Source | Corrected MAE | Enabled |
|--------|---------------|---------|
| CHIRPS | ~8.1 mm | ✓ |
| ERA5 | ~8.0 mm | ✓ |
| MERRA-2 | ~7.2 mm | ✓ |

Legacy Belagavi models under `models/` root were removed; UP artefacts live only under `models/regions/uttar_pradesh/`.
