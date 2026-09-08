# SIH26071 — AI/ML-Based Integrated Heavy Rainfall Early Warning & Inundation Prediction System

**Problem Statement:** [SIH26071](https://sih2026.vuce.in/ps/SIH26071)  
**Organization:** Ministry of Earth Sciences (MoES) · India Meteorological Department (IMD)  
**Theme:** Disaster Management · Software  
**Deadline:** 30 September 2026

An end-to-end system that integrates **satellite**, **radar**, **observational weather**, and **numerical weather prediction (NWP)** data through AI/ML bias correction and multi-source fusion to deliver **heavy rainfall early warnings** and **inundation (flood) predictions** with actionable alerts.

---

## Problem alignment (SIH26071 requirements)

| Requirement | Implementation |
|-------------|----------------|
| Satellite data | CHIRPS daily rainfall (`data/regions/*/satellite/`) |
| Radar data | IMD Doppler sample (Jaipur) + MERRA-2 hourly proxy for optical-flow nowcast |
| Observational weather | IMD 0.25° gridded rainfall + GHCN station QC |
| NWP model data | ERA5 precipitation + MERRA-2 |
| AI/ML integration | Multi-source bias correctors (per-source + ensemble), ConvLSTM nowcast (Belagavi) |
| Heavy rainfall early warning | IMD colour-coded thresholds (Green/Yellow/Orange/Red) with terrain adjustment |
| Inundation prediction | SCS-CN runoff → terrain-based flood depth + risk classification |
| Lead-time tagging | Immediate / short / medium / extended buckets with human-readable messages |
| Alert delivery | Console, desktop, email (SMTP), SMS (Twilio), webhook + district subscriptions |

---

## Architecture

```
┌─────────────┐  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐
│  Satellite  │  │    Radar    │  │ Observations │  │     NWP     │
│   (CHIRPS)  │  │ IMD/MERRA-2 │  │  IMD/GHCN    │  │ ERA5/MERRA2 │
└──────┬──────┘  └──────┬──────┘  └──────┬───────┘  └──────┬──────┘
       │                │                │                  │
       └────────────────┴────────────────┴──────────────────┘
                                    │
                          Harmonisation (0.25° grid)
                                    │
                    ML Bias Correction (trained vs IMD)
                                    │
                    Multi-source Inverse-Variance Fusion
                                    │
              ┌─────────────────────┴─────────────────────┐
              │                                         │
     Optical-flow Nowcast (0–6 h)              SCS-CN Runoff
              │                                         │
              └─────────────────────┬─────────────────────┘
                                    │
                         Inundation Model (DEM/TWI)
                                    │
                    IMD Alert Thresholds + Lead Time
                                    │
              ┌─────────────────────┴─────────────────────┐
              │                                         │
        Streamlit Dashboard                    Alert Monitor
     (maps, advisories, search)            (email/SMS/desktop)
```

---

## Quick start

### 1. Install dependencies

```bash
cd SIH/SIH2026
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download data (Uttar Pradesh — default region)

```bash
export SIH_REGION=uttar_pradesh
python scripts/download_training_years.py --region uttar_pradesh
python scripts/download_region_terrain.py --region uttar_pradesh
python scripts/download_open_data.py
python scripts/download_imd_radar_sample.py   # optional IMD Doppler demo
```

### 3. Train ML models

```bash
PYTHONPATH=src python scripts/train_models.py --region uttar_pradesh
```

### 4. Run the pipeline

```bash
PYTHONPATH=src SIH_REGION=uttar_pradesh python -m pipeline.pipeline
```

Outputs:
- `pipeline_results.nc` — fused rainfall, runoff, flood depth, risk, confidence
- `pipeline_output.png` — quick-look maps
- `alerts/district_alerts.json` — per-district alert summary

### 5. Launch the dashboard

```bash
PYTHONPATH=src SIH_REGION=uttar_pradesh streamlit run app.py
```

Pages:
- **Home** — district search, map, public advisory
- **Rainfall Nowcast** — 0–6 h optical-flow forecast
- **Flood Risk Map** — inundation overlays
- **Alerts Dashboard** — all-district warnings + subscriptions
- **Historical Backtest** — CSI/F1 validation on past flood events

### 6. Get automatic alerts (flood / heavy rain)

```bash
# One-shot: run pipeline + notify
PYTHONPATH=src python scripts/run_alert_monitor.py --channels console,desktop

# With email (configure .env first — see .env.example)
cp .env.example .env   # edit with your SMTP/Twilio credentials
export $(grep -v '^#' .env | xargs)
PYTHONPATH=src python scripts/run_alert_monitor.py --channels console,desktop,email

# Cron (every 30 min during monsoon)
*/30 * * * * cd /path/to/SIH2026 && PYTHONPATH=src python scripts/run_alert_monitor.py --channels console,desktop,email
```

**Alert triggers:**
- IMD heavy-rain colour: Yellow (64–115 mm/24h), Orange (116–204 mm), Red (>204 mm)
- Inundation risk class ≥ 1 or depth ≥ 0.3 m escalates the alert
- Desktop notification on macOS/Linux when active alerts exist

Subscribe per district in the **Alerts Dashboard** (email / SMS / webhook).

---

## Project structure

```
SIH2026/
├── app.py                    # Main Streamlit dashboard
├── pages/                    # Multi-page app (nowcast, map, alerts, backtest)
├── src/
│   ├── pipeline/             # End-to-end orchestration
│   ├── integerations/        # Satellite, radar, NWP, observational ingest
│   ├── harmonisation/        # Multi-source grid alignment
│   ├── forecasting/          # ML bias correction + ConvLSTM training
│   ├── fusion/               # Inverse-variance fusion + confidence
│   ├── hydrology/            # SCS-CN runoff
│   ├── flood/                # Inundation depth + risk
│   ├── nowcasting/           # Optical-flow + ConvLSTM 0–6 h
│   ├── alerts/               # Thresholds, lead time, notifications
│   ├── validation/           # Historical flood event backtest
│   └── ui/                   # Maps, styles, location data (75 UP districts)
├── scripts/
│   ├── train_models.py
│   ├── run_alert_monitor.py  # Automated alert delivery
│   ├── run_flood_validation.py
│   └── download_*.py
├── data/regions/uttar_pradesh/
├── models/regions/uttar_pradesh/
└── alerts/
    ├── district_alerts.json
    ├── subscriptions.json
    └── notification_log.json
```

---

## Regions

Set `SIH_REGION` environment variable:

| Region | Coverage | Grid |
|--------|----------|------|
| `uttar_pradesh` | Full UP state (75 districts) | 0.25° (~840 cells) |
| `belagavi` | Karnataka demo domain | 0.25° (~25 cells) |
| `india` | National IMD grid | 0.25° (~14k cells) |

---

## Model performance (Uttar Pradesh holdout 2019)

| Metric | Value |
|--------|-------|
| Ensemble MAE | ~6.2 mm/day |
| Ensemble R² | ~0.39 |
| Wet-day R² (≥2 mm) | ~0.20 |
| Sources enabled | CHIRPS, ERA5, MERRA-2 |

Flood validation metrics: `models/regions/uttar_pradesh/flood_validation_metrics.json`

---

## Data sources

| Source | Product | Role |
|--------|---------|------|
| IMD | 0.25° daily gridded rainfall | Ground truth + observations |
| CHIRPS | Satellite rainfall | Bias-corrected estimate |
| ERA5 | NWP precipitation | Bias-corrected forecast proxy |
| MERRA-2 | Radar proxy / hourly frames | Nowcast input |
| IMD Doppler | Jaipur sample volumes | Real radar format demo |
| SRTM/Copernicus | DEM, slope, TWI | Inundation terrain |
| GHCN | Station precipitation | Observational QC |

---

## Environment variables

See `.env.example` for full list. Key variables:

- `SIH_REGION` — active region (`uttar_pradesh` default)
- `SIH_SMTP_*` — email alert delivery
- `SIH_TWILIO_*` — SMS via Twilio
- `SIH_ALERT_WEBHOOK_URL` — Slack/Discord/custom webhook
- `SIH_DESKTOP_ALERTS` — macOS/Linux desktop notifications (default on)

---

## Team / demo tips for judges

1. Run `streamlit run app.py` and search a district (e.g. Varanasi, Lucknow).
2. Click **Update forecast** to run the live pipeline on real IMD data.
3. Open **Alerts Dashboard** to see state-wide warnings and subscribe.
4. Run `python scripts/run_alert_monitor.py --channels console,desktop` to demonstrate push alerts.
5. Open **Historical Backtest** for CSI/F1 inundation validation.

---

## License

Built for Smart India Hackathon 2026 — SIH26071.
