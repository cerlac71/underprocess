#!/usr/bin/env bash
# One-command launcher for SIH26071 dashboard + optional alert monitor
set -euo pipefail
cd "$(dirname "$0")/.."
export SIH_REGION="${SIH_REGION:-uttar_pradesh}"
export PYTHONPATH=src

if [[ ! -f pipeline_results.nc ]]; then
  echo "Running initial pipeline…"
  python -m pipeline.pipeline
fi

if [[ "${1:-}" == "--with-alerts" ]]; then
  python scripts/run_alert_monitor.py --skip-pipeline --channels console,desktop &
fi

echo "Starting dashboard at http://localhost:8501"
streamlit run app.py
