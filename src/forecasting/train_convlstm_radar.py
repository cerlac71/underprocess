"""Train a lightweight ConvLSTM nowcaster on hourly MERRA-2 radar-proxy data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC_ROOT))
from config.bootstrap import bootstrap_project

bootstrap_project()

import torch
import xarray as xr

from data_utils.imd_ingest import discover_training_years
from nowcasting.nowcasting import EncoderDecoderConvLSTM, RainfallSequenceDataset, train_convlstm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "convlstm_merra2.pt"
METRICS_PATH = PROJECT_ROOT / "models" / "convlstm_merra2_metrics.json"


def _load_merra2_stack(years: list[int]) -> xr.Dataset:
    """Concatenate yearly MERRA-2 hourly files into one dataset."""
    paths = [PROJECT_ROOT / "data" / "radar" / f"merra2_belagavi_{y}.nc" for y in years]
    existing = [p for p in paths if p.exists()]
    if not existing:
        fallback = PROJECT_ROOT / "data" / "radar" / "merra2_power_belagavi_20190720_20190808.nc"
        if fallback.exists():
            with xr.open_dataset(fallback) as raw:
                return raw.load()
        raise FileNotFoundError("No MERRA-2 radar proxy data found.")
    arrays = []
    for path in existing:
        with xr.open_dataset(path) as raw:
            arrays.append(raw["rainfall"].load())
    rainfall = xr.concat(arrays, dim="time").sortby("time")
    return xr.Dataset({"rainfall": rainfall})


def train(epochs: int = 25, batch_size: int = 8) -> dict:
    years = discover_training_years(PROJECT_ROOT / "data" / "gridded_rainfall")
    if not years:
        years = list(range(2012, 2020))

    ds = _load_merra2_stack(years)
    if ds.sizes["time"] < 24:
        raise ValueError(f"Need at least 24 hourly steps; got {ds.sizes['time']}")

    model = train_convlstm(
        ds,
        input_len=6,
        output_len=6,
        hidden_dims=[32, 32],
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=1e-3,
    )

    MODEL_PATH.parent.mkdir(exist_ok=True)
    dataset = RainfallSequenceDataset(ds, input_len=6, output_len=6)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_len": 6,
            "output_len": 6,
            "hidden_dims": [32, 32],
            "mean": dataset.mean,
            "std": dataset.std,
            "training_years": years,
            "source": "NASA POWER MERRA-2 hourly (radar proxy)",
        },
        MODEL_PATH,
    )
    metrics = {
        "epochs": epochs,
        "training_sequences": len(dataset),
        "training_years": years,
        "grid_shape": list(ds["rainfall"].shape[1:]),
        "time_steps": int(ds.sizes["time"]),
        "model_path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        "caveat": f"Trained on {len(years)} monsoon seasons of MERRA-2 hourly proxy.",
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
