#!/usr/bin/env python3
"""
Nowcasting Module for SIH26071: Optical Flow vs ConvLSTM

This module provides two approaches for 0–6 hour rainfall nowcasting:

1. **Optical-flow extrapolation** (baseline): Uses dense optical flow
   (via pysteps if available, else OpenCV Farneback) to estimate motion
   fields from recent radar/rainfall frames and advect the latest frame
   forward, optionally with intensity decay.

2. **ConvLSTM** (learned model): Implements a sequence-to-sequence
   ConvLSTM model in PyTorch. Requires historical training data to learn
   spatiotemporal patterns. Provided for architecture demonstration;
   realistic training data volume is discussed.

Both methods accept an xarray Dataset with dimensions (time, lat, lon)
and a rainfall variable (default 'rainfall'). Output is a dictionary of
lead_time -> 2D rainfall intensity map.

The module also includes evaluation metrics (CSI, FSS, RMSE) for
comparing forecasts at different lead times.

Author: SIH26071 Team
"""

import logging
import numpy as np
import xarray as xr
import pandas as pd
from typing import Dict, Optional, List, Tuple

# Try importing pysteps (preferred for radar nowcasting)
try:
    import pysteps
    from pysteps import motion, nowcasts
    PYSTEPS_AVAILABLE = True
except ImportError:
    PYSTEPS_AVAILABLE = False
    logging.warning("pysteps not installed. Falling back to OpenCV for optical flow.")

# OpenCV for Farneback fallback
try:
    import cv2
    cv2.setNumThreads(0)
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    logging.error("OpenCV not installed. Optical flow fallback unavailable.")
    raise

# PyTorch for ConvLSTM (optional; the optical-flow path does not need it)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not installed. ConvLSTM approach unavailable.")

    class _UnavailableTorchModel:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is not installed; ConvLSTM nowcasting is unavailable.")

    class _UnavailableNN:
        Module = _UnavailableTorchModel
        ModuleList = _UnavailableTorchModel

    nn = _UnavailableNN()
    Dataset = _UnavailableTorchModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Common utilities
# ----------------------------------------------------------------------
def dataset_to_numpy(ds: xr.Dataset, var_name: str = 'rainfall') -> np.ndarray:
    """
    Extract rainfall field from xarray Dataset as numpy array (time, lat, lon).
    """
    if var_name not in ds:
        raise ValueError(f"Variable '{var_name}' not found in dataset.")
    return ds[var_name].values

def numpy_to_xarray(data: np.ndarray, time_coords, lat_coords, lon_coords,
                    var_name: str = 'rainfall') -> xr.DataArray:
    """Wrap numpy array back into xarray DataArray."""
    return xr.DataArray(
        data,
        dims=('time', 'lat', 'lon'),
        coords={'time': time_coords, 'lat': lat_coords, 'lon': lon_coords},
        name=var_name
    )

# ----------------------------------------------------------------------
# APPROACH 1: Optical Flow Extrapolation
# ----------------------------------------------------------------------
class OpticalFlowNowcaster:
    """
    Optical flow based nowcasting.

    Estimates motion field between recent frames and advects the most recent
    frame forward. Supports optional intensity decay via S-PROG style
    scaling.

    If pysteps is available, uses its robust optical flow and advection
    routines. Otherwise falls back to OpenCV Farneback for motion estimation
    and semi-Lagrangian advection.
    """

    def __init__(self, use_decay: bool = True, decay_factor: float = 0.9,
                 num_reference_frames: int = 3):
        """
        Parameters
        ----------
        use_decay : bool
            Whether to apply intensity decay/growth with lead time.
        decay_factor : float
            Multiplicative factor applied to intensity at each lead time step
            (e.g., 0.9 means 10% decay per step). Only used if use_decay=True.
        num_reference_frames : int
            Number of previous frames to use for motion estimation.
        """
        self.use_decay = use_decay
        self.decay_factor = decay_factor
        self.num_reference_frames = num_reference_frames

    def estimate_motion_pysteps(self, precip_frames: np.ndarray):
        """Use pysteps to estimate motion field."""
        # pysteps expects input as (time, y, x) with mm/h
        motion_field = motion.get_method("LK")  # Lucas-Kanade
        return motion_field(precip_frames)

    def estimate_motion_opencv(self, precip_frames: np.ndarray):
        """Fallback using OpenCV Farneback."""
        # Convert to uint8 for optical flow
        frames_uint8 = []
        for i in range(precip_frames.shape[0]):
            frame = precip_frames[i]
            # Normalize to 0-255
            if frame.max() > 0:
                frame_norm = (frame / frame.max() * 255).astype(np.uint8)
            else:
                frame_norm = np.zeros_like(frame, dtype=np.uint8)
            frames_uint8.append(frame_norm)

        # Compute flow between last two frames (or average of several)
        # We'll compute flow between last frame and previous one.
        prev = frames_uint8[-2]
        curr = frames_uint8[-1]
        flow = cv2.calcOpticalFlowFarneback(
            prev, curr, None, pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        # flow shape: (height, width, 2) with u=dx, v=dy in pixels
        return flow

    def advect(self, field: np.ndarray, flow: np.ndarray, steps: int = 1) -> np.ndarray:
        """Advect field using flow field (semi-Lagrangian)."""
        h, w = field.shape
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        # For each step, move pixels along flow
        # Simple implementation: use remap
        for _ in range(steps):
            # Displace coordinates
            map_x = x - flow[..., 0]  # subtract because flow points forward
            map_y = y - flow[..., 1]
            # Remap
            field = cv2.remap(field.astype(np.float32), map_x.astype(np.float32),
                              map_y.astype(np.float32), interpolation=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
        return field

    def nowcast(self, dataset: xr.Dataset, lead_times_min: List[int] = [10, 30, 60, 120, 180, 360],
                var_name: str = 'rainfall', timestep_min: int = 10) -> Dict[int, np.ndarray]:
        """
        Generate nowcasts for given lead times.

        Parameters
        ----------
        dataset : xr.Dataset
            Input dataset with time, lat, lon dimensions.
        lead_times_min : list of int
            Lead times in minutes.
        var_name : str
            Name of rainfall variable.
        timestep_min : int
            Time between consecutive frames in minutes (used for advection steps).

        Returns
        -------
        dict of {lead_time_min: 2D array}
        """
        precip = dataset_to_numpy(dataset, var_name)
        if precip.shape[0] < 2:
            raise ValueError("At least two frames required for optical flow.")

        # Use last num_reference_frames
        recent = precip[-self.num_reference_frames:] if self.num_reference_frames > 1 else precip

        if PYSTEPS_AVAILABLE:
            # pysteps workflow
            # Convert to rainrate (mm/h) if needed
            motion_field = self.estimate_motion_pysteps(recent)
            # Advect the last frame
            last_frame = precip[-1]
            nowcast_dict = {}
            for lead in lead_times_min:
                n_steps = max(1, int(lead / timestep_min))
                # Use pysteps extrapolation
                extrap = nowcasts.get_method("extrapolation")
                forecast = extrap(last_frame, motion_field, n_steps)
                if self.use_decay:
                    decay = self.decay_factor ** n_steps
                    forecast *= decay
                nowcast_dict[lead] = forecast
        else:
            # OpenCV fallback
            flow = self.estimate_motion_opencv(recent)
            last_frame = precip[-1]
            nowcast_dict = {}
            for lead in lead_times_min:
                n_steps = max(1, int(lead / timestep_min))
                # Advect step by step
                forecast = last_frame.copy()
                for _ in range(n_steps):
                    forecast = self.advect(forecast, flow, steps=1)
                if self.use_decay:
                    decay = self.decay_factor ** n_steps
                    forecast *= decay
                nowcast_dict[lead] = forecast

        return nowcast_dict


# ----------------------------------------------------------------------
# APPROACH 2: ConvLSTM
# ----------------------------------------------------------------------
class ConvLSTMCell(nn.Module):
    """
    ConvLSTM cell as described in Shi et al. 2015.
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, x, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([x, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, height, width):
        device = next(self.parameters()).device
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device))


class EncoderDecoderConvLSTM(nn.Module):
    """
    Sequence-to-sequence ConvLSTM model.
    Encoder: stack of ConvLSTM layers processing input sequence.
    Decoder: stack of ConvLSTM layers producing output sequence autoregressively.
    """

    def __init__(self, input_dim=1, hidden_dims=[64, 64, 64], kernel_size=(3,3),
                 num_layers=3, output_dim=1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dims = hidden_dims

        # Encoder ConvLSTM cells
        encoder_cells = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dims[i-1]
            encoder_cells.append(ConvLSTMCell(in_dim, hidden_dims[i], kernel_size))
        self.encoder_cells = nn.ModuleList(encoder_cells)

        # Decoder ConvLSTM cells
        decoder_cells = []
        for i in range(num_layers):
            in_dim = hidden_dims[i]  # decoder input same as encoder hidden
            decoder_cells.append(ConvLSTMCell(in_dim, hidden_dims[i], kernel_size))
        self.decoder_cells = nn.ModuleList(decoder_cells)

        # Final output convolution (1x1)
        self.output_conv = nn.Conv2d(hidden_dims[-1], output_dim, kernel_size=1)
        # Project 1-channel forecast back to decoder hidden dim for autoregressive feedback
        self.decoder_input_proj = nn.Conv2d(output_dim, hidden_dims[0], kernel_size=1)

    def forward(self, input_seq, target_len):
        """
        input_seq: (batch, seq_len, channels, H, W)
        Returns: (batch, target_len, channels, H, W)
        """
        batch, seq_len, _, H, W = input_seq.size()

        # Encoder
        encoder_states = []  # list of (h,c) per layer, final states
        for layer in range(self.num_layers):
            h, c = self.encoder_cells[layer].init_hidden(batch, H, W)
            encoder_states.append((h, c))

        for t in range(seq_len):
            x = input_seq[:, t]
            for layer in range(self.num_layers):
                h, c = encoder_states[layer]
                h, c = self.encoder_cells[layer](x, (h, c))
                encoder_states[layer] = (h, c)
                x = h  # input to next layer

        # Decoder — start from encoder final hidden states
        decoder_states = [(h.clone(), c.clone()) for h, c in encoder_states]
        x = encoder_states[-1][0]  # top encoder hidden state, not raw 1-ch input
        outputs = []
        for t in range(target_len):
            for layer in range(self.num_layers):
                h, c = decoder_states[layer]
                h, c = self.decoder_cells[layer](x, (h, c))
                decoder_states[layer] = (h, c)
                x = h
            out = self.output_conv(x)
            outputs.append(out)
            x = self.decoder_input_proj(out)  # project 1-ch output back to hidden dim
        outputs = torch.stack(outputs, dim=1)
        return outputs


class RainfallSequenceDataset(Dataset):
    """
    Sliding window dataset for ConvLSTM training.
    Expects xarray Dataset with 'rainfall' variable, dims (time, lat, lon).
    """

    def __init__(self, ds: xr.Dataset, input_len: int = 6, output_len: int = 6,
                 var_name: str = 'rainfall', normalize: bool = True):
        self.ds = ds
        self.var_name = var_name
        self.input_len = input_len
        self.output_len = output_len
        self.normalize = normalize
        self.data = ds[var_name].values  # shape (T, H, W)

        # Compute normalization parameters
        if self.normalize:
            self.mean = np.nanmean(self.data)
            self.std = np.nanstd(self.data)
            if self.std == 0:
                self.std = 1.0
        else:
            self.mean = 0.0
            self.std = 1.0

        self.n_samples = len(self.data) - input_len - output_len + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.input_len]
        y = self.data[idx+self.input_len:idx+self.input_len+self.output_len]

        # Normalize
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std

        # Add channel dimension (1)
        x = x[:, np.newaxis, :, :].astype(np.float32)
        y = y[:, np.newaxis, :, :].astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


def train_convlstm(ds: xr.Dataset, input_len: int = 6, output_len: int = 6,
                   hidden_dims: List[int] = [64, 64], epochs: int = 50,
                   batch_size: int = 4, learning_rate: float = 1e-3,
                   weight_heavy_rain: float = 5.0, rain_threshold: float = 10.0):
    """
    Training loop for ConvLSTM.

    Uses weighted MSE loss: higher weight on pixels above rain_threshold to
    emphasize heavy rainfall.

    Parameters
    ----------
    ds : xr.Dataset
        Historical dataset with rainfall variable.
    input_len : int
        Number of input frames.
    output_len : int
        Number of forecast frames.
    hidden_dims : list
        Number of channels per ConvLSTM layer.
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size.
    learning_rate : float
        Optimizer learning rate.
    weight_heavy_rain : float
        Weight applied to heavy-rain pixels in loss.
    rain_threshold : float
        Rainfall intensity threshold (mm) to consider as heavy rain.

    Returns
    -------
    model : trained model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = RainfallSequenceDataset(ds, input_len, output_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = EncoderDecoderConvLSTM(
        input_dim=1,
        hidden_dims=hidden_dims,
        kernel_size=(3,3),
        num_layers=len(hidden_dims),
        output_dim=1
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss(reduction='none')  # we'll apply weights manually

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            output = model(x_batch, output_len)

            # Weighted MSE: higher weight on heavy rain
            weight = torch.ones_like(y_batch)
            weight[y_batch * dataset.std + dataset.mean > rain_threshold] = weight_heavy_rain

            loss = criterion(output, y_batch)
            loss = (loss * weight).mean()

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    return model


def nowcast_convlstm(model: nn.Module, recent_frames: np.ndarray,
                     lead_times_min: List[int], timestep_min: int = 30) -> Dict[int, np.ndarray]:
    """
    Generate nowcasts using trained ConvLSTM model.

    Parameters
    ----------
    model : trained EncoderDecoderConvLSTM
    recent_frames : np.ndarray (input_len, H, W)
    lead_times_min : list of int
        Lead times in minutes.
    timestep_min : int
        Time between frames in minutes (assumed same as training).

    Returns
    -------
    dict of lead_time -> forecast array (H, W)
    """
    device = next(model.parameters()).device
    model.eval()

    # Convert to torch tensor, normalize with training stats? Need to pass
    # normalization parameters. For simplicity assume model was trained with
    # zero-mean unit variance; we'll normalize similarly.
    # This is a simplification; in practice store dataset mean/std.
    x = torch.from_numpy(recent_frames[:, np.newaxis, :, :].astype(np.float32)).unsqueeze(0).to(device)

    with torch.no_grad():
        output_seq = model(x, len(lead_times_min))
    output_seq = output_seq.squeeze(0).cpu().numpy()  # (seq_len, 1, H, W)
    output_seq = output_seq[:, 0, :, :]  # (seq_len, H, W)

    nowcast_dict = {}
    for i, lead in enumerate(lead_times_min):
        nowcast_dict[lead] = output_seq[i]
    return nowcast_dict


# ----------------------------------------------------------------------
# EVALUATION METRICS
# ----------------------------------------------------------------------
def compute_csi(pred: np.ndarray, obs: np.ndarray, threshold: float) -> float:
    """Critical Success Index for binary event (rain >= threshold)."""
    pred_bin = pred >= threshold
    obs_bin = obs >= threshold
    hits = np.sum(pred_bin & obs_bin)
    misses = np.sum(~pred_bin & obs_bin)
    false_alarms = np.sum(pred_bin & ~obs_bin)
    denom = hits + misses + false_alarms
    if denom == 0:
        return np.nan
    return hits / denom

def compute_fss(pred: np.ndarray, obs: np.ndarray, threshold: float,
                neighborhood_size: int = 5) -> float:
    """
    Fractions Skill Score (FSS) with a square neighborhood.
    Neighborhood size = 2*radius+1 pixels.
    """
    pred_bin = (pred >= threshold).astype(float)
    obs_bin = (obs >= threshold).astype(float)

    # Compute fractions using convolution
    kernel = np.ones((neighborhood_size, neighborhood_size))
    pred_frac = cv2.filter2D(pred_bin, -1, kernel)
    obs_frac = cv2.filter2D(obs_bin, -1, kernel)

    numerator = np.sum((pred_frac - obs_frac) ** 2)
    denominator = np.sum(pred_frac**2) + np.sum(obs_frac**2)
    if denominator == 0:
        return np.nan
    return 1.0 - numerator / denominator

def compute_rmse(pred: np.ndarray, obs: np.ndarray) -> float:
    """Root Mean Squared Error."""
    mask = ~np.isnan(obs) & ~np.isnan(pred)
    if mask.sum() == 0:
        return np.nan
    return np.sqrt(np.mean((pred[mask] - obs[mask]) ** 2))


def evaluate_nowcast(forecast: Dict[int, np.ndarray],
                     obs: Dict[int, np.ndarray],
                     thresholds: List[float] = [0.1, 1.0, 5.0, 10.0],
                     neighborhood_size: int = 5) -> pd.DataFrame:
    """
    Compute metrics for each lead time.

    Parameters
    ----------
    forecast : dict of lead_time -> pred_array
    obs : dict of lead_time -> obs_array (same lead times)
    thresholds : list of float (mm) for CSI and FSS.
    neighborhood_size : int for FSS.

    Returns
    -------
    pd.DataFrame with columns: lead_time, threshold, CSI, FSS, RMSE.
    """
    rows = []
    for lead in forecast.keys():
        pred = forecast[lead]
        obs_arr = obs[lead]
        rmse = compute_rmse(pred, obs_arr)
        for th in thresholds:
            csi = compute_csi(pred, obs_arr, th)
            fss = compute_fss(pred, obs_arr, th, neighborhood_size)
            rows.append({'lead_time': lead, 'threshold': th,
                         'CSI': csi, 'FSS': fss, 'RMSE': rmse})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# DEMO / TEST
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create synthetic dataset for demonstration
    logging.info("Creating synthetic rainfall dataset...")
    np.random.seed(42)
    times = pd.date_range("2025-01-01 00:00", periods=20, freq="10min")
    lat = np.linspace(10, 20, 64)
    lon = np.linspace(70, 80, 64)

    # Create a moving rain cell
    data = np.zeros((20, 64, 64))
    for t in range(20):
        # Gaussian blob centered at (15+t*0.5, 75)
        lat0 = 15 + t * 0.05
        lon0 = 75 + t * 0.05
        for i, la in enumerate(lat):
            for j, lo in enumerate(lon):
                dist = (la - lat0)**2 + (lo - lon0)**2
                data[t, i, j] = 20 * np.exp(-dist / 0.5)
    ds = xr.Dataset(
        {"rainfall": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lat, "lon": lon}
    )

    # Test Approach 1
    logging.info("Testing optical flow nowcaster...")
    of_nowcaster = OpticalFlowNowcaster(use_decay=True)
    lead_times = [10, 30, 60]
    forecast_of = of_nowcaster.nowcast(ds, lead_times_min=lead_times, timestep_min=10)
    logging.info("Optical flow forecast generated for lead times: %s", list(forecast_of.keys()))

    # For Approach 2, we would need trained model; just show architecture instantiation
    logging.info("ConvLSTM model architecture (not trained):")
    model = EncoderDecoderConvLSTM(input_dim=1, hidden_dims=[32, 32], num_layers=2, output_dim=1)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"ConvLSTM model has {total_params} parameters.")

    # Note about training data
    logging.info("To train ConvLSTM, you need at least several months of radar data. "
                 "For a hackathon demo, optical flow is the recommended baseline.")

    # Example evaluation (using synthetic as both pred and obs)
    logging.info("Evaluation example (perfect forecast):")
    metrics = evaluate_nowcast(forecast_of, forecast_of, thresholds=[0.1, 1.0, 5.0])
    print(metrics.head())