"""Depth decoding: raw sensor files -> metric meters, float32 (H, W).

[Reasonable Assumption] The user's `left_filled` PNGs follow the de-facto
RGB-D convention (RealSense / Azure Kinect, the same sensors GID used) of
16-bit millimeters; `left_filled_np` are float arrays in either mm or m.
Unit is auto-detected per sequence (median positive value > 80 => mm) and can
be pinned in DepthConfig. Invalid pixels (<=0 after conversion, or beyond the
metric clamp) are set to 0 and treated as "no ground truth" downstream —
matching the model losses, which mask out non-positive depth.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import DepthConfig
from .discover import FrameRecord, SequenceRecord

log = logging.getLogger("data_engine.depth")

_UNIT_SCALE = {"m": 1.0, "mm": 1e-3, "cm": 1e-2}


def _read_raw(rec: FrameRecord, cfg: DepthConfig) -> np.ndarray:
    order = (
        [rec.depth_npy, rec.depth_png] if cfg.prefer_npy else [rec.depth_png, rec.depth_npy]
    )
    for p in order:
        if p is None:
            continue
        if p.suffix.lower() in (".npy", ".npz"):
            arr = np.load(str(p))
            if isinstance(arr, np.lib.npyio.NpzFile):
                arr = arr[list(arr.files)[0]]
            return np.asarray(arr)
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)  # preserves 16-bit
        if img is None:
            raise IOError(f"cv2 failed to read depth: {p}")
        if img.ndim == 3:                                # encoded-in-channels fallback
            img = img[..., 0]
        return img
    raise FileNotFoundError(f"frame {rec.name}: no depth file available")


def detect_unit_scale(seq: SequenceRecord, cfg: DepthConfig, probe_frames: int = 5) -> float:
    """Return multiplicative scale raw->meters for this sequence."""
    if cfg.unit != "auto":
        return _UNIT_SCALE[cfg.unit] * cfg.extra_scale
    vals = []
    step = max(len(seq.frames) // probe_frames, 1)
    for rec in seq.frames[::step][:probe_frames]:
        raw = _read_raw(rec, cfg).astype(np.float64)
        pos = raw[raw > 0]
        if pos.size:
            vals.append(np.median(pos))
    med = float(np.median(vals)) if vals else 0.0
    # Metric scenes live in 0.01-10 m. A median of, say, 4500 must be mm.
    if med > 800.0:
        scale = 1e-3
    elif med > 80.0:        # ambiguous mm/cm zone; cm sensors are rare -> mm
        scale = 1e-3
    else:
        scale = 1.0
    log.info("[%s] depth unit auto-detect: median raw=%.2f -> scale=%g", seq.seq_id, med, scale)
    return scale * cfg.extra_scale


def load_depth_meters(rec: FrameRecord, cfg: DepthConfig, scale: float) -> np.ndarray:
    """Return float32 (H, W) metric depth; invalid pixels are exactly 0."""
    d = _read_raw(rec, cfg).astype(np.float32) * scale
    invalid = ~np.isfinite(d) | (d < cfg.min_depth_m) | (d > cfg.max_depth_m)
    d[invalid] = 0.0
    return d
