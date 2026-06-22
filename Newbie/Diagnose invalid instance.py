#!/usr/bin/env python
"""diagnose_invalid_instances.py - is the ~18% zero-depth a SENSOR hole or a BUG?

A FALSIFIABLE test. For every instance it measures signatures that separate
"the ZED produced no depth there" (real data hole) from "the mask and depth are
misaligned / mispaired / mis-scaled" (pipeline bug), and saves overlay images so
you can SEE the depth holes sitting on the people.

Signatures and what each direction means
----------------------------------------
1. Mask AREA, invalid vs valid. Stereo fails on small/distant objects, so a
   sensor hole predicts invalid instances are much SMALLER. A bug has no reason
   to prefer small objects.   -> ratio >> 1 supports SENSOR.
2. Mask CENTROID-X, invalid vs valid. Stereo has NO depth at the left image
   border (no right-image correspondence), so invalid masks should skew LEFT.
   -> invalid median-x noticeably smaller supports SENSOR (ZED fingerprint).
3. Per-instance VALID-PIXEL FRACTION histogram. A sensor characteristic gives a
   SMOOTH distribution piling toward 0; an all-or-nothing bug gives a BIMODAL
   spike at exactly 0 with a gap.   -> smooth tail supports SENSOR.
4. ALIGNMENT CONTROL on VALID instances: depth INSIDE the mask vs a RING just
   outside, plus within-mask coefficient of variation. A clean foreground/
   background depth STEP at the boundary + low internal variance means masks
   track the depth -> alignment is correct for the pipeline, so the 18% cannot
   be a misalignment artifact.   -> large step + low CV supports SENSOR.
5. Column-wise INVALIDITY profile of the raw depth (no masks): characterizes the
   sensor. Stereo -> invalid fraction spikes at the left edge.

The clinching logic: the SAME reader / pairing / scale yields the valid
instances too. A pairing/scale/coordinate bug is all-or-nothing -- it corrupts
EVERY instance, not exactly the small/left/distant 18%. Selective failure that
correlates with object size IS the signature of missing data.

Run:
  python diagnose_invalid_instances.py --data-root gid_custom --split train \\
      --max-seqs 30 --frames-per-seq 8 --num-overlays 12 --out-dir invalid_diag
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

MIN_DEPTH = 0.01
RGB_KEYS = ("rgb", "image", "rgb_path", "left_rgb", "rgb_filled")
DEPTH_KEYS = ("depth_npy", "depth_png")


# --------------------------------------------------------------------------- #
#  IO (no clamp)
# --------------------------------------------------------------------------- #
def _first_path(frame: dict, keys) -> Optional[str]:
    for k in keys:
        p = frame.get(k)
        if p and os.path.exists(p):
            return p
    return None


def load_raw_depth(frame: dict, scale: float) -> Optional[np.ndarray]:
    p = _first_path(frame, DEPTH_KEYS)
    if p is None:
        return None
    if str(p).lower().endswith((".npy", ".npz")):
        arr = np.load(p)
        if isinstance(arr, np.lib.npyio.NpzFile):
            arr = arr[list(arr.files)[0]]
        d = np.asarray(arr, np.float32)
    else:
        d = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if d is None:
            return None
        d = d.astype(np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    return d * scale


def load_id_map(frame: dict, hw) -> Optional[np.ndarray]:
    p = frame.get("object_mask")
    if not p or not os.path.exists(p):
        return None
    m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.shape[:2] != hw:
        m = cv2.resize(m, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    return m


def load_rgb(frame: dict, hw) -> Optional[np.ndarray]:
    p = _first_path(frame, RGB_KEYS)
    if p is None:
        return None
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        return None
    if img.shape[:2] != hw:
        img = cv2.resize(img, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
    return img


# --------------------------------------------------------------------------- #
#  overlay: RGB+outline | depth (invalid=black) | validity (valid=white)
# --------------------------------------------------------------------------- #
def save_overlay(rgb, depth, valid, mask, path, vmax) -> None:
    h, w = depth.shape
    rgb = rgb.copy() if rgb is not None else np.zeros((h, w, 3), np.uint8)

    dn = np.clip(depth / max(vmax, 1e-6), 0, 1)
    depth_col = cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    depth_col[~valid] = (0, 0, 0)                       # holes are black

    valid_img = np.where(valid[..., None], 255, 0).astype(np.uint8)
    valid_img = np.repeat(valid_img, 3, axis=2)

    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    for img in (rgb, depth_col, valid_img):
        cv2.drawContours(img, cnts, -1, (0, 0, 255), 2)

    panel = np.concatenate([rgb, depth_col, valid_img], axis=1)
    cv2.imwrite(str(path), panel)


def pct(a: np.ndarray, qs=(10, 25, 50, 75, 90)) -> str:
    if a.size == 0:
        return "(empty)"
    v = np.percentile(a, qs)
    return "  ".join(f"p{q}={x:.3g}" for q, x in zip(qs, v))


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Invalid-instance sensor-vs-bug diagnostic")
    ap.add_argument("--data-root", default="gid_custom")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-seqs", type=int, default=30)
    ap.add_argument("--frames-per-seq", type=int, default=8)
    ap.add_argument("--num-overlays", type=int, default=12)
    ap.add_argument("--ring-px", type=int, default=15, help="dilation for the outside ring")
    ap.add_argument("--out-dir", default="invalid_diag")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    random.seed(args.seed)

    out = Path(args.out_dir)
    (out / "overlays").mkdir(parents=True, exist_ok=True)
    kernel = np.ones((args.ring_px, args.ring_px), np.uint8)

    root = Path(args.data_root)
    seqs = [s for s in (root / f"{args.split}.txt").read_text().splitlines() if s.strip()]
    random.shuffle(seqs)
    seqs = seqs[: args.max_seqs]

    # per-instance arrays
    area_valid, area_invalid = [], []
    cx_valid, cx_invalid = [], []
    vfrac_all = []
    step_valid, cv_valid = [], []        # alignment control
    ring_valid_for_invalid = 0           # invalid instances that DO have valid depth nearby
    n_invalid = 0
    col_invalid = None                   # column-wise invalid fraction accumulator
    col_count = 0
    overlay_pool = []                    # (rgb, depth, valid, mask, vmax) for invalid

    vmax_global = 40.0

    for sid in seqs:
        mpath = root / sid / "annotations.json"
        if not mpath.exists():
            continue
        man = json.load(open(mpath))
        scale = float(man.get("depth_scale_to_m", 1.0))
        fkeys = sorted(man["frames"])
        step = max(len(fkeys) // args.frames_per_seq, 1)
        for fkey in fkeys[::step][: args.frames_per_seq]:
            frame = man["frames"][fkey]
            d = load_raw_depth(frame, scale)
            if d is None:
                continue
            h, w = d.shape
            valid = np.isfinite(d) & (d > MIN_DEPTH)
            id_map = load_id_map(frame, (h, w))
            if id_map is None:
                continue

            # column invalidity profile (sensor characterization, mask-free)
            prof = 1.0 - valid.mean(axis=0)               # length-w
            if col_invalid is None or col_invalid.shape[0] != w:
                col_invalid = np.zeros(w) if col_invalid is None else col_invalid
            if col_invalid.shape[0] == w:
                col_invalid = col_invalid + prof
                col_count += 1

            rgb = load_rgb(frame, (h, w))

            for inst in frame["instances"]:
                tid = inst["track_id"]
                m = id_map == tid
                area = int(m.sum())
                if area == 0:
                    continue
                vin = valid & m
                vfrac = vin.sum() / area
                vfrac_all.append(vfrac)
                xs = np.where(m.any(axis=0))[0]
                cx = (xs.mean() / w) if xs.size else 0.5

                if vfrac == 0:                            # INVALID instance
                    n_invalid += 1
                    area_invalid.append(area)
                    cx_invalid.append(cx)
                    ring = (cv2.dilate(m.astype(np.uint8), kernel) > 0) & ~m & valid
                    if ring.any():
                        ring_valid_for_invalid += 1
                    if len(overlay_pool) < args.num_overlays and rgb is not None:
                        overlay_pool.append((rgb, d, valid, m, vmax_global))
                else:                                     # VALID instance
                    area_valid.append(area)
                    cx_valid.append(cx)
                    di = d[vin]
                    inside = float(np.median(di))
                    cvv = float(di.std() / max(inside, 1e-6))
                    cv_valid.append(cvv)
                    ring = (cv2.dilate(m.astype(np.uint8), kernel) > 0) & ~m & valid
                    if ring.any():
                        step_valid.append(abs(inside - float(np.median(d[ring]))))

    area_valid = np.array(area_valid); area_invalid = np.array(area_invalid)
    cx_valid = np.array(cx_valid); cx_invalid = np.array(cx_invalid)
    vfrac_all = np.array(vfrac_all)
    step_valid = np.array(step_valid); cv_valid = np.array(cv_valid)

    print("=== (1) MASK AREA (pixels): invalid vs valid ===")
    print(f"  valid   : {pct(area_valid)}")
    print(f"  invalid : {pct(area_invalid)}")
    if area_valid.size and area_invalid.size:
        ratio = np.median(area_valid) / max(np.median(area_invalid), 1)
        verdict = "SENSOR (invalid are smaller)" if ratio > 2 else "inconclusive / check bug"
        print(f"  median-area ratio valid/invalid = {ratio:.1f}x  -> {verdict}\n")

    print("=== (2) MASK CENTROID-X (0=left edge, 1=right): invalid vs valid ===")
    print(f"  valid   median-x = {np.median(cx_valid):.3f}" if cx_valid.size else "  valid: (empty)")
    print(f"  invalid median-x = {np.median(cx_invalid):.3f}" if cx_invalid.size else "  invalid: (empty)")
    if cx_valid.size and cx_invalid.size:
        d_x = np.median(cx_valid) - np.median(cx_invalid)
        print(f"  invalid skewed left by {d_x:+.3f}  -> "
              f"{'SENSOR (left-border stereo dropout)' if d_x > 0.05 else 'no strong left skew'}\n")

    print("=== (3) PER-INSTANCE VALID-PIXEL FRACTION ===")
    if vfrac_all.size:
        zero = (vfrac_all == 0).mean()
        partial = ((vfrac_all > 0) & (vfrac_all < 0.5)).mean()
        print(f"  vfrac==0     : {100*zero:5.1f}%   (the lost instances)")
        print(f"  0<vfrac<0.5  : {100*partial:5.1f}%   (smooth tail toward 0)")
        print(f"  vfrac>=0.5   : {100*(vfrac_all>=0.5).mean():5.1f}%")
        print("  -> a non-trivial 0<vfrac<0.5 band = SMOOTH sensor falloff (not an "
              "all-or-nothing bug)\n")

    print("=== (4) ALIGNMENT CONTROL on VALID instances ===")
    print(f"  boundary depth STEP |inside-ring| (m) : {pct(step_valid)}")
    print(f"  within-mask coeff. of variation       : {pct(cv_valid)}")
    print("  -> a clear depth step at the boundary + low CV means masks ARE aligned")
    print("     to the depth, so the lost 18% cannot be a misalignment artifact.\n")

    print("=== (5) INVALID instances: is depth present NEARBY? ===")
    if n_invalid:
        print(f"  {ring_valid_for_invalid}/{n_invalid} "
              f"({100*ring_valid_for_invalid/n_invalid:.0f}%) sit next to valid depth")
        print("  (object-specific holes); the rest fall in larger regional holes.\n")

    if col_invalid is not None and col_count:
        prof = col_invalid / col_count
        w = prof.shape[0]
        left = prof[: w // 20].mean(); mid = prof[w//2 - w//40 : w//2 + w//40].mean()
        right = prof[-w // 20:].mean()
        print("=== (6) COLUMN-WISE DEPTH INVALIDITY (sensor fingerprint) ===")
        print(f"  invalid fraction: left-5%={left:.3f}  middle={mid:.3f}  right-5%={right:.3f}")
        print(f"  -> {'left-edge dropout present (stereo geometry)' if left > mid + 0.05 else 'no strong left-edge dropout'}\n")
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].plot(np.linspace(0, 1, w), prof); ax[0].set_title("invalid fraction vs image column")
            ax[0].set_xlabel("normalized x"); ax[0].set_ylabel("fraction invalid")
            if vfrac_all.size:
                ax[1].hist(vfrac_all, bins=40, color="#cc5533")
                ax[1].set_title("per-instance valid-pixel fraction"); ax[1].set_xlabel("vfrac")
            plt.tight_layout(); plt.savefig(out / "profiles.png", dpi=110)
            print(f"[plots] saved -> {out/'profiles.png'}")
        except Exception as e:
            print(f"[plots] skipped ({e})")

    for i, (rgb, d, valid, m, vmax) in enumerate(overlay_pool):
        save_overlay(rgb, d, valid, m, out / "overlays" / f"invalid_{i:02d}.png", vmax)
    print(f"[overlays] {len(overlay_pool)} saved -> {out/'overlays'} "
          f"(RGB+outline | depth, holes black | validity, valid white)")


if __name__ == "__main__":
    main()
