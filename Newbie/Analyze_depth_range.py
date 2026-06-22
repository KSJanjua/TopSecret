#!/usr/bin/env python
"""analyze_depth_range.py - choose max_depth and depth-range bins from YOUR data.

The paper's range (0.01-10 m, balanced, objects at 4-8 m) does not match a ZED
capture (~0-45 m, foreground people at 0-4 m, a long static background tail).
This standalone script (numpy + opencv + json only; no project imports) reads the
generated annotations, reloads a SAMPLE of the raw depth maps WITHOUT the 10 m
clamp, and reports everything needed to set `max_depth` and the binning scheme:

  1. Dense depth distribution: percentiles + tail fractions (how much lives past
     10 m -> how much the current clamp is throwing away).
  2. Recommended max_depth = round-up of the 99.5th percentile.
  3. Per-instance depth layers recomputed from RAW depth, and how many instances
     are LOST to the clamp at several candidate max_depth values (i.e. the source
     of your "~19% zero-depth instances"). This tells you how high max_depth must
     go to recover them.
  4. Foreground (inside instance masks) vs background depth distributions, which
     quantifies the 0-4 m people vs far-background skew.
  5. Bin populations for several UNIFORM partitions, plus QUANTILE (equal-mass)
     and LOG-spaced bin edges. Uniform is the paper-faithful choice; quantile/log
     fit a skewed distribution better. The printed populations + balance score let
     you see the fragmentation the paper's Table 5 warns about.

Run:
  python analyze_depth_range.py --data-root gid_custom --split train \\
      --max-seqs 40 --frames-per-seq 10 --num-bins 8 \\
      --candidate-maxdepths 10 15 20 30 45 --save-hist depth_hist.png

Notes
-----
* RAW depth is read from each frame's `depth_npy` (falling back to `depth_png`)
  and multiplied by the manifest `depth_scale_to_m`, exactly as the data engine
  does -- but WITHOUT the upper clamp, so the true far field is visible.
* Reports the raw median too, so you can sanity-check that depth is in metres
  (median should look like a plausible scene distance, e.g. 3-15, not ~5000).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

MIN_DEPTH = 0.01                      # valid-pixel floor (paper Fig. 2)
MAX_PIX_PER_FRAME = 20000             # subsample dense pixels per frame for memory
MAX_TOTAL_PIX = 6_000_000             # global cap on collected dense pixels


# --------------------------------------------------------------------------- #
#  raw depth IO (no clamp)
# --------------------------------------------------------------------------- #
def load_raw_depth(frame: dict, scale: float) -> Optional[np.ndarray]:
    """Return float32 (H, W) metric depth with NO upper clamp; None on failure."""
    path = frame.get("depth_npy") or frame.get("depth_png")
    if not path or not os.path.exists(path):
        return None
    if str(path).lower().endswith((".npy", ".npz")):
        arr = np.load(path)
        if isinstance(arr, np.lib.npyio.NpzFile):
            arr = arr[list(arr.files)[0]]
        d = np.asarray(arr, dtype=np.float32)
    else:
        d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
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


# --------------------------------------------------------------------------- #
#  reporting helpers
# --------------------------------------------------------------------------- #
def pct_line(name: str, a: np.ndarray) -> None:
    if a.size == 0:
        print(f"  {name:<22}: (empty)")
        return
    qs = [1, 5, 25, 50, 75, 90, 95, 99, 99.5, 99.9]
    vals = np.percentile(a, qs)
    s = "  ".join(f"p{q}={v:.2f}" for q, v in zip(qs, vals))
    print(f"  {name:<22}: {s}  max={a.max():.2f}")


def balance_score(counts: np.ndarray) -> float:
    """Normalized entropy in [0,1]; 1 = perfectly balanced bins, ~0 = one bin."""
    p = counts / max(counts.sum(), 1)
    p = p[p > 0]
    if p.size <= 1:
        return 0.0
    return float(-(p * np.log(p)).sum() / math.log(len(counts)))


def uniform_report(depth: np.ndarray, max_depth: float, partition: float) -> None:
    r_d = max(1, int(math.ceil(max_depth / partition)))
    edges = np.arange(0, max_depth + partition, partition)[: r_d + 1]
    edges[-1] = max_depth
    counts, _ = np.histogram(np.clip(depth, 0, max_depth), bins=edges)
    frac = counts / max(counts.sum(), 1) * 100
    bars = "  ".join(f"[{edges[i]:.0f}-{edges[i+1]:.0f}]={frac[i]:.1f}%"
                     for i in range(len(counts)))
    print(f"  partition={partition:>4.1f} m -> r_d={r_d:>2} bins  "
          f"balance={balance_score(counts):.2f}")
    print(f"      {bars}")


def quantile_edges(depth: np.ndarray, k: int, max_depth: float) -> np.ndarray:
    qs = np.linspace(0, 1, k + 1)
    e = np.quantile(np.clip(depth, MIN_DEPTH, max_depth), qs)
    e[0], e[-1] = 0.0, max_depth
    return np.unique(np.round(e, 2))


def log_edges(k: int, lo: float, max_depth: float) -> np.ndarray:
    lo = max(lo, MIN_DEPTH)
    e = np.logspace(math.log10(lo), math.log10(max_depth), k + 1)
    e[0] = 0.0
    return np.round(e, 2)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Depth range / bin analyzer")
    ap.add_argument("--data-root", default="gid_custom")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-seqs", type=int, default=40,
                    help="sample at most this many sequences")
    ap.add_argument("--frames-per-seq", type=int, default=10,
                    help="evenly sample this many frames per sequence")
    ap.add_argument("--num-bins", type=int, default=8,
                    help="K for the quantile/log bin proposals")
    ap.add_argument("--candidate-maxdepths", type=float, nargs="+",
                    default=[10, 15, 20, 30, 45])
    ap.add_argument("--uniform-partitions", type=float, nargs="+",
                    default=[1, 2, 3, 5])
    ap.add_argument("--save-hist", default=None, help="optional histogram PNG path")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    random.seed(args.seed)

    root = Path(args.data_root)
    seq_ids = [s for s in (root / f"{args.split}.txt").read_text().splitlines() if s.strip()]
    random.shuffle(seq_ids)
    seq_ids = seq_ids[: args.max_seqs]
    print(f"[data] {len(seq_ids)} sampled sequences from {args.split} split\n")

    dense: List[np.ndarray] = []
    fg: List[np.ndarray] = []
    bg: List[np.ndarray] = []
    inst_true_layers: List[float] = []
    inst_total = 0
    raw_medians: List[float] = []
    # per candidate max_depth: how many instances keep >=1 valid pixel <= cand
    recover: Dict[float, int] = {c: 0 for c in args.candidate_maxdepths}

    for sid in seq_ids:
        man_path = root / sid / "annotations.json"
        if not man_path.exists():
            continue
        man = json.load(open(man_path))
        scale = float(man.get("depth_scale_to_m", 1.0))
        fkeys = sorted(man["frames"])
        if not fkeys:
            continue
        step = max(len(fkeys) // args.frames_per_seq, 1)
        for fkey in fkeys[::step][: args.frames_per_seq]:
            frame = man["frames"][fkey]
            d = load_raw_depth(frame, scale)
            if d is None:
                continue
            raw_medians.append(float(np.median(d[d > 0])) if (d > 0).any() else 0.0)
            valid = np.isfinite(d) & (d > MIN_DEPTH)

            id_map = load_id_map(frame, d.shape[:2])
            fg_mask = np.zeros(d.shape, bool) if id_map is None else (id_map > 0)

            # dense + fg/bg samples (subsampled)
            dv = d[valid]
            if dv.size:
                if dv.size > MAX_PIX_PER_FRAME:
                    dv = dv[np.random.randint(0, dv.size, MAX_PIX_PER_FRAME)]
                dense.append(dv)
            fgv = d[valid & fg_mask]
            bgv = d[valid & ~fg_mask]
            if fgv.size:
                fg.append(fgv[np.random.randint(0, fgv.size, min(fgv.size, 5000))])
            if bgv.size:
                bg.append(bgv[np.random.randint(0, bgv.size, min(bgv.size, 5000))])

            # per-instance TRUE depth layers (no clamp) + clamp-recovery counts
            for inst in frame["instances"]:
                inst_total += 1
                tid = inst["track_id"]
                if id_map is None:
                    continue
                mvals = d[(id_map == tid) & valid]
                if mvals.size == 0:
                    continue
                inst_true_layers.append(float(mvals.mean()))
                for cand in args.candidate_maxdepths:
                    if (mvals <= cand).any():
                        recover[cand] += 1

        if sum(x.size for x in dense) > MAX_TOTAL_PIX:
            break

    dense_a = np.concatenate(dense) if dense else np.array([])
    fg_a = np.concatenate(fg) if fg else np.array([])
    bg_a = np.concatenate(bg) if bg else np.array([])
    layers_a = np.array(inst_true_layers)

    print("=== units sanity check ===")
    print(f"  raw per-frame median depth (after scale): "
          f"{np.median(raw_medians):.2f} m   (expect a plausible scene distance)\n")

    print("=== DENSE DEPTH DISTRIBUTION (all valid pixels) ===")
    pct_line("dense", dense_a)
    if dense_a.size:
        for t in args.candidate_maxdepths:
            print(f"      fraction beyond {t:>4.0f} m : "
                  f"{100*(dense_a > t).mean():5.2f}%")
    rec_max = float(np.percentile(dense_a, 99.5)) if dense_a.size else 10.0
    print(f"\n  >> recommended max_depth ~= {math.ceil(rec_max)} m "
          f"(99.5th percentile = {rec_max:.2f})\n")

    print("=== FOREGROUND (in-mask) vs BACKGROUND ===")
    pct_line("foreground (people)", fg_a)
    pct_line("background", bg_a)
    print()

    print("=== PER-INSTANCE DEPTH LAYERS (recomputed from RAW depth) ===")
    pct_line("true depth layers", layers_a)
    print(f"  instances sampled            : {inst_total}")
    print(f"  with >=1 valid raw pixel     : {layers_a.size}")
    print("  instances RECOVERED vs lost to the clamp, per candidate max_depth:")
    for cand in args.candidate_maxdepths:
        kept = recover[cand]
        lost = inst_total - kept
        print(f"      max_depth={cand:>4.0f} m : kept={kept:>6}  "
              f"lost(->depth 0)={lost:>6}  ({100*lost/max(inst_total,1):5.2f}%)")
    print("  (the 'lost' column at max_depth=10 should match your ~19%.)\n")

    if dense_a.size:
        md = math.ceil(rec_max)
        print(f"=== UNIFORM BIN POPULATIONS over [0, {md}] m ===")
        for p in args.uniform_partitions:
            uniform_report(dense_a, md, p)
        print("  (paper Table 5 found too-many fine bins hurt; prefer the largest")
        print("   partition whose balance score stays reasonable.)\n")

        print(f"=== ADAPTIVE BIN EDGES (K={args.num_bins}) ===")
        qe = quantile_edges(dense_a, args.num_bins, md)
        le = log_edges(args.num_bins, np.percentile(dense_a, 1), md)
        print(f"  quantile (equal-mass): {qe.tolist()}")
        print(f"  log-spaced           : {le.tolist()}")
        print("  -> quantile/log concentrate resolution where your people are (0-4 m);")
        print("     using them requires per-bin width in the HDI update (I will supply it).\n")

    if args.save_hist and dense_a.size:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.figure(figsize=(9, 4))
            plt.hist(dense_a, bins=100, alpha=0.6, label="all", color="#4488cc")
            if fg_a.size:
                plt.hist(fg_a, bins=100, alpha=0.6, label="foreground", color="#cc5533")
            plt.xlabel("depth (m)"); plt.ylabel("pixel count"); plt.legend()
            plt.title(f"Depth distribution ({args.split})")
            plt.tight_layout(); plt.savefig(args.save_hist, dpi=110)
            print(f"[hist] saved -> {args.save_hist}")
        except Exception as e:                       # matplotlib optional
            print(f"[hist] skipped ({e})")


if __name__ == "__main__":
    main()
