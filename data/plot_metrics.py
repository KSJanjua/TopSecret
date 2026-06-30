#!/usr/bin/env python3
"""
Compute metrics on inferred photos and plot them.

Two ways to use it:

  (A) You already ran evaluate.py and have bench/<name>_per_image.csv files
      -> just plot (and compare methods if there are several):
        python plot_metrics.py --csv-dir bench --out-dir plots

  (B) Compute fresh from a folder of enhanced images vs the ground truth,
      then plot (uses the SAME metric functions as evaluate.py):
        python plot_metrics.py --gt-dir dataset/LOLv1/eval15/high \
            --pred-dir results/ccdnet --name ccdnet --out-dir plots

  You can also compute several methods into the same out-dir (run B once per
  method with a different --name / --pred-dir), then run A on that dir to get a
  comparison across all of them.

Produces three PNGs in --out-dir:
  metrics_summary_bar.png  - mean of each metric per method (with std error bars)
  metrics_per_image.png    - each metric across the individual test images
  metrics_box.png          - distribution (box plot) of each metric per method
and writes <name>_per_image.csv in compute mode.
"""
from __future__ import annotations

import argparse
import glob
import os
import os.path as osp

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # headless-safe (no display needed)
import matplotlib.pyplot as plt

# reuse the exact, validated metric functions + loader from the harness
from evaluate import (load_image_rgb, list_images, stem_key,
                      psnr_rgb, psnr_y, ssim_canonical, delta_e2000, lab_rmse)

# metric -> (nice title, lower_is_better)
METRIC_INFO = {
    "DeltaE2000": ("ΔE2000  (colour, lower better)", True),
    "LAB_RMSE":   ("LAB-RMSE  (colour, lower better)", True),
    "PSNR_RGB":   ("PSNR  (dB, higher better)", False),
    "PSNR_Y":     ("PSNR-Y  (dB, higher better)", False),
    "SSIM":       ("SSIM  (higher better)", False),
    "LPIPS":      ("LPIPS  (lower better)", True),
    "NIQE":       ("NIQE  (lower better)", True),
}
# consistent, slide-matching palette (PRISM blue + Office accents)
PALETTE = ["#2E5496", "#ED7D31", "#70AD47", "#7030A0", "#1F4E79", "#C55A11"]


# --------------------------------------------------------------------------- #
def compute_per_image(gt_dir, pred_dir, name, out_dir):
    """Compute per-image metrics for one method; returns a DataFrame and saves CSV."""
    gt_map = {stem_key(p): p for p in list_images(gt_dir)}
    rows = []
    for pred_path in list_images(pred_dir):
        k = stem_key(pred_path)
        if k not in gt_map:
            print(f"  [skip] no ground-truth match for {osp.basename(pred_path)}")
            continue
        pred, gt = load_image_rgb(pred_path), load_image_rgb(gt_map[k])
        if pred.shape != gt.shape:
            print(f"  [skip] shape mismatch for {osp.basename(pred_path)}: {pred.shape} vs {gt.shape}")
            continue
        rows.append({"image": osp.basename(pred_path),
                     "DeltaE2000": delta_e2000(pred, gt),
                     "LAB_RMSE": lab_rmse(pred, gt),
                     "PSNR_RGB": psnr_rgb(pred, gt),
                     "PSNR_Y": psnr_y(pred, gt),
                     "SSIM": ssim_canonical(pred, gt)})
    if not rows:
        raise SystemExit("No matched image pairs found — check --gt-dir / --pred-dir and filenames.")
    df = pd.DataFrame(rows).sort_values("image").reset_index(drop=True)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = osp.join(out_dir, f"{name}_per_image.csv")
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}  ({len(df)} images)")
    print("Means:", {k: round(float(df[k].mean()), 4) for k in df.columns if k != "image"})
    return df


def load_csvs(csv_dir):
    """Load every *_per_image.csv in a dir -> {method_name: DataFrame}."""
    files = sorted(glob.glob(osp.join(csv_dir, "*_per_image.csv")))
    if not files:
        raise SystemExit(f"No *_per_image.csv files in {csv_dir}")
    return {osp.basename(f).replace("_per_image.csv", ""): pd.read_csv(f) for f in files}


def metrics_present(tables):
    cols = set()
    for df in tables.values():
        cols |= {c for c in df.columns if c != "image"}
    # keep canonical order, only those actually present
    return [m for m in METRIC_INFO if m in cols]


def _grid(n):
    ncol = 2 if n > 1 else 1
    nrow = int(np.ceil(n / ncol))
    return nrow, ncol


# --------------------------------------------------------------------------- #
def plot_summary_bar(tables, metrics, out_path):
    methods = list(tables)
    nrow, ncol = _grid(len(metrics))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 4.0 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, m in zip(axes, metrics):
        title, lower = METRIC_INFO[m]
        means = [tables[mt][m].mean() for mt in methods]
        errs = [tables[mt][m].std(ddof=1) if len(tables[mt]) > 1 else 0 for mt in methods]
        bars = ax.bar(methods, means, yerr=errs, capsize=4,
                      color=[PALETTE[i % len(PALETTE)] for i in range(len(methods))],
                      edgecolor="white", linewidth=0.8)
        for b, v in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.margins(y=0.18)
        best = (min if lower else max)(means)
        ax.annotate("best ↓" if lower else "best ↑",
                    xy=(means.index(best), best), fontsize=8, color="#555",
                    xytext=(0, 14), textcoords="offset points", ha="center")
    for ax in axes[len(metrics):]:
        ax.axis("off")
    fig.suptitle("Metric summary (mean ± std across test images)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=160); plt.close(fig)
    print("Saved", out_path)


def plot_per_image(tables, metrics, out_path):
    methods = list(tables)
    ref = tables[methods[0]]
    x = np.arange(len(ref)); labels = ref["image"].tolist()
    nrow, ncol = _grid(len(metrics))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0 * ncol, 4.0 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, m in zip(axes, metrics):
        title, _ = METRIC_INFO[m]
        for i, mt in enumerate(methods):
            d = tables[mt]
            ax.plot(np.arange(len(d)), d[m].values, marker="o", ms=4,
                    color=PALETTE[i % len(PALETTE)], label=mt)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        ax.grid(alpha=0.3)
        if len(methods) > 1:
            ax.legend(fontsize=8)
    for ax in axes[len(metrics):]:
        ax.axis("off")
    fig.suptitle("Per-image metrics", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=160); plt.close(fig)
    print("Saved", out_path)


def plot_box(tables, metrics, out_path):
    methods = list(tables)
    nrow, ncol = _grid(len(metrics))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 4.0 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, m in zip(axes, metrics):
        title, _ = METRIC_INFO[m]
        data = [tables[mt][m].values for mt in methods]
        bp = ax.boxplot(data, patch_artist=True, showmeans=True)
        ax.set_xticks(range(1, len(methods) + 1)); ax.set_xticklabels(methods)
        for patch, i in zip(bp["boxes"], range(len(methods))):
            patch.set_facecolor(PALETTE[i % len(PALETTE)]); patch.set_alpha(0.55)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
    for ax in axes[len(metrics):]:
        ax.axis("off")
    fig.suptitle("Metric distribution across test images", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=160); plt.close(fig)
    print("Saved", out_path)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Compute + plot metrics on inferred photos.")
    ap.add_argument("--csv-dir", help="dir of existing *_per_image.csv (plot/compare mode)")
    ap.add_argument("--gt-dir", help="ground-truth dir (compute mode)")
    ap.add_argument("--pred-dir", help="enhanced/inferred images dir (compute mode)")
    ap.add_argument("--name", default="ccdnet", help="method name for the CSV (compute mode)")
    ap.add_argument("--out-dir", default="plots")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.gt_dir and args.pred_dir:                 # compute mode
        compute_per_image(args.gt_dir, args.pred_dir, args.name, args.out_dir)
        tables = load_csvs(args.out_dir)
    elif args.csv_dir:                                # plot-only mode
        tables = load_csvs(args.csv_dir)
    else:
        raise SystemExit("Give either --csv-dir, or (--gt-dir and --pred-dir).")

    metrics = metrics_present(tables)
    print(f"Methods: {list(tables)} | metrics: {metrics}")
    plot_summary_bar(tables, metrics, osp.join(args.out_dir, "metrics_summary_bar.png"))
    plot_per_image(tables, metrics, osp.join(args.out_dir, "metrics_per_image.png"))
    plot_box(tables, metrics, osp.join(args.out_dir, "metrics_box.png"))
    print("Done.")


if __name__ == "__main__":
    main()
