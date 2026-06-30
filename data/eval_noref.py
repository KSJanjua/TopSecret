#!/usr/bin/env python3
"""
No-reference image-quality evaluation for UNPAIRED low-light datasets
(DICM, LIME, NPE, MEF) — where there is no ground truth, so full-reference
metrics (ΔE2000/PSNR/SSIM) do not apply.

Computes NIQE and BRISQUE (both: lower = better) per image, using pyiqa — the
same library evaluate.py uses for NIQE, so numbers stay consistent. Supports
several methods x several datasets in one run and plots a comparison.

IMPORTANT (be honest in the report): NIQE/BRISQUE measure *naturalness /
distortion*, NOT colour fidelity. They show generalisation to real-world
photos; they cannot prove "no colour shift" (that needs paired data -> use
evaluate.py on LOL-v1/LOL-v2).

Usage — one --set per (method, dataset, folder-of-enhanced-images):
  python eval_noref.py \
      --set ccdnet DICM results/ccdnet/DICM \
      --set dccnet DICM results/dccnet/DICM \
      --set ccdnet LIME results/ccdnet/LIME \
      --set dccnet LIME results/dccnet/LIME \
      --set ccdnet NPE  results/ccdnet/NPE \
      --set dccnet NPE  results/dccnet/NPE \
      --set ccdnet MEF  results/ccdnet/MEF \
      --set dccnet MEF  results/dccnet/MEF \
      --out-dir noref_results --device cuda

Outputs in --out-dir:
  <method>_<dataset>_noref.csv   per-image NIQE/BRISQUE for each set
  noref_summary.csv              mean ± std per (method, dataset)
  noref_summary_bar.png          grouped bars: dataset on x, method = colour
  noref_box.png                  per-image distribution (box plots)
"""
from __future__ import annotations

import argparse
import os
import os.path as osp

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate import load_image_rgb, list_images          # reuse canonical sRGB loader

PALETTE = ["#2E5496", "#ED7D31", "#70AD47", "#7030A0", "#1F4E79", "#C55A11"]
METRICS = ["NIQE", "BRISQUE"]                              # both lower-is-better


def to_tensor(img_hwc, device):
    return torch.from_numpy(img_hwc.transpose(2, 0, 1)).unsqueeze(0).float().to(device)


def build_metrics(device):
    import pyiqa
    m = {"NIQE": pyiqa.create_metric("niqe", device=device),
         "BRISQUE": pyiqa.create_metric("brisque", device=device)}
    for v in m.values():
        v.eval()
    return m


@torch.no_grad()
def score_folder(folder, metrics, device):
    rows = []
    files = list_images(folder)
    if not files:
        print(f"  [warn] no images in {folder}")
        return pd.DataFrame(columns=["image"] + METRICS)
    for p in files:
        try:
            t = to_tensor(load_image_rgb(p), device)
            row = {"image": osp.basename(p)}
            for name, fn in metrics.items():
                row[name] = float(fn(t).item())
            rows.append(row)
        except Exception as e:
            print(f"  [skip] {osp.basename(p)}: {e}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def plot_summary_bar(summary, out_path):
    datasets = list(dict.fromkeys(summary["dataset"]))
    methods = list(dict.fromkeys(summary["method"]))
    x = np.arange(len(datasets))
    w = 0.8 / max(len(methods), 1)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(6.6 * len(METRICS), 4.6))
    axes = np.atleast_1d(axes).ravel()
    for ax, metric in zip(axes, METRICS):
        for i, mt in enumerate(methods):
            means, errs = [], []
            for ds in datasets:
                sub = summary[(summary.method == mt) & (summary.dataset == ds)]
                means.append(float(sub[f"{metric}_mean"].iloc[0]) if len(sub) else np.nan)
                errs.append(float(sub[f"{metric}_std"].iloc[0]) if len(sub) else 0.0)
            pos = x + (i - (len(methods) - 1) / 2) * w
            bars = ax.bar(pos, means, w, yerr=errs, capsize=3, label=mt,
                          color=PALETTE[i % len(PALETTE)], edgecolor="white", linewidth=0.7)
            for b, v in zip(bars, means):
                if not np.isnan(v):
                    ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}",
                            ha="center", va="bottom", fontsize=8)
        ax.set_title(f"{metric}  (lower is better)", fontsize=12, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(datasets)
        ax.grid(axis="y", alpha=0.3); ax.margins(y=0.16)
        ax.legend(fontsize=9, title="method")
    fig.suptitle("No-reference quality on unpaired datasets", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=160); plt.close(fig)
    print("Saved", out_path)


def plot_box(per_set, out_path):
    # per_set: dict (method,dataset) -> DataFrame
    datasets = sorted({d for _, d in per_set})
    methods = sorted({m for m, _ in per_set})
    fig, axes = plt.subplots(1, len(METRICS), figsize=(6.6 * len(METRICS), 4.6))
    axes = np.atleast_1d(axes).ravel()
    for ax, metric in zip(axes, METRICS):
        data, labels, colors = [], [], []
        for di, ds in enumerate(datasets):
            for mi, mt in enumerate(methods):
                df = per_set.get((mt, ds))
                if df is not None and metric in df and len(df):
                    data.append(df[metric].values)
                    labels.append(f"{ds}\n{mt}")
                    colors.append(PALETTE[mi % len(PALETTE)])
        if not data:
            continue
        bp = ax.boxplot(data, patch_artist=True, showmeans=True)
        ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels, fontsize=8)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.55)
        ax.set_title(f"{metric}  (lower is better)", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Per-image distribution (no-reference)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=160); plt.close(fig)
    print("Saved", out_path)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="No-reference (NIQE/BRISQUE) evaluation for unpaired datasets.")
    ap.add_argument("--set", nargs=3, action="append", metavar=("METHOD", "DATASET", "DIR"),
                    required=True, help="repeatable: method name, dataset name, folder of enhanced images")
    ap.add_argument("--out-dir", default="noref_results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading NIQE + BRISQUE on {args.device} ...")
    metrics = build_metrics(torch.device(args.device))

    per_set, summary_rows = {}, []
    for method, dataset, folder in args.set:
        print(f"[{method} / {dataset}] scoring {folder}")
        df = score_folder(folder, metrics, args.device)
        if df.empty:
            continue
        df.to_csv(osp.join(args.out_dir, f"{method}_{dataset}_noref.csv"), index=False)
        per_set[(method, dataset)] = df
        row = {"method": method, "dataset": dataset, "n": len(df)}
        for m in METRICS:
            row[f"{m}_mean"] = float(df[m].mean()); row[f"{m}_std"] = float(df[m].std(ddof=1) if len(df) > 1 else 0)
        summary_rows.append(row)
        print(f"   n={len(df)} | " + " | ".join(f"{m}={row[f'{m}_mean']:.3f}" for m in METRICS))

    if not summary_rows:
        raise SystemExit("Nothing scored — check your --set folders.")
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(osp.join(args.out_dir, "noref_summary.csv"), index=False)
    print("\nSummary:\n", summary.to_string(index=False))

    plot_summary_bar(summary, osp.join(args.out_dir, "noref_summary_bar.png"))
    plot_box(per_set, osp.join(args.out_dir, "noref_box.png"))
    print("Done.")


if __name__ == "__main__":
    main()
