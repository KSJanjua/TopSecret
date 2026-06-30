#!/usr/bin/env python3
"""Visualize InstanceDepth predictions in TensorBoard, with CLEAN depth panels.

For N samples from a split it logs, per sample, a single row image:

    [ RGB | GT depth | Pred depth (clamped) | Pred depth (masked to GT) ]  (+ masks)

and the per-sample + mean depth metrics (RMS/REL/sigma1) as scalars.

Why the predicted depth looks clean here
----------------------------------------
A depth model predicts a value at EVERY pixel, including the far field where the
sensor returns no GT (GT == 0 -> shown black). Naive per-image min-max colorizing
lets a few extreme far-field predictions stretch the palette, so the unsupervised
region scatters into "random colors". This script instead:
  * picks ONE robust range [vmin,vmax] from VALID GT only (2nd/98th percentile),
    and colorizes pred AND GT with that same range (directly comparable);
  * clamps pred into [vmin,vmax] and scrubs NaN/Inf before colorizing;
  * paints invalid pixels black: GT where gt<=0, and the "masked" pred panel where
    GT is invalid -> the prediction goes black past the same distance as the GT.
The middle "Pred (clamped)" panel still shows the model's true far-field behavior;
the right "Pred (masked to GT)" panel is the clean, apples-to-apples comparison.

Usage
-----
    python visualize_tb.py \
        --model-config instancedepth/configs/instance_depth.yaml \
        --data-root gid_custom --checkpoint runs/phase3/ckpt_final.pth \
        --phase 3 --split test --num-samples 16 --out runs/phase3/tb_vis

    tensorboard --logdir runs/phase3/tb_vis      # then open the IMAGES tab
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from instancedepth.build import build_instance_depth_from_yaml
from instancedepth.data.gid_dataset import (
    GIDDatasetConfig, GIDInstanceDepthDataset, IMAGENET_MEAN, IMAGENET_STD)
from instancedepth.utils.checkpoint import load_checkpoint
from instancedepth.utils.metrics import depth_metrics

log = logging.getLogger("visualize_tb")

CV2_COLORMAPS = {
    "inferno": cv2.COLORMAP_INFERNO, "magma": cv2.COLORMAP_MAGMA,
    "turbo": getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET),
    "viridis": cv2.COLORMAP_VIRIDIS, "plasma": cv2.COLORMAP_PLASMA,
    "jet": cv2.COLORMAP_JET, "spectral": cv2.COLORMAP_JET,
}


# --------------------------------------------------------------------------- #
#  image helpers (all return HxWx3 uint8 RGB)
# --------------------------------------------------------------------------- #
def denorm_rgb(img: torch.Tensor) -> np.ndarray:
    """(3,H,W) ImageNet-normalized tensor -> (H,W,3) uint8 RGB."""
    x = img.detach().cpu().float().numpy().transpose(1, 2, 0)        # HWC
    x = x * np.array(IMAGENET_STD, np.float32) + np.array(IMAGENET_MEAN, np.float32)
    return (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)


def robust_range(gt: np.ndarray, min_d: float, max_d: float,
                 lo: float = 2.0, hi: float = 98.0) -> Tuple[float, float]:
    """Color range from VALID GT percentiles (consistent pred<->GT scale)."""
    d = gt[np.isfinite(gt) & (gt > min_d) & (gt <= max_d)]
    if d.size == 0:
        return min_d, max_d
    vmin = max(min_d, float(np.percentile(d, lo)))
    vmax = min(max_d, float(np.percentile(d, hi)))
    if vmax - vmin < 1e-3:
        vmin, vmax = min_d, max_d
    return vmin, vmax


def colorize(depth: np.ndarray, vmin: float, vmax: float, cmap: int,
             invalid: Optional[np.ndarray] = None) -> np.ndarray:
    """Metric depth (H,W) -> (H,W,3) uint8 RGB. Near = bright. Invalid -> black.

    NaN/Inf are scrubbed and values are clamped to [vmin,vmax] BEFORE colorizing,
    which is what removes the 'random colors' in unconstrained far regions.
    """
    rng = max(vmax - vmin, 1e-6)
    d = np.nan_to_num(depth.astype(np.float32), nan=vmax, posinf=vmax, neginf=vmin)
    norm = np.clip((vmax - d) / rng, 0.0, 1.0)                       # near=1=bright
    bgr = cv2.applyColorMap((norm * 255.0).astype(np.uint8), cmap)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if invalid is not None:
        rgb[invalid] = 0
    return rgb


def label(img: np.ndarray, text: str) -> np.ndarray:
    img = np.ascontiguousarray(img)
    cv2.rectangle(img, (0, 0), (12 + 11 * len(text), 30), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def overlay_masks(rgb: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """rgb (H,W,3) uint8; masks (M,H,W) bool -> instances tinted with a palette."""
    out = rgb.copy()
    rng = np.random.default_rng(0)
    for i in range(masks.shape[0]):
        color = rng.integers(60, 256, size=3)
        out[masks[i]] = (0.5 * out[masks[i]] + 0.5 * color).astype(np.uint8)
    return out


def row(images, labels) -> np.ndarray:
    """hstack equal-height labelled panels into one (H, sumW, 3) uint8 RGB image."""
    h = max(im.shape[0] for im in images)
    panels = []
    for im, txt in zip(images, labels):
        if im.shape[0] != h:
            scale = h / im.shape[0]
            im = cv2.resize(im, (int(im.shape[1] * scale), h), interpolation=cv2.INTER_NEAREST)
        panels.append(label(im, txt))
    return np.concatenate(panels, axis=1)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True, help="TensorBoard log dir")
    ap.add_argument("--phase", type=int, choices=(1, 2, 3), default=3)
    ap.add_argument("--split", default="test")
    ap.add_argument("--num-samples", type=int, default=16)
    ap.add_argument("--image-size", type=int, nargs=2, default=(504, 896), help="H W, /14")
    ap.add_argument("--depth-key", default="auto", choices=("auto", "refined_depth", "init_depth"),
                    help="auto: refined for phase 3, init otherwise")
    ap.add_argument("--colormap", default="inferno", choices=list(CV2_COLORMAPS))
    ap.add_argument("--show-masks", action="store_true", help="overlay instance masks on RGB")
    ap.add_argument("--mask-score", type=float, default=0.4, help="instance score threshold for overlay")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default="val", help="TensorBoard image tag prefix")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # ---- model ----
    model = build_instance_depth_from_yaml(args.model_config).to(args.device)
    info = load_checkpoint(args.checkpoint, model)
    log.info("loaded %s (missing=%d unexpected=%d)", args.checkpoint,
             len(info["missing"]), len(info["unexpected"]))
    model.set_phase(args.phase)
    model.eval()

    # ---- data ----
    cfg = GIDDatasetConfig(annotations_root=args.data_root, split=args.split,
                           image_size=tuple(args.image_size), hflip_prob=0.0)
    ds = GIDInstanceDepthDataset(cfg)
    if len(ds) == 0:
        raise SystemExit("dataset is empty")
    n = min(args.num_samples, len(ds))
    idxs = np.unique(np.linspace(0, len(ds) - 1, n).astype(int))     # spread across split
    log.info("visualizing %d sample(s) from '%s' (%d frames) -> %s",
             len(idxs), args.split, len(ds), args.out)

    run_instance = args.phase >= 2 or args.show_masks
    run_refine = args.phase == 3
    depth_key = ("refined_depth" if args.phase == 3 else "init_depth") \
        if args.depth_key == "auto" else args.depth_key
    cmap = CV2_COLORMAPS[args.colormap]
    writer = SummaryWriter(args.out)

    agg = {}
    for n_done, i in enumerate(idxs):
        s = ds[int(i)]
        rgb_t = s["image"].unsqueeze(0).to(args.device)
        gt = s["depth"][0].cpu().numpy().astype(np.float32)          # (H,W) metres, 0=invalid

        with torch.no_grad():
            out = model(rgb_t, run_instance=run_instance, run_refine=run_refine)
        pred = out[depth_key][0, 0].detach().cpu().numpy().astype(np.float32)   # (H,W)

        # one shared color range from valid GT; same for pred + GT
        vmin, vmax = robust_range(gt, cfg.min_depth, cfg.max_depth)
        gt_invalid = ~(np.isfinite(gt) & (gt > 0))

        rgb_img = denorm_rgb(s["image"])
        gt_col = colorize(gt, vmin, vmax, cmap, invalid=gt_invalid)            # black past sensor range
        pred_col = colorize(pred, vmin, vmax, cmap)                            # full model output, clamped
        pred_masked = colorize(pred, vmin, vmax, cmap, invalid=gt_invalid)     # clean: black where GT is

        panels = [rgb_img, gt_col, pred_col, pred_masked]
        names = ["RGB", f"GT [{vmin:.1f}-{vmax:.1f}m]", "Pred (clamped)", "Pred (masked to GT)"]

        if args.show_masks and "pred_masks" in out:
            ml = out["pred_masks"][0]                                          # (Q,Hf,Wf) logits
            cl = out["pred_logits"][0].softmax(-1)[:, :-1].max(-1).values      # (Q,)
            keep = cl > args.mask_score
            if keep.any():
                m = torch.nn.functional.interpolate(
                    ml[keep].unsqueeze(1), size=gt.shape, mode="bilinear",
                    align_corners=False).squeeze(1).sigmoid().cpu().numpy() > 0.5
                panels.insert(1, overlay_masks(rgb_img, m))
                names.insert(1, f"Instances ({int(keep.sum())})")

        composite = row(panels, names)                                        # (H, W, 3) RGB uint8
        writer.add_image(f"{args.tag}/sample_{n_done:02d}", composite, 0, dataformats="HWC")

        m = depth_metrics(torch.from_numpy(pred), torch.from_numpy(gt),
                          max_d=cfg.max_depth, min_d=cfg.min_depth)
        if m:
            for k in ("RMS", "REL", "sigma1"):
                writer.add_scalar(f"{args.tag}_metrics/{k}", m[k], n_done)
                agg.setdefault(k, []).append(m[k])

    if agg:
        log.info("mean over %d samples: %s", len(next(iter(agg.values()))),
                 " ".join(f"{k}={np.mean(v):.4f}" for k, v in agg.items()))
        for k, v in agg.items():
            writer.add_scalar(f"{args.tag}_metrics/mean_{k}", float(np.mean(v)), 0)

    writer.close()
    log.info("done. run:  tensorboard --logdir %s   (open the IMAGES tab)", args.out)


if __name__ == "__main__":
    main()
