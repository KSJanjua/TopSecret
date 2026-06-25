"""Visualize Phase-2 instance predictions against GT, to diagnose a train/test gap.

When a model scores ~95% on train but ~0% on test, the picture tells you which
case you're in:

  * predictions look like correct person masks, but DON'T overlap the GT panel
        -> test GT is MISALIGNED / wrong convention (data-prep bug on the test
           split). Fix the annotations, not the model.
  * predictions look like correct masks AND overlap good-looking GT, yet the
        metric is ~0 -> an alignment/indexing bug in how test is loaded.
  * predictions are blobby / empty while GT looks fine
        -> the test IMAGES are out-of-distribution for the model (true shift or
           a test-time input problem).

Dumps RGB | GT | Pred panels as PNGs (uses cv2, no matplotlib). Run it on BOTH
--split test and --split train so you can compare a known-good case (train) with
the failing one (test).

    python viz_phase2_predictions.py --model-config instancedepth/configs/instance_depth.yaml \
        --data-root gid_custom --checkpoint runs/phase2_v4/ckpt_final.pth \
        --split test --num-frames 8 --out-dir viz_test
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from instancedepth.build import build_instance_depth_from_yaml
from instancedepth.data.gid_dataset import (GIDDatasetConfig,
                                            GIDInstanceDepthDataset)
from instancedepth.models.instance.inference import instance_segmentation
from instancedepth.utils.checkpoint import load_checkpoint

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

# distinct BGR colors for instances
_PALETTE = np.array([
    [66, 135, 245], [245, 130, 48], [60, 180, 75], [240, 50, 230],
    [255, 225, 25], [70, 240, 240], [230, 25, 75], [145, 30, 180],
    [128, 128, 0], [0, 128, 128], [170, 110, 40], [128, 0, 0],
], np.uint8)


def _denorm(img_t: torch.Tensor) -> np.ndarray:
    """(3,H,W) normalized tensor -> (H,W,3) BGR uint8."""
    img = img_t.detach().cpu().numpy().transpose(1, 2, 0)
    img = (img * IMAGENET_STD + IMAGENET_MEAN) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _overlay(base: np.ndarray, masks: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    out = base.copy()
    for i in range(masks.shape[0]):
        color = _PALETTE[i % len(_PALETTE)]
        m = masks[i].astype(bool)
        out[m] = (alpha * color + (1 - alpha) * out[m]).astype(np.uint8)
        # outline
        cont, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cont, -1, [int(c) for c in color], 2)
    return out


def _label(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return img


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize Phase-2 predictions vs GT")
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--image-size", type=int, nargs=2, default=(504, 896))
    ap.add_argument("--num-frames", type=int, default=8)
    ap.add_argument("--score-thresh", type=float, default=0.5)
    ap.add_argument("--use-nms", action="store_true",
                    help="route through inference.instance_segmentation (duplicate + "
                         "fragment NMS) -- i.e. show the DEPLOYED output, not raw queries.")
    ap.add_argument("--out-dir", default="viz")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.model_config) as f:
        max_depth = float(yaml.safe_load(f).get("max_depth", 10.0))

    model = build_instance_depth_from_yaml(args.model_config).to(args.device)
    load_checkpoint(args.checkpoint, model)
    model.eval()

    ds = GIDInstanceDepthDataset(GIDDatasetConfig(
        annotations_root=args.data_root, split=args.split,
        image_size=tuple(args.image_size), max_depth=max_depth,
        hflip_prob=0.0, require_valid_depth_layer=False))

    n = len(ds)
    stride = max(1, n // args.num_frames)
    picks = list(range(0, n, stride))[:args.num_frames]
    print(f"{args.split}: {n} frames, visualizing {picks}")

    with torch.inference_mode():
        for idx in picks:
            sample = ds[idx]
            rgb = sample["image"].unsqueeze(0).to(args.device)
            gt = sample["targets"]["masks"]                    # (G,H,W)
            H, W = rgb.shape[-2:]

            out = model(rgb, run_instance=True, run_refine=False)
            logits = out["pred_logits"][0]                     # (N,K+1)
            masks = F.interpolate(out["pred_masks"][0].unsqueeze(0), size=(H, W),
                                  mode="bilinear", align_corners=False)[0]
            if args.use_nms:                                   # deployed path (de-duplicated)
                insts = instance_segmentation(logits, masks, score_thresh=args.score_thresh)
                pred = (np.stack([i["mask"].cpu().numpy() for i in insts])
                        if insts else np.zeros((0, H, W), bool))
            else:                                              # raw per-query masks
                fg = logits.softmax(-1)[:, :-1].max(-1).values
                keep = fg >= args.score_thresh
                pred = (masks[keep].sigmoid() > 0.5).cpu().numpy() if keep.any() \
                    else np.zeros((0, H, W), bool)
            gt_b = (gt > 0.5).cpu().numpy() if gt.numel() else np.zeros((0, H, W), bool)

            base = _denorm(rgb[0])
            panel = np.hstack([
                _label(base, f"RGB  idx={idx}"),
                _label(_overlay(base, gt_b), f"GT  ({gt_b.shape[0]} inst)"),
                _label(_overlay(base, pred),
                       f"PRED {'NMS' if args.use_nms else 'raw'} "
                       f"({pred.shape[0]} @>{args.score_thresh})"),
            ])
            path = os.path.join(args.out_dir, f"{args.split}_{idx:06d}.png")
            cv2.imwrite(path, panel)

            # numeric: best IoU per GT vs kept preds
            best = [max((_iou(g, p) for p in pred), default=0.0) for g in gt_b]
            best_str = ", ".join(f"{b:.2f}" for b in best) if best else "-"
            print(f"  idx={idx:6d}  GT={gt_b.shape[0]}  pred={pred.shape[0]}  "
                  f"bestIoU/GT=[{best_str}]  -> {path}")

    print(f"\nWrote {len(picks)} panels to {args.out_dir}/  (scp them and look).")


if __name__ == "__main__":
    main()
