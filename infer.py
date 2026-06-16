"""Standalone inference for InstanceDepth (paper Sec. 4, full pipeline).

Runs RGB frames through the trained 3-phase model and writes the final
occlusion-rectified metric depth (`out["refined_depth"]`, Sec. 4.2.2). Accepts a
single image, a directory of frames, or a glob. Preprocessing matches the
training dataset exactly (ImageNet normalization; aspect-exact resize divisible
by 14, the ViT-L/14 requirement) so train/inference distributions agree.

Place this file at the repository root (next to train.py / evaluate.py) and run
from there so `import instancedepth` resolves.

Usage
-----
    python infer.py \
        --model-config instancedepth/configs/instance_depth.yaml \
        --checkpoint   runs/phase3/ckpt_final.pth \
        --input        path/to/frames_dir \
        --out-dir      predictions \
        --image-size   504 896 \
        [--save-color] [--save-masks] [--use-init-depth] [--batch-size 2]

Outputs (per frame, restored to the ORIGINAL input resolution, e.g. 720x1280):
    <out-dir>/depth_npy/<name>.npy     float32 metric depth in meters
    <out-dir>/depth_color/<name>.png   colorized depth                 (--save-color)
    <out-dir>/masks/<name>.png         uint16 instance id map (0=bg)   (--save-masks)
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from instancedepth.build import build_instance_depth
from instancedepth.utils.checkpoint import load_checkpoint

IMAGENET_MEAN = np.array((0.485, 0.456, 0.406), np.float32)
IMAGENET_STD = np.array((0.229, 0.224, 0.225), np.float32)
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# --------------------------------------------------------------------------- #
def list_inputs(inp: str) -> List[Path]:
    p = Path(inp)
    if p.is_dir():
        files = [q for q in sorted(p.iterdir()) if q.suffix.lower() in IMG_EXTS]
    elif any(ch in inp for ch in "*?["):
        files = [Path(q) for q in sorted(glob.glob(inp))]
    elif p.is_file():
        files = [p]
    else:
        raise FileNotFoundError(f"input not found: {inp}")
    if not files:
        raise FileNotFoundError(f"no images found at: {inp}")
    return files


def preprocess(bgr: np.ndarray, H: int, W: int) -> torch.Tensor:
    """BGR uint8 (Ho,Wo,3) -> normalized CHW float tensor at (H,W)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
    x = rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def colorize(depth: np.ndarray, min_d: float, max_d: float, invert: bool = True) -> np.ndarray:
    """Metric depth (H,W) -> BGR uint8 heatmap, fixed [min_d,max_d] scale.

    invert=True maps near->bright (the common depth-visualization convention);
    the fixed scale keeps colors comparable across frames of a video.
    """
    d = np.clip(depth, min_d, max_d)
    d = (d - min_d) / (max_d - min_d + 1e-8)
    if invert:
        d = 1.0 - d
    return cv2.applyColorMap((d * 255.0).astype(np.uint8), cv2.COLORMAP_INFERNO)


@torch.no_grad()
def to_idmap(
    pred_masks: torch.Tensor,   # (N, Hf, Wf) logits
    pred_logits: torch.Tensor,  # (N, K+1)
    pred_depth: torch.Tensor,   # (N,)
    out_hw: Tuple[int, int],
    cls_thresh: float,
    mask_thresh: float,
) -> np.ndarray:
    """Build a single-label uint16 instance id map (occluder wins the overlap).

    Filtering mirrors the paper's Sec. 4.2.2 confidence gating (cls>0.9,
    mask>0.8); overlaps resolved nearest-depth-wins, as in the data engine.
    """
    H, W = out_hw
    idmap = np.zeros((H, W), np.uint16)
    probs = pred_logits.softmax(-1)[:, :-1]                    # drop no-object
    cls_conf, _ = probs.max(dim=-1)
    mp = pred_masks.sigmoid()
    binm = mp >= 0.5
    area = binm.flatten(1).sum(-1).clamp_min(1)
    mask_conf = (mp * binm).flatten(1).sum(-1) / area
    keep = (cls_conf > cls_thresh) & (mask_conf > mask_thresh)
    idx = torch.nonzero(keep, as_tuple=False).squeeze(-1)
    if idx.numel() == 0:
        return idmap

    masks_up = F.interpolate(
        pred_masks[idx][:, None], size=(H, W), mode="bilinear", align_corners=False
    ).squeeze(1).sigmoid() >= 0.5                              # (M, H, W) bool
    order = torch.argsort(pred_depth[idx], descending=True)    # far first -> near overwrites
    masks_np = masks_up.cpu().numpy()
    for new_id, j in enumerate(order.tolist(), start=1):
        idmap[masks_np[j]] = new_id
    return idmap


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="InstanceDepth inference")
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--checkpoint", required=True, help="phase-3 ckpt_final.pth")
    ap.add_argument("--input", required=True, help="image file, directory, or glob")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--image-size", type=int, nargs=2, default=(504, 896),
                    help="H W; must be divisible by 14 and match training")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--use-init-depth", action="store_true",
                    help="output Stage-1 holistic depth instead of refined_depth")
    ap.add_argument("--save-color", action="store_true")
    ap.add_argument("--save-masks", action="store_true")
    ap.add_argument("--no-invert-color", action="store_true",
                    help="colorize far->bright instead of near->bright")
    ap.add_argument("--cls-thresh", type=float, default=0.9)
    ap.add_argument("--mask-thresh", type=float, default=0.8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    H, W = args.image_size
    assert H % 14 == 0 and W % 14 == 0, "image-size must be divisible by 14 (ViT-L/14)"

    out_dir = Path(args.out_dir)
    (out_dir / "depth_npy").mkdir(parents=True, exist_ok=True)
    if args.save_color:
        (out_dir / "depth_color").mkdir(parents=True, exist_ok=True)
    if args.save_masks:
        (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    # Build with the pretrained ViT download disabled — the checkpoint supplies
    # all weights, so we must not depend on network access at inference time.
    with open(args.model_config) as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict.setdefault("backbone", {})["pretrained"] = False
    max_depth = float(cfg_dict.get("max_depth", 10.0))

    model = build_instance_depth(cfg_dict).to(args.device).eval()
    info = load_checkpoint(args.checkpoint, model)
    print(f"loaded {args.checkpoint} (missing={len(info['missing'])} "
          f"unexpected={len(info['unexpected'])}, phase={info['phase']})")

    files = list_inputs(args.input)
    print(f"running on {len(files)} frame(s) at {H}x{W} -> {out_dir}")

    run_instance = (not args.use_init_depth) or args.save_masks
    run_refine = not args.use_init_depth

    with torch.inference_mode():
        for start in range(0, len(files), args.batch_size):
            chunk = files[start:start + args.batch_size]
            bgrs = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in chunk]
            if any(b is None for b in bgrs):
                bad = [str(p) for p, b in zip(chunk, bgrs) if b is None]
                raise IOError(f"failed to read: {bad}")
            orig_hw = [b.shape[:2] for b in bgrs]               # (Ho, Wo) per frame
            rgb = torch.stack([preprocess(b, H, W) for b in bgrs]).to(args.device)

            out = model(rgb, run_instance=run_instance, run_refine=run_refine)
            depth = out["init_depth"] if args.use_init_depth else out["refined_depth"]

            for i, p in enumerate(chunk):
                Ho, Wo = orig_hw[i]
                d = depth[i, 0].float().cpu().numpy()           # (H, W) meters
                d = cv2.resize(d, (Wo, Ho), interpolation=cv2.INTER_LINEAR)
                np.save(out_dir / "depth_npy" / f"{p.stem}.npy", d.astype(np.float32))

                if args.save_color:
                    cv2.imwrite(str(out_dir / "depth_color" / f"{p.stem}.png"),
                                colorize(d, 0.01, max_depth, invert=not args.no_invert_color))
                if args.save_masks:
                    idmap = to_idmap(
                        out["pred_masks"][i], out["pred_logits"][i],
                        out["pred_depth"][i].squeeze(-1), (Ho, Wo),
                        args.cls_thresh, args.mask_thresh,
                    )
                    cv2.imwrite(str(out_dir / "masks" / f"{p.stem}.png"), idmap)

            print(f"  {start + len(chunk)}/{len(files)} done")

    print(f"finished. depth maps in {out_dir/'depth_npy'} (float32 meters).")


if __name__ == "__main__":
    main()
