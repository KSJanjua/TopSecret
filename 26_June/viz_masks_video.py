#!/usr/bin/env python
"""viz_masks_video.py - render ONLY the predicted instance masks on a video.

Unlike infer_video.py (RGB | depth side-by-side), this writes a single full-frame
panel with the de-duplicated person masks overlaid. No depth is computed, so it
works with a Phase-2 checkpoint and does NOT touch the (possibly stale) Phase-3
refinement head.

Masks come from the SAME resolver as infer_video.py / tb_progress.py
(instance_segmentation: IoU-NMS + fragment containment, defaults 0.3 / 0.6), so
what you see here matches the deployed output.

Example
-------
  python viz_masks_video.py \
      --model-config instancedepth/configs/instance_depth.yaml \
      --checkpoint runs/phase2_v4/ckpt_final.pth \
      --input my_clip.mp4 --output my_clip_masks.mp4 --height 504

  # quick preview: every 2nd frame, first 200 frames, higher score threshold
  python viz_masks_video.py --checkpoint runs/phase2_v4/ckpt_final.pth \
      --input clip.mp4 --stride 2 --max-frames 200 --score-thresh 0.6
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from instancedepth.build import build_instance_depth
from instancedepth.utils.checkpoint import load_checkpoint
from instancedepth.models.instance.inference import instance_segmentation

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PATCH = 14

# Distinct, vivid colors (RGB) picked for high mutual contrast and good
# visibility when alpha-blended over skin/clothing. Converted to BGR for cv2.
_PALETTE_RGB = [
    (255, 0, 0),    (0, 200, 0),    (0, 90, 255),   (255, 165, 0),  (180, 0, 255),
    (0, 220, 220),  (255, 0, 220),  (160, 230, 0),  (255, 215, 0),  (0, 160, 130),
    (255, 105, 180),(140, 110, 255),(0, 255, 130),  (255, 130, 70), (100, 150, 255),
    (210, 0, 100),  (70, 200, 255), (190, 255, 100),(255, 70, 130), (0, 190, 255),
    (230, 160, 0),  (150, 0, 200),  (0, 230, 180),  (255, 90, 0),
]
_PALETTE = [(b, g, r) for (r, g, b) in _PALETTE_RGB]   # cv2 uses BGR


def color_for(idx: int) -> Tuple[int, int, int]:
    """Distinct BGR color for instance `idx`.

    Uses the curated palette first; beyond it, golden-angle hue rotation keeps
    every additional instance a different bright, saturated color (so even large
    crowds never reuse a color on adjacent people).
    """
    if idx < len(_PALETTE):
        return _PALETTE[idx]
    h = int(((idx * 0.61803398875) % 1.0) * 179)          # OpenCV hue range [0,179]
    bgr = cv2.cvtColor(np.uint8([[[h, 230, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


# --------------------------------------------------------------------------- #
#  preprocessing (aspect-preserving, multiple-of-14, ImageNet-normalized)
# --------------------------------------------------------------------------- #
def round_to_patch(x: int, patch: int = PATCH) -> int:
    return max(patch, int(round(x / patch)) * patch)


def target_size(orig_h: int, orig_w: int, height: int) -> Tuple[int, int]:
    h = round_to_patch(height)
    w = round_to_patch(int(round(h * orig_w / orig_h)))
    return h, w


def preprocess(frame_bgr: np.ndarray, size_hw: Tuple[int, int], device: str) -> torch.Tensor:
    h, w = size_hw
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(rgb).float().div_(255.0).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    t = (t - mean) / std
    return t.unsqueeze(0).to(device)


# --------------------------------------------------------------------------- #
#  mask extraction + overlay
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_instances(out: dict, orig_hw: Tuple[int, int], score_thresh: float,
                      mask_thresh: float, min_area: int) -> List[Tuple[np.ndarray, float]]:
    if "pred_masks" not in out or "pred_logits" not in out:
        return []
    h, w = orig_hw
    logits = out["pred_logits"][0]
    masks = F.interpolate(out["pred_masks"][0][None].float(), size=(h, w),
                          mode="bilinear", align_corners=False)[0]
    insts = instance_segmentation(logits, masks, score_thresh=score_thresh,
                                  mask_thresh=mask_thresh, min_area=min_area)
    return [(d["mask"].cpu().numpy(), float(d["score"])) for d in insts]


def overlay_masks(frame_bgr: np.ndarray, dets: List[Tuple[np.ndarray, float]],
                  alpha: float, outline: bool, show_score: bool) -> np.ndarray:
    out = frame_bgr.copy()
    for idx, (mask, score) in enumerate(dets):
        color = color_for(idx)
        layer = np.zeros_like(out)
        layer[mask] = color
        out = cv2.addWeighted(out, 1.0, layer, alpha, 0.0)
        if outline:
            cont, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cont, -1, (0, 0, 0), 3)     # dark halo: pops on any bg
            cv2.drawContours(out, cont, -1, color, 2)
        if show_score:
            ys, xs = np.where(mask)
            if xs.size:
                x0, y0 = int(xs.min()), max(int(ys.min()) - 6, 16)
                cv2.putText(out, f"{score:.2f}", (x0, y0), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 0, 0), 3, cv2.LINE_AA)          # dark outline
                cv2.putText(out, f"{score:.2f}", (x0, y0), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, color, 1, cv2.LINE_AA)
    return out


def label(img: np.ndarray, text: str) -> np.ndarray:
    cv2.rectangle(img, (0, 0), (10 + 12 * len(text), 34), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return img


# --------------------------------------------------------------------------- #
#  robust video sink (OpenCV codec, else PNG frames + ffmpeg) -- from infer_video
# --------------------------------------------------------------------------- #
class _Cv2Sink:
    def __init__(self, wr, path):
        self.wr, self.out_path = wr, path
    def write(self, frame):
        self.wr.write(frame)
    def close(self):
        self.wr.release()
        return self.out_path


class _FramesSink:
    def __init__(self, out_path, fps):
        self.out_path = Path(out_path).with_suffix(".mp4")
        self.dir = self.out_path.with_suffix("")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fps, self.i = fps, 0
        print(f"[writer] no OpenCV codec; writing PNG frames to {self.dir}/ + ffmpeg")
    def write(self, frame):
        cv2.imwrite(str(self.dir / f"f_{self.i:06d}.png"), frame)
        self.i += 1
    def close(self):
        ff = shutil.which("ffmpeg")
        if ff is None:
            print(f"[writer] ffmpeg not found; PNG frames left in {self.dir}/")
            return self.dir
        cmd = [ff, "-y", "-framerate", f"{self.fps}", "-i", str(self.dir / "f_%06d.png"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               str(self.out_path)]
        print("[writer] encoding:", " ".join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and self.out_path.exists():
            shutil.rmtree(self.dir, ignore_errors=True)
            return self.out_path
        print("[writer] ffmpeg failed:\n", r.stderr[-1500:])
        return self.dir


def open_sink(out_path, fps, w, h):
    for cc, ext in [("mp4v", ".mp4"), ("MJPG", ".avi"), ("XVID", ".avi")]:
        p = Path(out_path).with_suffix(ext)
        wr = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*cc), fps, (w, h))
        if wr.isOpened():
            print(f"[writer] codec={cc} -> {p}")
            return _Cv2Sink(wr, p)
        wr.release()
    return _FramesSink(out_path, fps)


# --------------------------------------------------------------------------- #
#  main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Render predicted instance masks on a video")
    ap.add_argument("--model-config", default="instancedepth/configs/instance_depth.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", "--video", dest="input", required=True)
    ap.add_argument("--output", default=None, help="default: <input>_masks.mp4")
    ap.add_argument("--height", type=int, default=504,
                    help="model input height (multiple of 14); width follows aspect ratio")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--score-thresh", type=float, default=0.5)
    ap.add_argument("--mask-thresh", type=float, default=0.5)
    ap.add_argument("--min-area", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.5, help="mask overlay opacity")
    ap.add_argument("--no-outline", action="store_true", help="don't draw mask contours")
    ap.add_argument("--no-score", action="store_true", help="don't print per-mask score")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"input video not found: {in_path}")
    out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_masks.mp4")

    with open(args.model_config) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("backbone", {})["pretrained"] = False     # checkpoint carries the backbone
    model = build_instance_depth(cfg).to(args.device)
    info = load_checkpoint(args.checkpoint, model)
    print(f"[load] {args.checkpoint}  (missing={len(info['missing'])} "
          f"unexpected={len(info['unexpected'])})")
    model.eval()

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {in_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    in_hw = target_size(orig_h, orig_w, args.height)
    out_fps = src_fps / max(args.stride, 1)
    print(f"[video] {orig_w}x{orig_h} @ {src_fps:.1f}fps, {n_total} frames "
          f"-> model input {in_hw[1]}x{in_hw[0]}  (masks only, no depth)")

    sink = None
    fi = processed = 0
    t0 = time.time()
    with torch.inference_mode():
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if fi % max(args.stride, 1) != 0:
                fi += 1
                continue
            x = preprocess(frame, in_hw, args.device)
            out = model(x, run_instance=True, run_refine=False)   # masks only
            dets = extract_instances(out, (orig_h, orig_w), args.score_thresh,
                                     args.mask_thresh, args.min_area)
            canvas = overlay_masks(frame, dets, args.alpha,
                                   outline=not args.no_outline,
                                   show_score=not args.no_score)
            canvas = label(canvas, f"{len(dets)} ppl")
            if sink is None:
                h, w = canvas.shape[:2]
                sink = open_sink(out_path, out_fps, w, h)
            sink.write(canvas)
            processed += 1
            fi += 1
            if processed % 25 == 0:
                print(f"  {processed} frames  ({processed / (time.time() - t0):.1f} fps)",
                      flush=True)
            if args.max_frames and processed >= args.max_frames:
                break

    cap.release()
    if sink is not None:
        out_path = sink.close()
    dt = time.time() - t0
    print(f"[done] {processed} frames in {dt:.1f}s "
          f"({processed / max(dt, 1e-6):.1f} fps) -> {out_path}")


if __name__ == "__main__":
    main()
