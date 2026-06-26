from __future__ import annotations

import argparse
import shutil
import subprocess
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from instancedepth.build import build_instance_depth
from instancedepth.utils.checkpoint import load_checkpoint

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

# Distinct, vivid colors (RGB) picked for high mutual contrast and good
# visibility when alpha-blended over skin/clothing. Converted to BGR for cv2.
_PALETTE_RGB = [
    (255, 0, 0),    (0, 200, 0),    (0, 90, 255),   (255, 165, 0),  (180, 0, 255),
    (0, 220, 220),  (255, 0, 220),  (160, 230, 0),  (255, 215, 0),  (0, 160, 130),
    (255, 105, 180),(140, 110, 255),(0, 255, 130),  (255, 130, 70), (100, 150, 255),
    (210, 0, 100),  (70, 200, 255), (190, 255, 100),(255, 70, 130), (0, 190, 255),
    (230, 160, 0),  (150, 0, 200),  (0, 230, 180),  (255, 90, 0),
]
PALETTE = [(b, g, r) for (r, g, b) in _PALETTE_RGB]   # cv2 uses BGR


def color_for(idx: int) -> Tuple[int, int, int]:
    """Distinct BGR color for instance `idx`.

    Curated palette first; beyond it, golden-angle hue rotation keeps every extra
    instance a different bright, saturated color (crowds never reuse a color).
    """
    if idx < len(PALETTE):
        return PALETTE[idx]
    h = int(((idx * 0.61803398875) % 1.0) * 179)          # OpenCV hue range [0,179]
    bgr = cv2.cvtColor(np.uint8([[[h, 230, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


# --------------------------------------------------------------------------- #
#  instance resolution (embedded so this script is standalone)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def resolve_instances(pred_logits, pred_masks, pred_depth=None,
                      score_thresh=0.4, mask_thresh=0.5, min_area=100,
                      nms_iou=0.5) -> List[Dict]:
    """Non-overlapping, de-duplicated instances for ONE image.

    pred_logits (N,K+1), pred_masks (N,H,W) logits, pred_depth (N,1) optional.
    Returns [{mask(H,W) bool, score, label, depth}], sorted by score desc.
    """
    scores = pred_logits.softmax(-1)[:, :-1]
    cls_score, cls_id = scores.max(-1)
    keep = cls_score > score_thresh
    if keep.sum() == 0:
        return []
    idx = keep.nonzero(as_tuple=True)[0]
    cls_score, cls_id = cls_score[idx], cls_id[idx]
    mask_prob = pred_masks[idx].sigmoid()
    bin_mask = mask_prob > mask_thresh

    order = torch.argsort(cls_score, descending=True).tolist()
    kept = []
    for i in order:
        if int(bin_mask[i].sum()) < min_area:
            continue
        if any((bin_mask[i] & bin_mask[j]).sum().float()
               / (bin_mask[i] | bin_mask[j]).sum().float().clamp(min=1) > nms_iou
               for j in kept):
            continue
        kept.append(i)
    if not kept:
        return []

    # strict non-overlap: each pixel to the survivor with the highest mask prob
    winner = mask_prob[kept].argmax(0)
    results = []
    for s, i in enumerate(kept):
        m = (winner == s) & bin_mask[i]
        if int(m.sum()) < min_area:
            continue
        r = {"mask": m.cpu().numpy(), "score": float(cls_score[i]), "label": int(cls_id[i])}
        if pred_depth is not None:
            r["depth"] = float(pred_depth[idx[i], 0])
        results.append(r)
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


# --------------------------------------------------------------------------- #
#  drawing
# --------------------------------------------------------------------------- #
def banner(img, text):
    cv2.rectangle(img, (0, 0), (10 + 11 * len(text), 26), (0, 0, 0), -1)
    cv2.putText(img, text, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


def overlay_masks(bgr, insts, alpha=0.5, outline=True, show_score=True, show_depth=False):
    out = bgr.copy()
    for j, inst in enumerate(insts):
        m = inst["mask"]
        color = color_for(j)
        layer = np.zeros_like(out)
        layer[m] = color
        out = cv2.addWeighted(out, 1.0, layer, alpha, 0.0)
        if outline:
            cont, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cont, -1, (0, 0, 0), 3)     # dark halo: visible on any bg
            cv2.drawContours(out, cont, -1, color, 2)
        if show_score:
            ys, xs = np.where(m)
            if len(xs):
                lab = f"{inst['score']:.2f}"
                if show_depth and "depth" in inst:
                    lab += f" {inst['depth']:.1f}m"
                x0, y0 = int(xs.mean()) - 20, int(ys.mean())
                cv2.putText(out, lab, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 0, 0), 3, cv2.LINE_AA)          # dark outline
                cv2.putText(out, lab, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            color, 1, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------- #
#  video writers
# --------------------------------------------------------------------------- #
class FFmpegWriter:
    def __init__(self, path, w, h, fps):
        self.w, self.h = w - (w % 2), h - (h % 2)            # libx264 needs even dims
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{self.w}x{self.h}", "-r", f"{fps:.3f}", "-i", "-",
               "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", path]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame):
        self.p.stdin.write(np.ascontiguousarray(frame[:self.h, :self.w]).tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()


class Cv2Writer:
    def __init__(self, path, w, h, fps):
        self.vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def write(self, frame):
        self.vw.write(frame)

    def close(self):
        self.vw.release()


def make_writer(path, w, h, fps):
    if shutil.which("ffmpeg"):
        return FFmpegWriter(path, w, h, fps)
    print("[warn] ffmpeg not found; using OpenCV mp4v (may not play in all players)")
    return Cv2Writer(path, w, h, fps)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize a phase-2 checkpoint's MASKS on an mp4")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True, help="input .mp4")
    ap.add_argument("--output", default="p2_masks.mp4")
    ap.add_argument("--model-config", default="instancedepth/configs/instance_depth.yaml")
    ap.add_argument("--image-size", type=int, nargs=2, default=(504, 896), help="H W")
    ap.add_argument("--score-thresh", type=float, default=0.4)
    ap.add_argument("--mask-thresh", type=float, default=0.5)
    ap.add_argument("--min-area", type=int, default=100)
    ap.add_argument("--nms-iou", type=float, default=0.5,
                    help="suppress a mask overlapping a higher-scoring one by > this IoU "
                         "(try 0.3 if you still see duplicates)")
    ap.add_argument("--alpha", type=float, default=0.5, help="mask overlay opacity")
    ap.add_argument("--no-outline", action="store_true", help="don't draw mask contours")
    ap.add_argument("--no-score", action="store_true", help="don't print per-mask score")
    ap.add_argument("--show-depth", action="store_true",
                    help="also print each instance's predicted depth-layer (metres)")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    H, W = args.image_size

    # ---- model (backbone weights come from the checkpoint, not the file) ----
    with open(args.model_config) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("backbone", {})["pretrained"] = False
    max_depth = float(cfg.get("max_depth", 10.0))
    model = build_instance_depth(cfg).to(args.device)
    info = load_checkpoint(args.checkpoint, model)
    print(f"loaded {args.checkpoint} (missing={len(info['missing'])} "
          f"unexpected={len(info['unexpected'])})")
    model.eval()

    # ---- video IO ----
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"could not open {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out_fps = fps / max(args.stride, 1)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    writer = make_writer(args.output, W, H, out_fps)        # masks-only -> single panel (W wide)
    print(f"input {args.input}  {total} frames @ {fps:.1f}fps  ->  {args.output}")

    mean = torch.tensor(IMAGENET_MEAN, device=args.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=args.device).view(1, 3, 1, 1)

    fi = done = 0
    with torch.inference_mode():
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if fi % args.stride != 0:
                fi += 1
                continue
            fi += 1

            disp = cv2.resize(frame_bgr, (W, H), interpolation=cv2.INTER_LINEAR)   # BGR display
            rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(args.device)
            x = (x - mean) / std

            out = model(x, run_instance=True, run_refine=False)

            masks_up = F.interpolate(out["pred_masks"][0][None].float(), size=(H, W),
                                     mode="bilinear", align_corners=False)[0]
            pd = out.get("pred_depth")
            insts = resolve_instances(out["pred_logits"][0], masks_up,
                                      pred_depth=pd[0] if pd is not None else None,
                                      score_thresh=args.score_thresh,
                                      mask_thresh=args.mask_thresh,
                                      min_area=args.min_area, nms_iou=args.nms_iou)

            panel = overlay_masks(disp, insts, alpha=args.alpha,
                                  outline=not args.no_outline,
                                  show_score=not args.no_score, show_depth=args.show_depth)
            panel = banner(panel, f"masks ({len(insts)})")
            writer.write(panel)

            done += 1
            if done % 50 == 0:
                print(f"  {done} frames written", flush=True)
            if args.max_frames and done >= args.max_frames:
                break

    cap.release()
    writer.close()
    print(f"[done] {done} frames -> {args.output}")


if __name__ == "__main__":
    main()
