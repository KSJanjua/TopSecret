"""demo_test_sequence.py - render a coherent demo video from ONE test sequence.

Unlike viz_phase2_predictions.py (which samples frames across many videos), this
takes a single test sequence, runs the model on its frames IN ORDER, and writes
an mp4 with two panels per frame:

    [ RGB + predicted masks (NMS) | RGB + GT masks ]

So you judge the model on IN-DISTRIBUTION frames (the ones it scored ~90% on),
in temporal order, with PRED and GT labeled side-by-side -- no arbitrary clip,
no resolution/aspect mismatch, and no ambiguity about which panel is which.

Predicted masks use the SAME instance_segmentation (nms_iou=0.3, containment=0.6)
as deployment. Uses the FFmpeg raw-pipe writer (plays in VS Code).

    python demo_test_sequence.py --checkpoint runs/phase2_v4/ckpt_final.pth \
        --data-root gid_custom --output demo_v4.mp4
    # choose a specific sequence (and list what's available):
    python demo_test_sequence.py --checkpoint ... --data-root gid_custom --list-seqs
    python demo_test_sequence.py --checkpoint ... --data-root gid_custom --seq-id <name>
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from collections import OrderedDict
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from instancedepth.build import build_instance_depth
from instancedepth.data.gid_dataset import GIDDatasetConfig, GIDInstanceDepthDataset
from instancedepth.utils.checkpoint import load_checkpoint
from instancedepth.models.instance.inference import instance_segmentation

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

_PALETTE_RGB = [
    (255, 0, 0),    (0, 200, 0),    (0, 90, 255),   (255, 165, 0),  (180, 0, 255),
    (0, 220, 220),  (255, 0, 220),  (160, 230, 0),  (255, 215, 0),  (0, 160, 130),
    (255, 105, 180),(140, 110, 255),(0, 255, 130),  (255, 130, 70), (100, 150, 255),
    (210, 0, 100),  (70, 200, 255), (190, 255, 100),(255, 70, 130), (0, 190, 255),
    (230, 160, 0),  (150, 0, 200),  (0, 230, 180),  (255, 90, 0),
]
PALETTE = [(b, g, r) for (r, g, b) in _PALETTE_RGB]


def color_for(idx: int) -> Tuple[int, int, int]:
    if idx < len(PALETTE):
        return PALETTE[idx]
    h = int(((idx * 0.61803398875) % 1.0) * 179)
    bgr = cv2.cvtColor(np.uint8([[[h, 230, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


# --------------------------------------------------------------------------- #
#  lightweight greedy IoU tracker -> persistent ID (and thus color) per person
# --------------------------------------------------------------------------- #
def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


class Tracker:
    """Greedy frame-to-frame mask association. Each person keeps one ID (->color)
    for as long as it's matched; a track survives `max_age` unmatched frames so it
    re-acquires the SAME id after a brief occlusion."""

    def __init__(self, iou_thresh: float = 0.3, max_age: int = 10) -> None:
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.tracks: List[dict] = []          # {id, mask, age}
        self.next_id = 0

    def update(self, masks: List[np.ndarray]) -> List[int]:
        assigned: List[Optional[int]] = [None] * len(masks)
        pairs = []
        for mi, m in enumerate(masks):
            for ti, t in enumerate(self.tracks):
                iou = _iou(m, t["mask"])
                if iou >= self.iou_thresh:
                    pairs.append((iou, mi, ti))
        pairs.sort(key=lambda p: p[0], reverse=True)        # highest IoU first
        used_m, used_t = set(), set()
        for _iouv, mi, ti in pairs:
            if mi in used_m or ti in used_t:
                continue
            used_m.add(mi); used_t.add(ti)
            assigned[mi] = self.tracks[ti]["id"]
            self.tracks[ti]["mask"] = masks[mi]
            self.tracks[ti]["age"] = 0
        # age + drop stale tracks (only pre-existing ones)
        survivors = []
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t["age"] += 1
            if t["age"] <= self.max_age:
                survivors.append(t)
        self.tracks = survivors
        # spawn new tracks for unmatched detections
        for mi, m in enumerate(masks):
            if assigned[mi] is None:
                assigned[mi] = self.next_id
                self.tracks.append({"id": self.next_id, "mask": m, "age": 0})
                self.next_id += 1
        return [int(a) for a in assigned]


# --------------------------------------------------------------------------- #
def denorm(img_t: torch.Tensor) -> np.ndarray:
    """(3,H,W) normalized tensor -> (H,W,3) BGR uint8."""
    x = img_t.detach().cpu().numpy().transpose(1, 2, 0)
    x = (x * IMAGENET_STD + IMAGENET_MEAN) * 255.0
    x = np.clip(x, 0, 255).astype(np.uint8)
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


def banner(img, text):
    cv2.rectangle(img, (0, 0), (12 + 12 * len(text), 30), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2, cv2.LINE_AA)
    return img


def overlay(bgr, masks, ids=None, scores=None, alpha=0.5, outline=True):
    """Color each mask by its ID (persistent -> stable color). If ids is None,
    falls back to list index. Labels show '#id' (+ score if given)."""
    out = bgr.copy()
    for j, m in enumerate(masks):
        cid = ids[j] if ids is not None else j
        color = color_for(cid)
        layer = np.zeros_like(out)
        layer[m] = color
        out = cv2.addWeighted(out, 1.0, layer, alpha, 0.0)
        if outline:
            cont, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cont, -1, (0, 0, 0), 3)
            cv2.drawContours(out, cont, -1, color, 2)
        ys, xs = np.where(m)
        if len(xs):
            lab = f"#{cid}" + (f" {scores[j]:.2f}" if scores is not None else "")
            x0, y0 = int(xs.mean()) - 18, int(ys.mean())
            cv2.putText(out, lab, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, lab, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        color, 1, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------- #
class FFmpegWriter:
    def __init__(self, path, w, h, fps):
        self.w, self.h = w - (w % 2), h - (h % 2)
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
    print("[warn] ffmpeg not found; using OpenCV mp4v (may not play everywhere)")
    return Cv2Writer(path, w, h, fps)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Demo video from one test sequence (PRED vs GT)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--output", default="demo_sequence.mp4")
    ap.add_argument("--model-config", default="instancedepth/configs/instance_depth.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--image-size", type=int, nargs=2, default=(504, 896), help="H W")
    ap.add_argument("--seq-id", default=None, help="sequence name (default: the longest one)")
    ap.add_argument("--list-seqs", action="store_true", help="list sequences and exit")
    ap.add_argument("--score-thresh", type=float, default=0.5)
    ap.add_argument("--mask-thresh", type=float, default=0.5)
    ap.add_argument("--min-area", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--track-iou", type=float, default=0.3,
                    help="min IoU to link a detection to an existing track")
    ap.add_argument("--track-max-age", type=int, default=10,
                    help="frames a track survives unmatched (re-acquires same color "
                         "after a brief occlusion)")
    ap.add_argument("--pred-only", action="store_true", help="only the PRED panel (no GT)")
    ap.add_argument("--fps", type=float, default=15.0, help="output video fps")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = whole sequence")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    H, W = args.image_size

    ds = GIDInstanceDepthDataset(GIDDatasetConfig(
        annotations_root=args.data_root, split=args.split,
        image_size=tuple(args.image_size), hflip_prob=0.0,
        require_valid_depth_layer=False))

    # group dataset indices by sequence (already contiguous + frame-ordered)
    seqs: "OrderedDict[str, List[int]]" = OrderedDict()
    for i, (man, fk) in enumerate(ds.index):
        seqs.setdefault(man["sequence"], []).append(i)

    if args.list_seqs:
        print(f"{len(seqs)} sequences in '{args.split}':")
        for name, idxs in sorted(seqs.items(), key=lambda kv: -len(kv[1])):
            print(f"  {len(idxs):5d} frames   {name}")
        return

    if args.seq_id:
        if args.seq_id not in seqs:
            raise SystemExit(f"sequence '{args.seq_id}' not found; use --list-seqs")
        name = args.seq_id
    else:
        name = max(seqs, key=lambda k: len(seqs[k]))     # longest by default
    frame_ids = seqs[name][::max(args.stride, 1)]
    if args.max_frames:
        frame_ids = frame_ids[:args.max_frames]
    print(f"[seq] '{name}'  ({len(frame_ids)} frames) -> {args.output}")

    # ---- model ----
    with open(args.model_config) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("backbone", {})["pretrained"] = False
    model = build_instance_depth(cfg).to(args.device)
    info = load_checkpoint(args.checkpoint, model)
    print(f"loaded {args.checkpoint} (missing={len(info['missing'])} "
          f"unexpected={len(info['unexpected'])})")
    model.eval()

    panel_w = W if args.pred_only else 2 * W
    writer = make_writer(args.output, panel_w, H, args.fps)

    tracker = Tracker(iou_thresh=args.track_iou, max_age=args.track_max_age)
    done = 0
    with torch.inference_mode():
        for n, i in enumerate(frame_ids):
            sample = ds[i]
            x = sample["image"].unsqueeze(0).to(args.device)
            out = model(x, run_instance=True, run_refine=False)

            masks_up = F.interpolate(out["pred_masks"][0][None].float(), size=(H, W),
                                     mode="bilinear", align_corners=False)[0]
            insts = instance_segmentation(out["pred_logits"][0], masks_up,
                                          score_thresh=args.score_thresh,
                                          mask_thresh=args.mask_thresh, min_area=args.min_area)
            pred_masks = [d["mask"].cpu().numpy() for d in insts]
            pred_scores = [float(d["score"]) for d in insts]
            pred_ids = tracker.update(pred_masks)            # persistent color per person

            base = denorm(sample["image"])
            left = overlay(base, pred_masks, ids=pred_ids, scores=pred_scores, alpha=args.alpha)
            left = banner(left, f"PRED tracked ({len(pred_masks)})")

            if args.pred_only:
                canvas = left
            else:
                gt = sample["targets"]["masks"]
                gt_tids = sample["targets"]["track_ids"]
                if gt.numel():
                    gt_masks = [(gt[k] > 0.5).cpu().numpy() for k in range(gt.shape[0])]
                    gt_ids = [int(gt_tids[k]) for k in range(gt.shape[0])]   # true identity
                else:
                    gt_masks, gt_ids = [], []
                right = overlay(base, gt_masks, ids=gt_ids, scores=None, alpha=args.alpha)
                right = banner(right, f"GT ids ({len(gt_masks)})")
                canvas = np.concatenate([left, right], axis=1)

            writer.write(canvas)
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(frame_ids)} frames", flush=True)

    writer.close()
    print(f"[done] {done} frames -> {args.output}")


if __name__ == "__main__":
    main()
