#!/usr/bin/env python
"""check_gt_tracking.py - is the GT itself fragmenting / swapping people?

When the model splits ONE person into multiple instance masks, with one-to-one
Hungarian matching it can only have learned that from the LABELS. This checks the
labels directly: it colours each GT instance by its TRACK ID (so a person keeps
ONE colour across frames if tracking is stable) and quantifies tracking quality.

What to look for in the saved strips (consecutive frames of one sequence):
  * one person shows TWO colours in a single frame      -> ID SPLIT (fragmentation)
  * a person's colour FLICKERS frame to frame           -> ID SWITCH
  * stable, one colour per person across the strip      -> tracking is fine
A stable tracker assigns each real person a single ID that persists; lots of
short-lived IDs is the signature of fragmentation/switching.

Run:
  python check_gt_tracking.py --data-root gid_custom --split train \\
      --max-seqs 8 --strip-frames 6 --min-instances 2 --out-dir gt_track_diag
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

RGB_KEYS = ("rgb", "image", "rgb_path", "left_rgb", "rgb_filled")
PALETTE = np.array([
    (66, 135, 245), (245, 130, 48), (60, 180, 75), (240, 50, 230),
    (255, 225, 25), (70, 240, 240), (230, 25, 75), (145, 30, 180),
    (210, 245, 60), (250, 190, 212), (0, 128, 128), (170, 110, 40),
], np.uint8)


def _first(frame, keys):
    for k in keys:
        p = frame.get(k)
        if p and os.path.exists(p):
            return p
    return None


def load_rgb(frame, hw) -> Optional[np.ndarray]:
    p = _first(frame, RGB_KEYS)
    if p is None:
        return np.zeros((hw[0], hw[1], 3), np.uint8)
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        return np.zeros((hw[0], hw[1], 3), np.uint8)
    if img.shape[:2] != hw:
        img = cv2.resize(img, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
    return img


def load_id_map(frame, hw) -> Optional[np.ndarray]:
    p = frame.get("object_mask")
    if not p or not os.path.exists(p):
        return None
    m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.shape[:2] != hw:
        m = cv2.resize(m, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    return m


def color_for(tid: int) -> np.ndarray:
    return PALETTE[tid % len(PALETTE)]      # STABLE colour per track id


def render_gt(rgb, id_map, tids) -> np.ndarray:
    out = rgb.copy()
    for tid in tids:
        m = id_map == tid
        if m.sum() < 30:
            continue
        layer = np.zeros_like(out)
        layer[m] = color_for(int(tid))[::-1]            # BGR for cv2
        out = cv2.addWeighted(out, 1.0, layer, 0.5, 0.0)
        ys, xs = np.where(m)
        cv2.putText(out, str(int(tid)), (int(xs.mean()), int(ys.mean())),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="GT tracking-quality diagnostic")
    ap.add_argument("--data-root", default="gid_custom")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-seqs", type=int, default=8)
    ap.add_argument("--strip-frames", type=int, default=6)
    ap.add_argument("--min-instances", type=int, default=2,
                    help="only strip sequences whose busiest frame has >= this many")
    ap.add_argument("--transient-max", type=int, default=3,
                    help="a track id seen in fewer than this many frames is 'transient'")
    ap.add_argument("--out-dir", default="gt_track_diag")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    random.seed(args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    root = Path(args.data_root)
    seqs = [s for s in (root / f"{args.split}.txt").read_text().splitlines() if s.strip()]
    random.shuffle(seqs)

    per_frame_counts = []
    transient_frac_all = []
    ids_per_seq = []
    stripped = 0

    for sid in seqs:
        mpath = root / sid / "annotations.json"
        if not mpath.exists():
            continue
        man = json.load(open(mpath))
        fkeys = sorted(man["frames"])
        if not fkeys:
            continue

        # tracking stats over the whole sequence
        id_frames = defaultdict(int)               # track id -> #frames present
        busiest = 0
        for fk in fkeys:
            tids = [inst["track_id"] for inst in man["frames"][fk]["instances"]]
            per_frame_counts.append(len(tids))
            busiest = max(busiest, len(tids))
            for t in set(tids):
                id_frames[t] += 1
        n_ids = len(id_frames)
        ids_per_seq.append(n_ids)
        transient = sum(1 for t, c in id_frames.items() if c < args.transient_max)
        if n_ids:
            transient_frac_all.append(transient / n_ids)

        # render a strip of consecutive frames for a busy sequence
        if busiest >= args.min_instances and stripped < args.max_seqs:
            start = 0
            for i, fk in enumerate(fkeys):
                if len(man["frames"][fk]["instances"]) >= args.min_instances:
                    start = i
                    break
            chosen = fkeys[start:start + args.strip_frames]
            panels = []
            for fk in chosen:
                frame = man["frames"][fk]
                # load the id map first to get the frame shape, then the rgb
                p = frame.get("object_mask")
                if not p or not os.path.exists(p):
                    continue
                idm = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if idm is None:
                    continue
                hw = idm.shape[:2]
                rgb = load_rgb(frame, hw)
                tids = sorted({inst["track_id"] for inst in frame["instances"]})
                panels.append(render_gt(rgb, idm, tids))
            if panels:
                hmin = min(p.shape[0] for p in panels)
                panels = [cv2.resize(p, (int(p.shape[1] * hmin / p.shape[0]), hmin)) for p in panels]
                strip = np.concatenate(panels, axis=1)
                cv2.imwrite(str(out / f"seq_{stripped:02d}_{sid.replace('/', '_')}.png"), strip)
                stripped += 1

    pf = np.array(per_frame_counts)
    print("=== INSTANCES PER FRAME (GT) ===")
    if pf.size:
        print(f"  mean={pf.mean():.2f}  median={np.median(pf):.0f}  "
              f"p90={np.percentile(pf,90):.0f}  max={pf.max():.0f}")
        print(f"  (you said ~2.34 people/frame; if mean/p90 are much higher, GT has "
              f"extra instances = fragmentation)\n")

    print("=== TRACK-ID STABILITY ===")
    if transient_frac_all:
        tf = np.array(transient_frac_all)
        print(f"  distinct track ids per sequence : mean={np.mean(ids_per_seq):.1f}")
        print(f"  fraction of ids that are TRANSIENT (<{args.transient_max} frames): "
              f"mean={tf.mean():.2f}  max={tf.max():.2f}")
        print("  -> a high transient fraction = the tracker keeps creating/dropping ids")
        print("     (fragmentation / id switching), which teaches the model to split people.\n")

    print(f"[strips] {stripped} sequence strips saved -> {out}/")
    print("  Open them: each person should keep ONE colour (and id number) across the")
    print("  strip. Two colours on one body = split; flickering colour = id switch.")


if __name__ == "__main__":
    main()
