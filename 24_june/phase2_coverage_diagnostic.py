"""Phase-2 target-coverage diagnostic.

Root-cause check for "missing / merged people": the training dataset
(`gid_dataset.py`) drops any instance whose GT depth layer is invalid
(`require_valid_depth_layer=True`, and `_depth_layer` returns 0 when a mask has
no valid sensor depth). On a depth-camera dataset that can silently remove fully
visible people from supervision.

This script reproduces the dataset's exact per-instance filtering (resize ->
min-area-after-resize -> depth validity) WITHOUT training, and reports how many
instances each filter removes. Decisive: if `dropped_by_depth` is a large
fraction, fix the filter (keep mask/class targets, skip only the depth term)
before touching the model.

Reads the same files the dataset reads; no torch required.

    python phase2_coverage_diagnostic.py --data-root <out_root> \
        --split train --image-size 504 896 --max-depth 10.0 [--max-frames N]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def _load_id_map(frame: dict, ann_dir: Path) -> np.ndarray | None:
    path = frame.get("object_mask")
    candidates = []
    if path:
        candidates.append(Path(path))
        candidates.append(ann_dir / Path(path).name)               # moved-data fallback
    for c in candidates:
        if c.exists():
            m = cv2.imread(str(c), cv2.IMREAD_UNCHANGED)
            if m is not None:
                return m
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase-2 instance coverage diagnostic")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--image-size", type=int, nargs=2, default=(504, 896), help="H W")
    ap.add_argument("--max-depth", type=float, default=10.0)
    ap.add_argument("--min-instance-px", type=int, default=64, help="match GIDDatasetConfig")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all; else subsample by stride")
    args = ap.parse_args()
    H, W = args.image_size

    root = Path(args.data_root)
    seq_ids = [s for s in (root / f"{args.split}.txt").read_text().splitlines() if s.strip()]

    # build (annotations_dir, frame_key) index, then optionally stride-subsample
    index = []
    for sid in seq_ids:
        ann_path = root / sid / "annotations.json"
        if not ann_path.exists():
            print(f"WARN missing {ann_path}")
            continue
        man = json.loads(ann_path.read_text())
        for fkey in sorted(man["frames"]):
            index.append((ann_path.parent, man["frames"][fkey]))
    if args.max_frames and len(index) > args.max_frames:
        stride = max(1, len(index) // args.max_frames)
        index = index[::stride][:args.max_frames]
    print(f"scanning {len(index)} frames ({args.split})")

    n_raw = n_area = n_depth = 0
    frames_with_depth_drop = 0
    g_current = Counter()          # final instances per frame (current training G)
    g_if_no_depth_filter = Counter()
    dropped_depth_reason = Counter()   # why depth was invalid
    missing_masks = 0

    for k, (ann_dir, frame) in enumerate(index):
        id_map = _load_id_map(frame, ann_dir)
        if id_map is None:
            missing_masks += 1
            continue
        if id_map.ndim == 3:
            id_map = id_map[..., 0]
        id_map = cv2.resize(id_map, (W, H), interpolation=cv2.INTER_NEAREST)

        f_raw = f_area = f_depth = 0
        for inst in frame["instances"]:
            f_raw += 1
            area = int((id_map == inst["track_id"]).sum())
            if area < args.min_instance_px:
                continue
            f_area += 1
            layer = float(inst["depth_layer_m"])
            if 0.0 < layer <= args.max_depth:
                f_depth += 1
            else:
                dropped_depth_reason["zero/invalid" if layer <= 0 else "above_max_depth"] += 1

        n_raw += f_raw
        n_area += f_area
        n_depth += f_depth
        g_current[f_depth] += 1
        g_if_no_depth_filter[f_area] += 1
        if f_area > f_depth:
            frames_with_depth_drop += 1

    # ---- report ----
    def pct(a, b):
        return f"{(100.0 * a / b):.1f}%" if b else "n/a"

    print("\n================ COVERAGE ================")
    print(f"frames scanned                : {len(index)}  (missing object_mask: {missing_masks})")
    print(f"raw instances (annotated)     : {n_raw}")
    print(f"after min-area ({args.min_instance_px}px)        : {n_area}  "
          f"(dropped by area: {n_raw - n_area}, {pct(n_raw - n_area, n_raw)})")
    print(f"after depth filter (= train G): {n_depth}  "
          f"(dropped by DEPTH: {n_area - n_depth}, {pct(n_area - n_depth, n_area)} of area-valid)")
    print(f"frames losing >=1 to depth    : {frames_with_depth_drop} "
          f"({pct(frames_with_depth_drop, len(index))} of frames)")
    print(f"depth-drop reasons            : {dict(dropped_depth_reason)}")

    def hist_summary(c: Counter, label: str):
        tot = sum(c.values())
        mean = sum(g * n for g, n in c.items()) / max(tot, 1)
        empty = c.get(0, 0)
        print(f"  {label:28s} mean G={mean:.2f}  frames with 0 instances={empty} "
              f"({pct(empty, tot)})")

    print("\n--- instances-per-frame ---")
    hist_summary(g_current, "current (depth-filtered)")
    hist_summary(g_if_no_depth_filter, "if depth filter REMOVED")
    print("\nIf 'dropped by DEPTH' is large, set require_valid_depth_layer=False for")
    print("Phase 2 and let the loss skip only the per-instance depth term.")


if __name__ == "__main__":
    main()
