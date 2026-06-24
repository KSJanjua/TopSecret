"""Instance-mask quality metrics for Phase 2 (Instance Depth Layer Prediction).

Phase 2's whole job is producing instance masks, yet `evaluate.py` only reports
DEPTH metrics, so the reported failures (incomplete / merged / missing masks) are
invisible. This module adds the missing signal.

Design:
  * The metric CORE is pure numpy/scipy (no torch) so it is unit-testable and can
    be imported anywhere cheaply. Masks are passed as boolean arrays (P|G, H, W).
  * The torch glue (sigmoid, resize pred->GT resolution, class softmax) lives in
    `main()` (full eval) and is imported lazily, so importing the metric helpers
    for training-time logging does NOT require torch.

Metrics reported:
  AP / AP50 / AP75    COCO-style single-class mask AP, global greedy matching.
  recall@0.5/0.75     fraction of GT instances recovered (>=IoU) by a kept pred.
                      -> low recall == "missing / merged people".
  count_mae           mean |#pred - #GT| after score threshold
                      -> systematic under/over-segmentation.
  mIoU_matched        mean IoU of Hungarian-matched (pred,GT) pairs
                      -> "incomplete masks / missing body parts".
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

DEFAULT_IOU_THRS: Tuple[float, ...] = tuple(round(x, 2) for x in np.arange(0.5, 1.0, 0.05))


# --------------------------------------------------------------------------- #
# pure-numpy core
# --------------------------------------------------------------------------- #
def mask_iou_matrix(pred_bool: np.ndarray, gt_bool: np.ndarray) -> np.ndarray:
    """(P,H,W) bool, (G,H,W) bool -> (P,G) IoU. Flattened matmul, no P*G*H*W tensor."""
    p, g = pred_bool.shape[0], gt_bool.shape[0]
    if p == 0 or g == 0:
        return np.zeros((p, g), dtype=np.float64)
    pf = pred_bool.reshape(p, -1).astype(np.float64)
    gf = gt_bool.reshape(g, -1).astype(np.float64)
    inter = pf @ gf.T                                   # (P,G)
    area_p = pf.sum(1)[:, None]
    area_g = gf.sum(1)[None, :]
    union = area_p + area_g - inter
    return inter / np.maximum(union, 1e-9)


def mean_matched_iou(pred_bool: np.ndarray, gt_bool: np.ndarray) -> float:
    """Hungarian-match pred<->GT by IoU; return mean IoU over matched pairs (0 if none).

    Cheap live training signal. Pred and GT must already share H,W (caller resizes).
    """
    if pred_bool.shape[0] == 0 or gt_bool.shape[0] == 0:
        return 0.0
    iou = mask_iou_matrix(pred_bool, gt_bool)
    r, c = linear_sum_assignment(-iou)
    if len(r) == 0:
        return 0.0
    return float(iou[r, c].mean())


def _ap_from_pr(recall: np.ndarray, precision: np.ndarray) -> float:
    """COCO 101-point interpolated AP. `recall` must be non-decreasing (score order)."""
    if recall.size == 0:
        return 0.0
    prec_env = np.maximum.accumulate(precision[::-1])[::-1]    # monotone precision envelope
    rthr = np.linspace(0.0, 1.0, 101)
    out = np.zeros(101)
    idx = np.searchsorted(recall, rthr, side="left")
    valid = idx < prec_env.size
    out[valid] = prec_env[idx[valid]]
    return float(out.mean())


# A per-image "record" = (scores (P,), iou (P,G), n_gt). Masks are NOT stored, only
# the small IoU matrix, so AP can be computed globally with negligible memory.
Record = Tuple[np.ndarray, np.ndarray, int]


def compute_mask_ap(records: List[Record],
                    iou_thrs: Sequence[float] = DEFAULT_IOU_THRS) -> Dict[str, float]:
    """Global, COCO-style single-class mask AP via greedy matching in score order."""
    total_gt = int(sum(r[2] for r in records))
    if total_gt == 0:
        return {"AP": float("nan"), "AP50": float("nan"), "AP75": float("nan")}

    # global prediction list -> (owning image, pred index within image)
    owner: List[Tuple[int, int]] = []
    scores_all: List[float] = []
    for ii, (scores, _iou, _n) in enumerate(records):
        for pi in range(scores.shape[0]):
            owner.append((ii, pi))
            scores_all.append(float(scores[pi]))
    if not scores_all:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0}
    order = np.argsort(-np.asarray(scores_all), kind="mergesort")    # stable, high->low

    ap_at: Dict[float, float] = {}
    for t in iou_thrs:
        gt_taken = [np.zeros(r[2], dtype=bool) for r in records]
        tp = np.zeros(order.size)
        fp = np.zeros(order.size)
        for k, gi in enumerate(order):
            ii, pi = owner[gi]
            _scores, iou, n_gt = records[ii]
            if n_gt == 0:
                fp[k] = 1.0
                continue
            row = iou[pi]                               # (G,)
            avail = ~gt_taken[ii]
            cand = np.where(avail & (row >= t))[0]
            if cand.size:
                g = cand[np.argmax(row[cand])]
                gt_taken[ii][g] = True
                tp[k] = 1.0
            else:
                fp[k] = 1.0
        tpc = np.cumsum(tp)
        fpc = np.cumsum(fp)
        recall = tpc / total_gt
        precision = tpc / np.maximum(tpc + fpc, 1e-9)
        ap_at[round(t, 2)] = _ap_from_pr(recall, precision)

    return {
        "AP": float(np.mean(list(ap_at.values()))),
        "AP50": float(ap_at.get(0.5, float("nan"))),
        "AP75": float(ap_at.get(0.75, float("nan"))),
    }


def aggregate_recall(records: List[Record], score_thresh: float = 0.5,
                     iou_thrs: Sequence[float] = (0.5, 0.75)) -> Dict[str, float]:
    """Recall@IoU and count error among predictions kept at `score_thresh`."""
    total_gt = int(sum(r[2] for r in records))
    res: Dict[str, float] = {f"recall@{t}": 0.0 for t in iou_thrs}
    if total_gt == 0:
        res["count_mae"] = 0.0
        return res
    hits = {t: 0 for t in iou_thrs}
    count_err = 0.0
    n_img = 0
    for scores, iou, n_gt in records:
        keep = scores >= score_thresh
        count_err += abs(int(keep.sum()) - n_gt)
        n_img += 1
        if n_gt == 0:
            continue
        best = iou[keep].max(0) if keep.any() else np.zeros(n_gt)
        for t in iou_thrs:
            hits[t] += int((best >= t).sum())
    for t in iou_thrs:
        res[f"recall@{t}"] = hits[t] / total_gt
    res["count_mae"] = count_err / max(n_img, 1)
    return res


# --------------------------------------------------------------------------- #
# full evaluation entrypoint (torch imported lazily; not needed for the core)
# --------------------------------------------------------------------------- #
def main() -> None:  # pragma: no cover - requires torch + data + checkpoint
    import argparse
    import json

    import torch
    import torch.nn.functional as F
    import yaml
    from torch.utils.data import DataLoader

    from instancedepth.build import build_instance_depth_from_yaml
    from instancedepth.data.gid_dataset import (
        GIDDatasetConfig, GIDInstanceDepthDataset, collate_gid)
    from instancedepth.utils.checkpoint import load_checkpoint

    ap = argparse.ArgumentParser(description="Phase-2 instance-mask evaluation")
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--image-size", type=int, nargs=2, default=(504, 896))
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--score-thresh", type=float, default=0.5,
                    help="threshold for recall/count metrics only (AP uses all preds)")
    ap.add_argument("--ap-score-floor", type=float, default=0.05,
                    help="discard preds below this score before AP (memory/speed)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    with open(args.model_config) as f:
        max_depth = float(yaml.safe_load(f).get("max_depth", 10.0))

    model = build_instance_depth_from_yaml(args.model_config).to(args.device)
    load_checkpoint(args.checkpoint, model)
    model.eval()

    ds = GIDInstanceDepthDataset(GIDDatasetConfig(
        annotations_root=args.data_root, split=args.split,
        image_size=tuple(args.image_size), max_depth=max_depth, hflip_prob=0.0,
        require_valid_depth_layer=False))   # eval mask quality on ALL instances
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_gid)

    records: List[Record] = []
    with torch.inference_mode():
        for batch in loader:
            rgb = batch["image"].to(args.device)
            out = model(rgb, run_instance=True, run_refine=False)
            for b in range(rgb.shape[0]):
                gt = batch["targets"][b]["masks"]          # (G,H,W) {0,1}
                n_gt = int(gt.shape[0])
                H, W = gt.shape[-2:] if n_gt else tuple(args.image_size)

                logits = out["pred_logits"][b]             # (N,K+1)
                masks = out["pred_masks"][b].unsqueeze(0)  # (1,N,h,w)
                masks = F.interpolate(masks, size=(H, W), mode="bilinear",
                                      align_corners=False).squeeze(0)   # (N,H,W)
                fg = logits.softmax(-1)[:, :-1].max(-1).values          # (N,) drop no-object
                keep = fg >= args.ap_score_floor
                if keep.sum() == 0 or n_gt == 0:
                    records.append((fg[keep].cpu().numpy(),
                                    np.zeros((int(keep.sum()), n_gt)), n_gt))
                    continue
                pred_bool = (masks[keep].sigmoid() > 0.5).cpu().numpy()
                gt_bool = (gt > 0.5).cpu().numpy()
                iou = mask_iou_matrix(pred_bool, gt_bool)
                records.append((fg[keep].cpu().numpy(), iou, n_gt))

    result = {}
    result.update(compute_mask_ap(records))
    result.update(aggregate_recall(records, score_thresh=args.score_thresh))
    result["num_frames"] = len(records)
    result["num_gt_instances"] = int(sum(r[2] for r in records))
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in result.items()}, indent=1))
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
