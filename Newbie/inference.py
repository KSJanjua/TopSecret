"""Resolve the instance head's per-query predictions into clean instance masks.

The head emits N independent per-query masks. Two failure modes when consumed
naively: duplicate masks on one object, and masks overlapping a neighbour. This
resolves them while PRIORITIZING RECALL (never silently delete a real person):

  1. score filter: drop low-confidence queries (score_thresh).
  2. IoU NMS: drop a query only if it overlaps a higher-scoring query a LOT
     (mask IoU > nms_iou). A true duplicate has near-identical mask -> high IoU
     and is removed; two side-by-side people touch only at the border -> low IoU
     and are BOTH kept. This is the key difference from an area-retention rule,
     which wrongly deletes an adjacent person whose mask is partly overlapped.
  3. by default, each surviving instance keeps its OWN full mask (no erosion, so
     nobody is reduced to "a few body parts"). Set resolve_overlaps=True for
     strictly non-overlapping masks (per-pixel argmax by MASK probability -- not
     class score, so a confident neighbour cannot steal pixels).

Pure post-processing; does NOT change training. Use for visualization / video.
"""

from __future__ import annotations

from typing import Dict, List

import torch


@torch.no_grad()
def instance_segmentation(
    pred_logits: torch.Tensor,        # (N, K+1) class logits, last class = no-object
    pred_masks: torch.Tensor,         # (N, H, W) mask logits
    score_thresh: float = 0.4,
    mask_thresh: float = 0.5,
    min_area: int = 100,
    nms_iou: float = 0.5,
    resolve_overlaps: bool = False,
) -> List[Dict[str, object]]:
    """Return de-duplicated instances for ONE image.

    Each result: {"mask": (H, W) bool, "score": float, "label": int},
    sorted by score (descending).
    """
    scores = pred_logits.softmax(-1)[:, :-1]              # drop no-object -> (N, K)
    cls_score, cls_id = scores.max(-1)                    # (N,), (N,)
    keep = cls_score > score_thresh
    if keep.sum() == 0:
        return []
    cls_score = cls_score[keep]
    cls_id = cls_id[keep]
    mask_prob = pred_masks[keep].sigmoid()                # (M, H, W)
    bin_mask = mask_prob > mask_thresh                    # (M, H, W)

    # ---- IoU NMS: keep highest-scoring, drop only near-identical duplicates ----
    order = torch.argsort(cls_score, descending=True).tolist()
    kept: List[int] = []
    for i in order:
        if int(bin_mask[i].sum()) < min_area:
            continue
        duplicate = False
        for j in kept:
            inter = (bin_mask[i] & bin_mask[j]).sum().float()
            union = (bin_mask[i] | bin_mask[j]).sum().float().clamp(min=1)
            if (inter / union) > nms_iou:                 # high overlap -> same object
                duplicate = True
                break
        if not duplicate:
            kept.append(i)
    if not kept:
        return []

    # ---- final masks ----
    if resolve_overlaps:
        # each pixel to the survivor with the highest MASK probability there
        surv_prob = mask_prob[kept]                       # (S, H, W)
        winner = surv_prob.argmax(0)                      # (H, W)
        final = [(winner == s) & bin_mask[kept[s]] for s in range(len(kept))]
    else:
        final = [bin_mask[i] for i in kept]               # full own masks (recall-first)

    results: List[Dict[str, object]] = []
    for s, i in enumerate(kept):
        m = final[s]
        if int(m.sum()) < min_area:
            continue
        results.append({"mask": m, "score": float(cls_score[i]), "label": int(cls_id[i])})
    results.sort(key=lambda r: r["score"], reverse=True)
    return results
