"""Resolve the instance head's per-query predictions into clean, NON-OVERLAPPING
instance masks (Mask2Former-style inference).

The head emits N independent per-query binary masks. Two queries can fire on the
same object (duplicate masks), and masks may overlap (one person's mask covering
part of a neighbor). This resolves both:

  * each pixel is assigned to the single highest-scoring query
    (score = class confidence x mask probability), so masks cannot overlap;
  * a query that ends up mostly subsumed by a stronger one is dropped, so
    duplicates disappear.

Use this wherever final instance masks are consumed (visualization, video
inference). It does NOT change training; it is pure post-processing.
"""

from __future__ import annotations

from typing import Dict, List

import torch


@torch.no_grad()
def instance_segmentation(
    pred_logits: torch.Tensor,        # (N, K+1) class logits, last class = no-object
    pred_masks: torch.Tensor,         # (N, H, W) mask logits
    score_thresh: float = 0.5,
    mask_thresh: float = 0.5,
    min_area: int = 100,
    overlap_thresh: float = 0.8,
) -> List[Dict[str, object]]:
    """Return non-overlapping, de-duplicated instances for ONE image.

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

    # per-pixel winner = query with the highest (class score * mask probability)
    weighted = mask_prob * cls_score[:, None, None]       # (M, H, W)
    winner = weighted.argmax(0)                           # (H, W)

    results: List[Dict[str, object]] = []
    for k in range(mask_prob.shape[0]):
        own = mask_prob[k] > mask_thresh                  # this query's own confident region
        kept = (winner == k) & own                        # the part it actually wins
        area = int(kept.sum())
        orig = int(own.sum())
        if orig == 0 or area < min_area:
            continue
        if area / orig < overlap_thresh:                  # mostly taken by a stronger query -> duplicate
            continue
        results.append({"mask": kept, "score": float(cls_score[k]), "label": int(cls_id[k])})

    results.sort(key=lambda r: r["score"], reverse=True)
    return results
