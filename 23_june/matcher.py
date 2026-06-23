"""Hungarian bipartite matcher (Eqs. 5-7), point-sampled.

Optimal one-to-one assignment between the N predictions and the GT instances by
minimizing  cost = w_mask*L_mask + w_dice*L_dice + w_class*L_class + w_depth*L_depth.

WHAT CHANGED vs. the previous version
-------------------------------------
The mask/dice costs are now computed on a shared set of UNIFORM RANDOM POINTS
(Mask2Former) instead of every pixel. This matches the point-sampled training
loss, is far cheaper (so the per-layer deep-supervision matching is light), and
because points are sampled with normalized coords, predictions and GT masks are
compared correctly even at different resolutions (no nearest-resize).

Still TOLERATES outputs without `pred_depth`: auxiliary decoder layers carry only
masks + classes, so the depth term is dropped when `pred_depth` is absent.

[Paper Specified]  cost uses depth-layer difference, mask term, category; L_d is
                   smoothed L1 (Eq. 7).
[Strongly Inferred] class cost = -prob[GT class]; mask cost = BCE + dice on sampled
                   points (Mask2Former).

Placed at: instancedepth/models/instance/matcher.py
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .point_sampling import point_sample


def _dice_cost(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """pred (N, P) logits, tgt (G, P) -> (N, G) dice cost."""
    pred = pred.sigmoid()
    num = 2 * pred @ tgt.t()
    den = pred.sum(-1)[:, None] + tgt.sum(-1)[None, :]
    return 1 - (num + 1) / (den + 1)


def _bce_cost(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """pred (N, P) logits, tgt (G, P) -> (N, G) BCE cost."""
    pos = F.binary_cross_entropy_with_logits(pred, torch.ones_like(pred), reduction="none")
    neg = F.binary_cross_entropy_with_logits(pred, torch.zeros_like(pred), reduction="none")
    return pos @ tgt.t() + neg @ (1 - tgt).t()


class HungarianMatcher:
    def __init__(self, w_mask: float = 5.0, w_dice: float = 5.0,
                 w_class: float = 2.0, w_depth: float = 1.0,
                 num_points: int = 12544) -> None:
        self.w_mask = w_mask
        self.w_dice = w_dice
        self.w_class = w_class
        self.w_depth = w_depth
        self.num_points = num_points

    @torch.no_grad()
    def __call__(
        self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        b, n = outputs["pred_logits"].shape[:2]
        pred_depth = outputs.get("pred_depth")              # None for aux layers
        indices: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for i in range(b):
            tgt = targets[i]
            g = tgt["labels"].numel()
            if g == 0:
                indices.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
                continue

            prob = outputs["pred_logits"][i].softmax(-1)            # (N, K+1)
            cost_class = -prob[:, tgt["labels"]]                     # (N, G)

            pred_m = outputs["pred_masks"][i][:, None]               # (N, 1, Hf, Wf)
            tgt_m = tgt["masks"][:, None].to(pred_m.dtype)           # (G, 1, Hg, Wg)
            # one shared point set per image, sampled in both maps' normalized space
            pts = torch.rand(self.num_points, 2, device=pred_m.device)
            pp = point_sample(pred_m, pts[None].expand(pred_m.shape[0], -1, -1))[:, 0]  # (N, P)
            tp = point_sample(tgt_m, pts[None].expand(tgt_m.shape[0], -1, -1))[:, 0]    # (G, P)
            cost_mask = _bce_cost(pp, tp) / pp.shape[1]
            cost_dice = _dice_cost(pp, tp)

            cost = self.w_mask * cost_mask + self.w_dice * cost_dice + self.w_class * cost_class

            if pred_depth is not None:                               # final layer only
                pd = pred_depth[i].squeeze(-1)                       # (N,)
                td = tgt["depths"].to(pd.dtype)                      # (G,)
                cost = cost + self.w_depth * torch.cdist(pd[:, None], td[:, None], p=1)

            cost = torch.nan_to_num(cost, nan=1e4, posinf=1e4, neginf=-1e4).cpu()
            row, col = linear_sum_assignment(cost)
            indices.append(
                (torch.as_tensor(row, dtype=torch.long), torch.as_tensor(col, dtype=torch.long)))
        return indices
