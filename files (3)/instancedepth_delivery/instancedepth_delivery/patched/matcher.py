"""Hungarian bipartite matcher (Eqs. 5-7).

Computes the optimal one-to-one assignment between the N predictions and the GT
instances by minimizing:
    cost = lambda1 * L_m(mask) + lambda2 * L_c(class) + lambda3 * L_d(depth)   (Eqs.5-7)

Faithfulness notes
-------------------
[Paper Specified]  Matching cost incorporates depth-layer difference, mask term,
                   and category classification; L_d is smoothed L1 (Eq. 7).
[Strongly Inferred] Following Mask2Former [12]/DETR [6], the class cost is the
                   negative predicted prob of the GT class, and the mask cost is
                   focal/BCE + dice evaluated on a sampled set of points. For
                   reproducibility we use full-mask BCE + dice cost (no point
                   sampling), which is exact though heavier.

Inputs / shapes
---------------
    pred_logits  (B, N, K+1)
    pred_masks   (B, N, Hf, Wf)
    pred_depth   (B, N, 1)
    targets[b]   dict: "labels" (G,), "masks" (G, Hf, Wf), "depths" (G,)
Returns: list of (index_pred, index_gt) LongTensors, one per batch element.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def _dice_cost(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """pred (N, P) sigmoid probs, tgt (G, P) -> (N, G) dice cost."""
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
    def __init__(self, w_mask: float = 5.0, w_dice: float = 5.0, w_class: float = 2.0, w_depth: float = 1.0) -> None:
        self.w_mask = w_mask
        self.w_dice = w_dice
        self.w_class = w_class
        self.w_depth = w_depth

    @torch.no_grad()
    def __call__(
        self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        b, n = outputs["pred_logits"].shape[:2]
        indices: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for i in range(b):
            tgt = targets[i]
            g = tgt["labels"].numel()
            if g == 0:
                indices.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
                continue

            prob = outputs["pred_logits"][i].softmax(-1)            # (N, K+1)
            cost_class = -prob[:, tgt["labels"]]                     # (N, G)

            pred_m = outputs["pred_masks"][i]                        # (N, Hf, Wf)
            tgt_m = tgt["masks"].to(pred_m.dtype)                    # (G, Hg, Wg)
            if tgt_m.shape[-2:] != pred_m.shape[-2:]:
                # GT masks arrive at image resolution; the cost is computed at
                # the prediction's mask resolution (Mask2Former computes both
                # on a common point set — nearest resize is the dense analog).
                tgt_m = F.interpolate(
                    tgt_m.unsqueeze(1), size=pred_m.shape[-2:], mode="nearest"
                ).squeeze(1)
            pm = pred_m.flatten(1)                                   # (N, P)
            tm = tgt_m.flatten(1)                                    # (G, P)
            cost_mask = _bce_cost(pm, tm) / pm.shape[1]
            cost_dice = _dice_cost(pm, tm)

            pd = outputs["pred_depth"][i].squeeze(-1)                # (N,)
            td = tgt["depths"].to(pd.dtype)                          # (G,)
            cost_depth = torch.cdist(pd[:, None], td[:, None], p=1)  # (N, G) |.|

            cost = (
                self.w_mask * cost_mask
                + self.w_dice * cost_dice
                + self.w_class * cost_class
                + self.w_depth * cost_depth
            )
            cost = torch.nan_to_num(cost, nan=1e4, posinf=1e4, neginf=-1e4).cpu()
            row, col = linear_sum_assignment(cost)
            indices.append(
                (torch.as_tensor(row, dtype=torch.long), torch.as_tensor(col, dtype=torch.long))
            )
        return indices
