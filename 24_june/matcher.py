# from __future__ import annotations

# from typing import Dict, List, Tuple

# import torch
# import torch.nn.functional as F
# from scipy.optimize import linear_sum_assignment


# def _dice_cost(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
#     """pred (N, P) sigmoid probs, tgt (G, P) -> (N, G) dice cost."""
#     pred = pred.sigmoid()
#     num = 2 * pred @ tgt.t()
#     den = pred.sum(-1)[:, None] + tgt.sum(-1)[None, :]
#     return 1 - (num + 1) / (den + 1)


# def _bce_cost(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
#     """pred (N, P) logits, tgt (G, P) -> (N, G) BCE cost."""
#     pos = F.binary_cross_entropy_with_logits(pred, torch.ones_like(pred), reduction="none")
#     neg = F.binary_cross_entropy_with_logits(pred, torch.zeros_like(pred), reduction="none")
#     return pos @ tgt.t() + neg @ (1 - tgt).t()


# class HungarianMatcher:
#     def __init__(self, w_mask: float = 5.0, w_dice: float = 5.0, w_class: float = 2.0, w_depth: float = 1.0) -> None:
#         self.w_mask = w_mask
#         self.w_dice = w_dice
#         self.w_class = w_class
#         self.w_depth = w_depth

#     @torch.no_grad()
#     def __call__(
#         self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]
#     ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
#         b, n = outputs["pred_logits"].shape[:2]
#         indices: List[Tuple[torch.Tensor, torch.Tensor]] = []

#         for i in range(b):
#             tgt = targets[i]
#             g = tgt["labels"].numel()
#             if g == 0:
#                 indices.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
#                 continue

#             prob = outputs["pred_logits"][i].softmax(-1)            # (N, K+1)
#             cost_class = -prob[:, tgt["labels"]]                     # (N, G)

#             pred_m = outputs["pred_masks"][i]                        # (N, Hf, Wf)
#             tgt_m = tgt["masks"].to(pred_m.dtype)                    # (G, Hg, Wg)
#             if tgt_m.shape[-2:] != pred_m.shape[-2:]:
#                 # GT masks arrive at image resolution; the cost is computed at
#                 # the prediction's mask resolution (Mask2Former computes both
#                 # on a common point set — nearest resize is the dense analog).
#                 tgt_m = F.interpolate(
#                     tgt_m.unsqueeze(1), size=pred_m.shape[-2:], mode="nearest"
#                 ).squeeze(1)
#             pm = pred_m.flatten(1)                                   # (N, P)
#             tm = tgt_m.flatten(1)                                    # (G, P)
#             cost_mask = _bce_cost(pm, tm) / pm.shape[1]
#             cost_dice = _dice_cost(pm, tm)

#             pd = outputs["pred_depth"][i].squeeze(-1)                # (N,)
#             td = tgt["depths"].to(pd.dtype)                          # (G,)
#             cost_depth = torch.cdist(pd[:, None], td[:, None], p=1)  # (N, G) |.|

#             cost = (
#                 self.w_mask * cost_mask
#                 + self.w_dice * cost_dice
#                 + self.w_class * cost_class
#                 + self.w_depth * cost_depth
#             )
#             cost = torch.nan_to_num(cost, nan=1e4, posinf=1e4, neginf=-1e4).cpu()
#             row, col = linear_sum_assignment(cost)
#             indices.append(
#                 (torch.as_tensor(row, dtype=torch.long), torch.as_tensor(col, dtype=torch.long))
#             )
#         return indices


from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

# Mask2Former-style point rendering: matching/loss are computed on a small set of
# sampled points instead of the full dense mask. This is the single most important
# ingredient for sharp, complete masks and STABLE query<->GT assignment (without it,
# dense BCE is dominated by easy background, matching flickers, and several queries
# each capture a *fragment* of one person). See Cheng et al., Mask2Former [12].
MATCH_NUM_POINTS = 12544        # 112 * 112


def point_sample(inp: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Sample `inp` (M,C,H,W) at normalized `coords` (M,P,2) in [0,1] -> (M,C,P)."""
    c = coords
    if c.dim() == 3:
        c = c.unsqueeze(2)                                   # (M,P,1,2)
    out = F.grid_sample(inp, 2.0 * c - 1.0, mode="bilinear",
                        padding_mode="border", align_corners=False)
    return out.squeeze(3)                                    # (M,C,P)


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
                 num_points: int = MATCH_NUM_POINTS) -> None:
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

            pred_m = outputs["pred_masks"][i]                        # (N, Hf, Wf)
            tgt_m = tgt["masks"].to(pred_m.dtype)                    # (G, Hg, Wg)
            # Sample ONE shared set of random points (normalized coords) for every
            # prediction and GT in this image, then compute the cost on the points.
            # Normalized sampling is resolution-agnostic -> no GT resize needed and
            # adjacent people are no longer averaged together by dense background.
            pts = torch.rand(1, self.num_points, 2, device=pred_m.device, dtype=pred_m.dtype)
            pm = point_sample(pred_m.unsqueeze(1), pts.expand(n, -1, -1)).squeeze(1)  # (N, P)
            tm = point_sample(tgt_m.unsqueeze(1), pts.expand(g, -1, -1)).squeeze(1)   # (G, P)
            cost_mask = _bce_cost(pm, tm) / pm.shape[1]
            cost_dice = _dice_cost(pm, tm)

            cost = self.w_mask * cost_mask + self.w_dice * cost_dice + self.w_class * cost_class

            if pred_depth is not None:                               # final layer only
                pd = pred_depth[i].squeeze(-1)                       # (N,)
                td = tgt["depths"].to(pd.dtype)                      # (G,)
                cost_depth = torch.cdist(pd[:, None], td[:, None], p=1)
                cost = cost + self.w_depth * cost_depth

            cost = torch.nan_to_num(cost, nan=1e4, posinf=1e4, neginf=-1e4).cpu()
            row, col = linear_sum_assignment(cost)
            indices.append(
                (torch.as_tensor(row, dtype=torch.long), torch.as_tensor(col, dtype=torch.long)))
        return indices
