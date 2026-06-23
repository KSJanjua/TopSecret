"""Set-prediction losses for Instance Depth Layer Prediction (Sec. 4.2.1), with
deep supervision and point-sampled masks.

After Hungarian matching, per decoder layer:
    L_m : mask BCE on boundary-biased sampled points  (Mask2Former / PointRend)
    L_d : dice on the same points
    L_c : classification CE over all N queries (matched -> GT label, else no-object)
    L_depth : depth-layer smoothed-L1 on matched instances (Eq. 7) -- FINAL layer only

WHAT CHANGED vs. the previous version
-------------------------------------
The mask and dice losses are now computed on POINTS sampled with importance
sampling (most-uncertain points near the boundary) rather than on every pixel.
Full-pixel averaging let the easy background dominate the gradient and left masks
fuzzy / incomplete, which is what allowed fragment slots to grab unclaimed body
parts. Point sampling sharpens the masks (and is far lighter, so deep supervision
across 9 layers stays cheap). Deep supervision and the invalid-depth guard are
unchanged.

[Paper Specified]    L_m mask loss, L_c cross-entropy, L_d smoothed L1 (Eqs. 5-7).
[Strongly Inferred]  no-object weight 0.1; deep supervision on every decoder layer;
                     aux layers supervise mask+class; point sampling (Mask2Former).

Placed at: instancedepth/losses/instance_losses.py
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.instance.matcher import HungarianMatcher
from ..models.instance.point_sampling import point_sample, uncertain_point_coords


def dice_loss(pred: torch.Tensor, tgt: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """pred (M, P) logits, tgt (M, P) -> scalar mean dice loss."""
    pred = pred.sigmoid()
    num = 2 * (pred * tgt).sum(-1)
    den = pred.sum(-1) + tgt.sum(-1)
    return (1 - (num + eps) / (den + eps)).mean()


class InstanceSetCriterion(nn.Module):
    def __init__(
        self,
        num_classes: int,
        w_mask: float = 5.0,
        w_dice: float = 5.0,
        w_class: float = 2.0,
        w_depth: float = 1.0,
        no_object_weight: float = 0.1,
        depth_beta: float = 1.0,
        num_points: int = 12544,
        matcher: Optional[HungarianMatcher] = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.w_mask = w_mask
        self.w_dice = w_dice
        self.w_class = w_class
        self.w_depth = w_depth
        self.depth_beta = depth_beta
        self.num_points = num_points
        self.matcher = matcher if matcher is not None else HungarianMatcher(
            w_mask=w_mask, w_dice=w_dice, w_class=w_class, w_depth=w_depth, num_points=num_points)
        weight = torch.ones(num_classes + 1)
        weight[-1] = no_object_weight                    # last index = no-object
        self.register_buffer("class_weight", weight)

    # --------------------------------------------------------- one layer's loss
    def _layer_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        with_depth: bool,
    ) -> Dict[str, torch.Tensor]:
        b, n = outputs["pred_logits"].shape[:2]
        device = outputs["pred_logits"].device
        no_obj = self.num_classes

        # classification over all queries (matched -> GT label, else no-object)
        target_classes = torch.full((b, n), no_obj, dtype=torch.long, device=device)
        for i, (pi, gi) in enumerate(indices):
            if pi.numel():
                target_classes[i, pi] = targets[i]["labels"][gi].to(device)
        loss_class = F.cross_entropy(
            outputs["pred_logits"].transpose(1, 2), target_classes,
            weight=self.class_weight.to(device))

        # gather matched masks (+ depth on the final layer)
        ml, mt, pdl, tdl = [], [], [], []
        for i, (pi, gi) in enumerate(indices):
            if pi.numel() == 0:
                continue
            ml.append(outputs["pred_masks"][i][pi])                     # (M, Hf, Wf)
            mt.append(targets[i]["masks"][gi].to(device))               # (M, Hg, Wg)
            if with_depth and "pred_depth" in outputs:
                pdl.append(outputs["pred_depth"][i][pi].squeeze(-1))
                tdl.append(targets[i]["depths"][gi].to(device))

        if ml:
            pm = torch.cat(ml, 0)[:, None]                              # (M, 1, Hf, Wf)
            tm = torch.cat(mt, 0)[:, None].to(pm.dtype)                 # (M, 1, Hg, Wg)
            # boundary-biased points chosen from the PREDICTION, sampled in both maps
            with torch.no_grad():
                pts = uncertain_point_coords(pm, self.num_points)       # (M, P, 2)
                tgt_pts = point_sample(tm, pts)[:, 0]                   # (M, P) in [0,1]
            pred_pts = point_sample(pm, pts)[:, 0]                      # (M, P) logits
            loss_mask = F.binary_cross_entropy_with_logits(pred_pts, tgt_pts)
            loss_dice = dice_loss(pred_pts, tgt_pts)
            if pdl:
                pd = torch.cat(pdl, 0)
                td = torch.cat(tdl, 0)
                # only supervise finite, valid (>0) depth layers (invalid GT makes
                # smooth_l1 backward NaN). Keep the graph connected if empty.
                valid = torch.isfinite(td) & (td > 0) & torch.isfinite(pd)
                loss_depth = (F.smooth_l1_loss(pd[valid], td[valid], beta=self.depth_beta)
                              if valid.any() else pd.sum() * 0.0)
            else:
                loss_depth = pm.sum() * 0.0
        else:
            z = outputs["pred_masks"].sum() * 0.0
            loss_mask = loss_dice = loss_depth = z

        return {"loss_class": loss_class, "loss_mask": loss_mask,
                "loss_dice": loss_dice, "loss_depth": loss_depth}

    # ----------------------------------------------------------------- forward
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        if indices is None:
            indices = self.matcher(outputs, targets)

        main = self._layer_loss(outputs, targets, indices, with_depth=True)
        total = (self.w_class * main["loss_class"] + self.w_mask * main["loss_mask"]
                 + self.w_dice * main["loss_dice"] + self.w_depth * main["loss_depth"])

        # deep supervision: one matched loss per auxiliary (per-layer) prediction
        for aux in outputs.get("aux_outputs", []):
            aux_idx = self.matcher(aux, targets)
            la = self._layer_loss(aux, targets, aux_idx, with_depth=False)
            total = total + (self.w_class * la["loss_class"] + self.w_mask * la["loss_mask"]
                             + self.w_dice * la["loss_dice"])

        out = dict(main)                 # expose main components for logging
        out["loss_total"] = total
        return out
