"""Set-prediction losses for Instance Depth Layer Prediction (Sec. 4.2.1) WITH
deep supervision.

After Hungarian matching, supervise per decoder layer:
    L_m : mask BCE + dice on matched masks (Mask2Former)
    L_c : classification CE over all N queries (matched -> GT label, else no-object)
    L_d : depth-layer smoothed-L1 on matched instances (Eq. 7) -- FINAL layer only

WHAT CHANGED vs. the previous version
-------------------------------------
The criterion now also processes `outputs["aux_outputs"]` (one prediction per
decoder layer). Each aux layer is matched independently and contributes mask +
dice + class loss (the standard DETR/Mask2Former deep-supervision scheme). The
depth-layer loss is applied only on the main (final) output, since the depth
layer is computed only there. The criterion owns a default matcher for the aux
layers; the MAIN matching can be passed in (so phase 3 can reuse the same
indices for the refinement targets).

[Paper Specified]    L_m mask loss, L_c cross-entropy, L_d smoothed L1 (Eqs. 5-7).
[Strongly Inferred]  no-object weight 0.1 (DETR/Mask2Former); deep supervision on
                     every decoder layer; aux layers supervise mask+class.

Returns a dict of scalar loss tensors (main components, for logging) + the
weighted total over all layers.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.instance.matcher import HungarianMatcher


def dice_loss(pred: torch.Tensor, tgt: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """pred (M,*) logits, tgt (M,*) -> scalar mean dice loss."""
    pred = pred.sigmoid().flatten(1)
    tgt = tgt.flatten(1)
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
        matcher: Optional[HungarianMatcher] = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.w_mask = w_mask
        self.w_dice = w_dice
        self.w_class = w_class
        self.w_depth = w_depth
        self.depth_beta = depth_beta
        self.matcher = matcher if matcher is not None else HungarianMatcher(
            w_mask=w_mask, w_dice=w_dice, w_class=w_class, w_depth=w_depth)
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

        # mask + depth on matched pairs
        ml, mt, pdl, tdl = [], [], [], []
        for i, (pi, gi) in enumerate(indices):
            if pi.numel() == 0:
                continue
            pm = outputs["pred_masks"][i][pi]                           # (M,Hf,Wf)
            tm = targets[i]["masks"][gi].to(pm.dtype).to(device)
            if tm.shape[-2:] != pm.shape[-2:]:
                tm = F.interpolate(tm.unsqueeze(1), size=pm.shape[-2:], mode="nearest").squeeze(1)
            ml.append(pm)
            mt.append(tm)
            if with_depth and "pred_depth" in outputs:
                pdl.append(outputs["pred_depth"][i][pi].squeeze(-1))
                tdl.append(targets[i]["depths"][gi].to(device))

        if ml:
            pm = torch.cat(ml, 0)
            tm = torch.cat(mt, 0)
            loss_mask = F.binary_cross_entropy_with_logits(pm, tm)
            loss_dice = dice_loss(pm, tm)
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
