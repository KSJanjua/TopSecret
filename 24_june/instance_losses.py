# from __future__ import annotations

# from typing import Dict, List, Tuple

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# def dice_loss(pred: torch.Tensor,tgt:torch.Tensor,eps:float=1.0) -> torch.Tensor:
#     """pred (M,P) logits, tgt (M,P) -> scalar mean dice loss"""

#     pred=pred.sigmoid().flatten(1)
#     tgt=tgt.flatten(1)
#     num=2*(pred*tgt).sum(-1)
#     den=pred.sum(-1)+tgt.sum(-1)
#     return (1-(num+eps)/(den+eps)).mean()

# class InstanceSetCriterion(nn.Module):
#     def __init__(
#             self,
#             num_classes: int,
#             w_mask: float=5.0,
#             w_dice: float=5.0,
#             w_class: float=2.0,
#             w_depth: float=1.0,
#             no_object_weight: float=0.1,
#             depth_beta: float=1.0,
#     ) -> None: 
#         super().__init__()
#         self.num_classes=num_classes
#         self.w_mask=w_mask
#         self.w_dice=w_dice
#         self.w_class=w_class
#         self.w_depth=w_depth
#         self.no_object_weight=no_object_weight
#         self.depth_beta=depth_beta
#         weight=torch.ones(num_classes+1)
#         weight[-1]=no_object_weight             #last index=no_object
#         self.register_buffer("class_weight",weight)
    
#     def forward(
#         self,
#         outputs: Dict[str, torch.Tensor],
#         targets: List[Dict[str, torch.Tensor]],
#         indices: List[Tuple[torch.Tensor, torch.Tensor]],
#     ) -> Dict[str, torch.Tensor]:
#         b, n = outputs["pred_logits"].shape[:2]
#         device = outputs["pred_logits"].device
#         no_obj = self.num_classes                    # no-object class index

#         # ---- classification loss over all queries ----
#         target_classes = torch.full((b, n), no_obj, dtype=torch.long, device=device)
#         for i, (pi, gi) in enumerate(indices):
#             if pi.numel():
#                 target_classes[i, pi] = targets[i]["labels"][gi].to(device)
#         loss_class = F.cross_entropy(
#             outputs["pred_logits"].transpose(1, 2), target_classes,
#             weight=self.class_weight.to(device))

#         # ---- mask + depth losses on matched pairs only ----
#         mask_logits_all, mask_tgt_all, pred_d_all, tgt_d_all = [], [], [], []
#         for i, (pi, gi) in enumerate(indices):
#             if pi.numel() == 0:
#                 continue
#             pm = outputs["pred_masks"][i][pi]                          # (M, Hf, Wf)
#             tm = targets[i]["masks"][gi].to(pm.dtype).to(device)
#             if tm.shape[-2:] != pm.shape[-2:]:
#                 tm = F.interpolate(tm.unsqueeze(1), size=pm.shape[-2:], mode="nearest").squeeze(1)
#             mask_logits_all.append(pm)
#             mask_tgt_all.append(tm)
#             pred_d_all.append(outputs["pred_depth"][i][pi].squeeze(-1))   # (M,)
#             tgt_d_all.append(targets[i]["depths"][gi].to(device))         # (M,)

#         if mask_logits_all:
#             pm = torch.cat(mask_logits_all, 0)
#             tm = torch.cat(mask_tgt_all, 0)
#             loss_mask = F.binary_cross_entropy_with_logits(pm, tm)
#             loss_dice = dice_loss(pm, tm)
#             pd = torch.cat(pred_d_all, 0)
#             td = torch.cat(tgt_d_all, 0)
#             # Only supervise finite, valid (>0) depth layers. A single invalid GT
#             # depth makes smooth_l1's BACKWARD NaN even when its forward looks fine
#             # -> this is the phase-2 NaN source. Keep the graph connected if empty.
#             valid = torch.isfinite(td) & (td > 0) & torch.isfinite(pd)
#             if valid.any():
#                 loss_depth = F.smooth_l1_loss(pd[valid], td[valid], beta=self.depth_beta)
#             else:
#                 loss_depth = pd.sum() * 0.0
#         else:
#             z = outputs["pred_masks"].sum() * 0.0
#             loss_mask = loss_dice = loss_depth = z

#         total = (self.w_class * loss_class + self.w_mask * loss_mask
#                  + self.w_dice * loss_dice + self.w_depth * loss_depth)
#         return {"loss_class": loss_class, "loss_mask": loss_mask,
#                 "loss_dice": loss_dice, "loss_depth": loss_depth, "loss_total": total}


from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.instance.matcher import HungarianMatcher, point_sample

LOSS_NUM_POINTS = 12544         # 112 * 112


def dice_loss(pred: torch.Tensor, tgt: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """pred (M,*) logits, tgt (M,*) -> scalar mean dice loss."""
    pred = pred.sigmoid().flatten(1)
    tgt = tgt.flatten(1)
    num = 2 * (pred * tgt).sum(-1)
    den = pred.sum(-1) + tgt.sum(-1)
    return (1 - (num + eps) / (den + eps)).mean()


def get_uncertain_point_coords_with_randomness(
    logits: torch.Tensor, num_points: int,
    oversample_ratio: float = 3.0, importance_sample_ratio: float = 0.75,
) -> torch.Tensor:
    """Mask2Former point sampling: bias points toward UNCERTAIN locations (|logit|~0,
    i.e. mask boundaries / occluded edges) plus uniform coverage. Concentrating the
    loss on hard pixels is what produces sharp, COMPLETE masks instead of blobs.

    logits: (M,1,H,W) -> coords (M, num_points, 2) in [0,1].
    """
    m = logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    coords = torch.rand(m, num_sampled, 2, device=logits.device, dtype=logits.dtype)
    pl = point_sample(logits, coords)                       # (M,1,num_sampled)
    uncertainty = -torch.abs(pl[:, 0])                      # (M,num_sampled): high near boundary
    num_uncertain = int(importance_sample_ratio * num_points)
    num_random = num_points - num_uncertain
    idx = torch.topk(uncertainty, k=num_uncertain, dim=1)[1]            # (M,num_uncertain)
    shift = num_sampled * torch.arange(m, device=logits.device).unsqueeze(1)
    chosen = coords.view(-1, 2)[(idx + shift).view(-1)].view(m, num_uncertain, 2)
    if num_random > 0:
        chosen = torch.cat(
            [chosen, torch.rand(m, num_random, 2, device=logits.device, dtype=logits.dtype)], dim=1)
    return chosen


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
        num_points: int = LOSS_NUM_POINTS,
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
            tm = targets[i]["masks"][gi].to(pm.dtype).to(device)        # (M,Hg,Wg) {0,1}
            ml.append(pm)
            mt.append(tm)
            if with_depth and "pred_depth" in outputs:
                pdl.append(outputs["pred_depth"][i][pi].squeeze(-1))
                tdl.append(targets[i]["depths"][gi].to(device))

        if ml:
            pm = torch.cat(ml, 0)                                        # (M,Hf,Wf) logits
            tm = torch.cat(mt, 0)                                        # (M,Hg,Wg)
            # Point-rendered mask loss (Mask2Former): supervise on hard sampled points
            # at NORMALIZED coords (no GT resize). Pred and GT are sampled at the SAME
            # points; uncertainty sampling focuses gradient on boundaries/occlusions.
            with torch.no_grad():
                pts = get_uncertain_point_coords_with_randomness(pm.unsqueeze(1), self.num_points)
            point_logits = point_sample(pm.unsqueeze(1), pts).squeeze(1)  # (M,P) grad
            point_labels = point_sample(tm.unsqueeze(1), pts).squeeze(1)  # (M,P)
            loss_mask = F.binary_cross_entropy_with_logits(point_logits, point_labels)
            loss_dice = dice_loss(point_logits, point_labels)
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
