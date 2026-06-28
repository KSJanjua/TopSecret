"""Occlusion-Aware Depth Refinement module (Section 4.2.2).

Orchestrates the full stage:
    instance predictions + HDI depth
      -> select occlusion pairs (filter, IoU>0.1, nearest-depth guest)
      -> boxes from masks
      -> ROIAlign F_obj + geometric priors G_obj
      -> Phi_o relation reasoning -> E_obj (Eq.8) -> D_hat (Eq.9)
      -> scatter refined instance depths back onto the holistic depth map.

Occlusion reasoning operates on instance DEPTH LAYERS (scalars per instance);
the refined layers are then composited into the per-pixel depth via the instance
masks, leaving non-instance pixels at their HDI value.

Inputs (forward)
----------------
    instance_out : dict from InstanceDepthLayerHead
        pred_logits (B,N,K+1), pred_masks (B,N,Hf,Wf), pred_depth (B,N,1),
        mask_features (B,C,Hf,Wf)
    init_depth   : (B,1,H,W) holistic depth from HDI
Outputs
-------
    dict:
        d_hat        (P,2)   refined instance depths (for loss)
        dt_targets   None    (filled by caller during training via GT match)
        pairs        list[(P_b,2)] query-index pairs per image
        batch_index  (P,)
        refined_depth (B,1,H,W) holistic depth with instance layers composited
Complexity
----------
    Dominated by pair IoU O(sum_b M_b^2 * Hf * Wf) and ROIAlign O(P * C * Hp^2).
    Both are small relative to the DINOv2 backbone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pair_selection import select_occlusion_pairs, masks_to_boxes
from .roi_extract import ROIPairExtractor
from .relation_reason import RelationReasoning


@dataclass
class RefineConfig:
    in_channels: int = 256          # depth feature channels (mask_features C)
    geom_channels: int = 4          # mask(1) + coords(2) + depth(1)
    roi_size: int = 7               # Hp = Wp
    sampling_ratio: int = 2
    mlp_hidden: int = 256
    cls_thresh: float = 0.9
    mask_thresh: float = 0.8
    iou_thresh: float = 0.1
    mask_binarize: float = 0.5
    max_depth: float = 10.0


class OcclusionAwareRefinement(nn.Module):
    def __init__(self, cfg: RefineConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.roi = ROIPairExtractor(output_size=cfg.roi_size, sampling_ratio=cfg.sampling_ratio)
        self.reason = RelationReasoning(
            in_channels=cfg.in_channels,
            geom_channels=cfg.geom_channels,
            output_size=cfg.roi_size,
            hidden=cfg.mlp_hidden,
        )

    def forward(
        self,
        instance_out: Dict[str, torch.Tensor],
        init_depth: torch.Tensor,                  # (B,1,H,W)
        depth_feats: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        cfg = self.cfg
        logits = instance_out["pred_logits"]       # (B,N,K+1)
        masks = instance_out["pred_masks"]         # (B,N,Hf,Wf)
        depth = instance_out["pred_depth"].squeeze(-1)  # (B,N)
        if depth_feats is None:
            depth_feats = instance_out["mask_features"]  # (B,C,Hf,Wf)
        b, n = logits.shape[:2]
        device = logits.device

        all_boxes: List[torch.Tensor] = []
        all_qidx: List[torch.Tensor] = []
        all_bidx: List[int] = []
        per_image_pairs: List[torch.Tensor] = []

        for i in range(b):
            pairs, _ = select_occlusion_pairs(
                logits[i], masks[i], depth[i],
                cls_thresh=cfg.cls_thresh, mask_thresh=cfg.mask_thresh,
                iou_thresh=cfg.iou_thresh, mask_binarize=cfg.mask_binarize,
            )
            per_image_pairs.append(pairs)
            if pairs.numel() == 0:
                continue
            boxes_i = masks_to_boxes(masks[i], cfg.mask_binarize)   # (N,4)
            for (mi, gi) in pairs.tolist():
                all_boxes.append(torch.stack([boxes_i[mi], boxes_i[gi]], dim=0))  # (2,4)
                all_qidx.append(torch.tensor([mi, gi], device=device))
                all_bidx.append(i)

        if len(all_boxes) == 0:
            return {
                "d_hat": depth.new_zeros(0, 2),
                "e_obj": depth.new_zeros(0, 2),
                "pair_query_idx": torch.empty(0, 2, dtype=torch.long, device=device),
                "batch_index": torch.empty(0, dtype=torch.long, device=device),
                "per_image_pairs": per_image_pairs,
                "refined_depth": init_depth,
            }

        boxes = torch.stack(all_boxes, dim=0)                       # (P,2,4)
        pair_qidx = torch.stack(all_qidx, dim=0)                    # (P,2)
        batch_index = torch.tensor(all_bidx, dtype=torch.long, device=device)  # (P,)

        # ROI features are fixed inputs to Phi_o (depth_feats/masks already detached
        # by the caller in phase 3; detach init_depth here too) so the refinement
        # loss trains only Phi_o. The HDI is trained via the differentiable
        # composite + dense anchor, not via this ROI depth patch.
        f_obj, g_obj = self.roi(depth_feats, masks, init_depth.detach(), boxes, batch_index, pair_qidx)

        d_obj = depth[batch_index[:, None].expand(-1, 2), pair_qidx]  # (P,2)
        d_hat, e_obj = self.reason(f_obj, g_obj, d_obj)              # (P,2),(P,2)
        d_hat = d_hat.clamp(0.0, cfg.max_depth)

        # --- composite refined instance layers back onto the depth map (grad) ---
        refined = self._composite(init_depth, masks, depth, d_hat, batch_index, pair_qidx, cfg.mask_binarize)

        return {
            "d_hat": d_hat,
            "e_obj": e_obj,
            "pair_query_idx": pair_qidx,
            "batch_index": batch_index,
            "per_image_pairs": per_image_pairs,
            "refined_depth": refined,
        }

    def _composite(
        self,
        init_depth: torch.Tensor,        # (B,1,H,W)  grad (HDI)
        masks: torch.Tensor,             # (B,N,Hf,Wf) detached
        orig_depth: torch.Tensor,        # (B,N) detached
        d_hat: torch.Tensor,             # (P,2) grad via Phi_o
        batch_index: torch.Tensor,       # (P,)
        pair_qidx: torch.Tensor,         # (P,2)
        binarize: float,
    ) -> torch.Tensor:
        """Composite refined instance depth layers onto the dense map (differentiable).

        Each refined instance shifts its mask region by (d_hat - d_obj). Built
        functionally (no in-place on grad tensors) so the dense refinement loss
        trains Phi_o (via d_hat) and the HDI (via init_depth). Masks and d_obj are
        detached, so the only gradient through a region is the scalar shift.
        """
        b, _, H, W = init_depth.shape
        binm = (F.interpolate(masks, size=(H, W), mode="bilinear", align_corners=False)
                .sigmoid() >= binarize).to(init_depth.dtype)        # (B,N,H,W) detached

        # one refined delta per unique (image, query); a query can be a main in one
        # pair and a guest in another -> average its d_hat.
        acc = {}
        for p in range(d_hat.shape[0]):
            i = int(batch_index[p])
            for k in range(2):
                acc.setdefault((i, int(pair_qidx[p, k])), []).append(d_hat[p, k])
        per_img = {}
        for (i, q), vals in acc.items():
            delta = torch.stack(vals).mean() - orig_depth[i, q]     # shift (orig detached)
            per_img.setdefault(i, []).append((delta, binm[i, q]))

        maps, covers = [], []
        for i in range(b):
            terms = per_img.get(i, [])
            if terms:
                deltas = torch.stack([t[0] for t in terms])         # (Ki,) grad
                regions = torch.stack([t[1] for t in terms])        # (Ki,H,W) detached
                maps.append(torch.einsum("k,khw->hw", deltas, regions))
                covers.append(regions.sum(0))
            else:
                maps.append(init_depth.new_zeros(H, W))
                covers.append(init_depth.new_zeros(H, W))
        delta_map = torch.stack(maps).unsqueeze(1)                  # (B,1,H,W)
        cover = torch.stack(covers).unsqueeze(1).clamp(min=1.0)     # average overlaps
        return (init_depth + delta_map / cover).clamp(0.0, self.cfg.max_depth)