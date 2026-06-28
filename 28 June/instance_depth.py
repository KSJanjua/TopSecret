from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configs.run_config import RunConfig
from .backbone.dinov2_dpt import DINOv2DPTBackbone
from .hdi.heads import ConfidenceNet, InitDepthHead, RangeSegHead
from .hdi.range_decoder import DepthRangeFeatureDecoder
from .instance.instance_head import InstanceDepthLayerHead
from .refine.occlusion_refine import OcclusionAwareRefinement
from .registry import register_model


@register_model("instance_depth")
class InstanceDepth(nn.Module):
    def __init__(self, cfg: RunConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c = cfg.hidden_dim

        # ---- shared backbone ----
        self.backbone = DINOv2DPTBackbone(cfg.backbone)

        # ---- Phase 1 HDI components (feature-driven, no internal backbone) ----
        self.range_decoder = DepthRangeFeatureDecoder(
            cfg.decoder, num_scales=len(cfg.backbone.out_strides)
        )
        rd = cfg.hdi.num_ranges
        self.num_ranges = rd
        self.range_seg_head = RangeSegHead(c, rd, cfg.hdi.head_hidden)
        self.init_depth_head = InitDepthHead(c, cfg.max_depth, cfg.hdi.head_hidden)
        self.confidence_net = ConfidenceNet(c, rd, cfg.hdi.head_hidden)
        self.register_buffer(
            "range_step", torch.tensor(cfg.max_depth / rd, dtype=torch.float32), persistent=False
        )
        self.num_refine_steps = cfg.hdi.num_refine_steps

        # ---- Phase 2 instance head ----
        self.instance_head = InstanceDepthLayerHead(cfg.instance)

        # ---- Phase 3 refinement ----
        self.refine = OcclusionAwareRefinement(cfg.refine)

        self._phase: int = 1

    # ----------------------------------------------------------- HDI internals
    def _hdi_from_feats(self, feats: List[torch.Tensor], out_hw):
        range_feats = self.range_decoder(feats)                 # (B,C,h,w)
        range_logits, range_probs = self.range_seg_head(range_feats)
        depth = self.init_depth_head(range_feats)               # (B,1,h,w)
        for _ in range(self.num_refine_steps):
            conf = self.confidence_net(range_feats, depth)      # C_i  Eq.1
            r = (conf * range_probs).sum(1, keepdim=True)       # R_i  Eq.2
            e = (2.0 * r - 1.0) * self.range_step               # E_i  signed form of Eq.3
            depth = (depth + e).clamp(0.0, self.cfg.max_depth)  # D_{i+1}  Eq.4
        init_depth = F.interpolate(depth, size=out_hw, mode="bilinear", align_corners=False)
        return init_depth, range_logits, range_probs

    # --------------------------------------------------------------- phases
    def set_phase(self, phase: int) -> None:
        """Configure freezing per Sec. 4.3 (phase in {1,2,3})."""
        assert phase in (1, 2, 3)
        self._phase = phase

        def req(module: nn.Module, flag: bool):
            for p in module.parameters():
                p.requires_grad_(flag)

        if phase == 1:
            req(self.backbone, True); req(self.range_decoder, True)
            req(self.range_seg_head, True); req(self.init_depth_head, True); req(self.confidence_net, True)
            req(self.instance_head, False); req(self.refine, False)
        elif phase == 2:
            req(self.backbone, False); req(self.range_decoder, False)
            req(self.range_seg_head, False); req(self.init_depth_head, False); req(self.confidence_net, False)
            req(self.instance_head, True); req(self.refine, False)
        else:  # phase 3
            req(self.backbone, True); req(self.range_decoder, True)
            req(self.range_seg_head, True); req(self.init_depth_head, True); req(self.confidence_net, True)
            req(self.instance_head, False); req(self.refine, True)

    # --------------------------------------------------------------- forward
    def forward(
        self,
        rgb: torch.Tensor,
        run_instance: bool = True,
        run_refine: bool = True,
    ) -> Dict[str, object]:
        b, _, H, W = rgb.shape
        feats = self.backbone(rgb)                              # [f8,f4,f2]

        init_depth, range_logits, range_probs = self._hdi_from_feats(feats, (H, W))
        out: Dict[str, object] = {
            "init_depth": init_depth,
            "range_logits": range_logits,
            "range_probs": range_probs,
            "refined_depth": init_depth,
        }
        if not run_instance:
            return out

        inst = self.instance_head(feats, init_depth=init_depth)
        out.update({
            "pred_masks": inst["pred_masks"],
            "pred_logits": inst["pred_logits"],
            "pred_depth": inst["pred_depth"],
            "mask_features": inst["mask_features"],
            "aux_outputs": inst.get("aux_outputs",[]),
        })
        if not run_refine:
            return out

        # Phase 3 fixes the instance decoder (Sec. 4.3): feed its outputs to the
        # refinement as DETACHED, fixed inputs so the refinement loss cannot
        # backprop into the (frozen) instance head or the shared backbone through
        # it. init_depth is NOT detached -- the dense anchor must train the HDI.
        inst_fixed = {
            "pred_logits": inst["pred_logits"].detach(),
            "pred_masks": inst["pred_masks"].detach(),
            "pred_depth": inst["pred_depth"].detach(),
            "mask_features": inst["mask_features"].detach(),
        }
        ref = self.refine(inst_fixed, init_depth)
        out.update({
            "refined_depth": ref["refined_depth"],
            "d_hat": ref["d_hat"],
            "e_obj": ref["e_obj"],
            "refine_meta": {
                "pair_query_idx": ref["pair_query_idx"],
                "batch_index": ref["batch_index"],
                "per_image_pairs": ref["per_image_pairs"],
            },
        })
        return out
