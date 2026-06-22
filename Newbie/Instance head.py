"""Instance Depth Layer Prediction head (Section 4.2.1, Fig. 7).

Builds per-pixel mask features + multi-scale memory from the (frozen-in-phase-2)
HDI/backbone features, runs two task-specific query streams (mask + depth) through
a SHARED Mask2Former decoder, fuses them, and emits per-instance
{mask logits, class logits, depth layer}.

WHAT CHANGED vs. the previous version
-------------------------------------
1. DEEP SUPERVISION. The decoder now returns a prediction after every layer; this
   head exposes them as `aux_outputs` so the criterion supervises each one (the
   Mask2Former mechanism that fixes jittery, low-quality masks).
2. UNIFIED MASK PROJECTION. A single `_predictor` applies the decoder-norm and the
   mask_embed/class heads; it is used both for the per-layer attention mask
   (inside the decoder) and for the outputs, so they are consistent.
3. SMALLER num_queries DEFAULT (32). With ~2-3 instances per frame, 100 queries
   left ~97 unmatched queries free to fire on background; 32 keeps comfortable
   headroom while sharpening the learning signal and cutting spurious masks.

The DepthDecoder/PixelDecoder split, the two query streams, and query fusion are
unchanged. Depth layers are computed only at the FINAL layer (the paper's depth
layer is a per-instance scalar; Mask2Former-style deep supervision applies to
mask + class).

[Paper Specified]    N predictions of {Mask, Cls, Dep}; Mask2Former-based;
                     task-specific queries + query fusion; Dep = sigmoid*max_depth.
[Strongly Inferred]  mask = dot(mask_embed(query), mask_features); class via linear;
                     depth via MLP on the fused query; deep supervision per layer.

Tensor shapes (N queries, K classes, C hidden)
    pred_masks   (B,N,Hf,Wf)
    pred_logits  (B,N,K+1)        +1 no-object class
    pred_depth   (B,N,1)          in [0, max_depth]
    aux_outputs  list[{pred_masks (B,N,Hf,Wf), pred_logits (B,N,K+1)}]  (per layer)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .pixel_decoder import PixelDecoder, build_2d_sincos_pos_embed
from .query_fusion import QueryFusion
from .transformer_decoder import TransformerDecoder


@dataclass
class InstanceHeadConfig:
    in_channels: int = 256
    hidden_dim: int = 256
    num_queries: int = 32           # was 100; ~2-3 instances/frame -> 32 is ample
    num_classes: int = 1
    num_scales: int = 3
    num_transformer_feats: int = 3
    num_decoder_layers: int = 9
    num_heads: int = 8
    ffn_hidden: int = 2048
    dropout: float = 0.0
    max_depth: float = 10.0
    mask_threshold: float = 0.5


class InstanceDepthLayerHead(nn.Module):
    def __init__(self, cfg: InstanceHeadConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c = cfg.hidden_dim

        self.pixel_decoder = PixelDecoder(
            in_channels=cfg.in_channels, hidden_dim=c,
            num_scales=cfg.num_scales, num_transformer_feats=cfg.num_transformer_feats,
        )

        # two task-specific query embedding sets + shared learned positional embeds
        self.mask_query_feat = nn.Embedding(cfg.num_queries, c)
        self.depth_query_feat = nn.Embedding(cfg.num_queries, c)
        self.query_pos = nn.Embedding(cfg.num_queries, c)

        self.transformer = TransformerDecoder(
            hidden_dim=c, num_heads=cfg.num_heads, num_layers=cfg.num_decoder_layers,
            ffn_hidden=cfg.ffn_hidden, dropout=cfg.dropout, mask_threshold=cfg.mask_threshold,
        )
        self.query_fusion = QueryFusion(c)

        # prediction heads (owned here; reached by the decoder via _predictor)
        self.decoder_norm = nn.LayerNorm(c)
        self.class_head = nn.Linear(c, cfg.num_classes + 1)         # +1 no-object
        self.mask_embed = nn.Sequential(
            nn.Linear(c, c), nn.ReLU(inplace=True),
            nn.Linear(c, c), nn.ReLU(inplace=True), nn.Linear(c, c),
        )
        self.depth_head = nn.Sequential(
            nn.Linear(c, c), nn.ReLU(inplace=True), nn.Linear(c, 1),
        )

    # --------------------------------------------------------------- predictor
    def _predictor(self, query: torch.Tensor, mask_features: torch.Tensor
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(Nq,B,C) -> (B,Nq,Hf,Wf) mask logits, (B,Nq,K+1) class logits.

        Used by the decoder for BOTH the per-layer attention mask and the output,
        so attention and prediction use the same mask.
        """
        q = self.decoder_norm(query).permute(1, 0, 2)               # (B,Nq,C)
        class_logits = self.class_head(q)                           # (B,Nq,K+1)
        mask_emb = self.mask_embed(q)                               # (B,Nq,C)
        mask_logits = torch.einsum("bnc,bchw->bnhw", mask_emb, mask_features)
        return mask_logits, class_logits

    # ----------------------------------------------------------------- forward
    def forward(
        self,
        decoder_feats: List[torch.Tensor],          # [f8,f4,f2], coarse->fine
        init_depth: Optional[torch.Tensor] = None,  # (B,1,H,W) HDI prior (unused by default)
    ) -> Dict[str, object]:
        b = decoder_feats[0].shape[0]
        device, dtype = decoder_feats[0].device, decoder_feats[0].dtype

        mask_features, ms_feats = self.pixel_decoder(decoder_feats)
        ms_pos = [
            build_2d_sincos_pos_embed(f.shape[1], f.shape[2], f.shape[3], device, dtype).expand(b, -1, -1, -1)
            for f in ms_feats
        ]

        n = self.cfg.num_queries
        mask_q = self.mask_query_feat.weight.unsqueeze(1).repeat(1, b, 1)   # (N,B,C)
        depth_q = self.depth_query_feat.weight.unsqueeze(1).repeat(1, b, 1)  # (N,B,C)
        qpos = self.query_pos.weight.unsqueeze(1).repeat(1, b, 1)
        cat_q = torch.cat([mask_q, depth_q], dim=0)                         # (2N,B,C)
        cat_pos = torch.cat([qpos, qpos], dim=0)

        # shared decoder over both streams; deep-supervised predictions returned
        predictions, final_q = self.transformer(
            cat_q, cat_pos, ms_feats, ms_pos, mask_features, self._predictor)

        # depth layer from the FINAL fused (mask + depth) queries
        final_mask_q, final_depth_q = final_q[:n], final_q[n:]
        fused = self.query_fusion(final_mask_q, final_depth_q)              # (N,B,C), normed
        depth_layer = torch.sigmoid(self.depth_head(fused.permute(1, 0, 2))) * self.cfg.max_depth  # (B,N,1)

        # final mask/class = last prediction, mask stream (first N of the 2N queries)
        final_mask_logits = predictions[-1][0][:, :n]
        final_class_logits = predictions[-1][1][:, :n]
        aux = [{"pred_masks": p[0][:, :n], "pred_logits": p[1][:, :n]}
               for p in predictions[:-1]]

        return {
            "pred_masks": final_mask_logits,        # (B,N,Hf,Wf)
            "pred_logits": final_class_logits,      # (B,N,K+1)
            "pred_depth": depth_layer,              # (B,N,1)
            "mask_features": mask_features,         # (B,C,Hf,Wf)
            "aux_outputs": aux,                     # per-layer for deep supervision
        }
