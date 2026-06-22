"""Masked-attention transformer decoder (Mask2Former [12], Fig. 7) WITH deep supervision.

WHAT CHANGED vs. the previous version
-------------------------------------
1. DEEP SUPERVISION. The decoder now produces a prediction (class + mask logits)
   from the INITIAL queries and after EVERY layer (num_layers + 1 predictions),
   and returns them all. This lets the loss supervise every decoder layer -- the
   mechanism Mask2Former depends on for sharp masks. Previously only the final
   layer was returned and supervised, which is the primary reason the masks were
   jittery and converged poorly.
2. UNIFIED MASK PROJECTION. The per-layer attention mask is now built from the
   SAME mask prediction used for the output, via an injected
   `predictor(query, mask_features) -> (mask_logits, class_logits)` supplied by
   the head (it applies the decoder-norm + mask_embed/class heads). Previously
   the attention mask used the RAW query dotted with mask_features -- a different,
   cruder mask than the predicted one -- which mis-steered cross-attention.

The decoder no longer owns the prediction heads or the final LayerNorm; those
live in the head and are reached through `predictor`.

Per-layer flow (Nq queries, C hidden, memory feat (B,C,h,w)):
    predictor(initial query) -> (mask_logits, class_logits)        [prediction 0]
    for each layer (round-robin over the multi-scale feats):
        attn_mask <- (downsample(mask_logits).sigmoid() < thresh)  (detached)
        query <- masked cross-attn -> self-attn -> FFN
        predictor(query) -> (mask_logits, class_logits)            [prediction i+1]
    return [predictions...], final_query
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _FFN(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(hidden, dim),
        )
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.drop(self.net(x)))


class _DecoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_hidden: int, dropout: float) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.norm_ca = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.norm_sa = nn.LayerNorm(dim)
        self.ffn = _FFN(dim, ffn_hidden, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,                    # (Nq,B,C)
        query_pos: torch.Tensor,                # (Nq,B,C)
        memory: torch.Tensor,                   # (HW,B,C)
        memory_pos: torch.Tensor,               # (HW,B,C)
        attn_mask: Optional[torch.Tensor],      # (B*heads,Nq,HW) or None
    ) -> torch.Tensor:
        # masked cross-attention
        q = query + query_pos
        k = memory + memory_pos
        ca, _ = self.cross_attn(q, k, value=memory, attn_mask=attn_mask)
        query = self.norm_ca(query + self.drop(ca))
        # self-attention
        q2 = query + query_pos
        sa, _ = self.self_attn(q2, q2, value=query)
        query = self.norm_sa(query + self.drop(sa))
        # FFN
        return self.ffn(query)


class TransformerDecoder(nn.Module):
    """Mask2Former decoder; returns per-layer predictions + final query features."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 9,
        ffn_hidden: int = 2048,
        dropout: float = 0.0,
        mask_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mask_threshold = mask_threshold
        self.layers = nn.ModuleList(
            [_DecoderLayer(hidden_dim, num_heads, ffn_hidden, dropout) for _ in range(num_layers)]
        )

    def _attn_mask_from_logits(self, mask_logits: torch.Tensor, feat_hw) -> torch.Tensor:
        """mask_logits (B,Nq,Hf,Wf) -> (B*heads,Nq,h*w) bool (True = block)."""
        h, w = feat_hw
        ml = F.interpolate(mask_logits, size=(h, w), mode="bilinear", align_corners=False)
        attn = (ml.sigmoid() < self.mask_threshold).flatten(2)      # (B,Nq,h*w)
        # A query blocking EVERY location -> softmax(all -inf) = NaN; let it see all.
        attn[attn.all(dim=-1)] = False
        attn = attn.unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1)
        return attn.detach()

    def forward(
        self,
        query_feat: torch.Tensor,                       # (Nq,B,C)
        query_pos: torch.Tensor,                        # (Nq,B,C)
        multi_scale_feats: List[torch.Tensor],          # each (B,C,h,w), coarse->fine
        multi_scale_pos: List[torch.Tensor],            # each (B,C,h,w)
        mask_features: torch.Tensor,                    # (B,C,Hf,Wf)
        predictor: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        query = query_feat
        num_feats = len(multi_scale_feats)

        predictions: List[Tuple[torch.Tensor, torch.Tensor]] = []
        mask_logits, class_logits = predictor(query, mask_features)     # prediction 0 (initial)
        predictions.append((mask_logits, class_logits))

        for i, layer in enumerate(self.layers):
            idx = i % num_feats
            feat, pos = multi_scale_feats[idx], multi_scale_pos[idx]
            h, w = feat.shape[-2:]
            memory = feat.flatten(2).permute(2, 0, 1)                   # (h*w,B,C)
            memory_pos = pos.flatten(2).permute(2, 0, 1)
            attn_mask = self._attn_mask_from_logits(mask_logits, (h, w))
            query = layer(query, query_pos, memory, memory_pos, attn_mask)
            mask_logits, class_logits = predictor(query, mask_features)  # prediction i+1
            predictions.append((mask_logits, class_logits))

        return predictions, query                                       # len = num_layers + 1
