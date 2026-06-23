"""Point sampling for mask matching and losses (Mask2Former [12] / PointRend [28]).

Comparing full-resolution masks lets the large, easy background dominate the
gradient, so masks converge fuzzy and the boundaries between adjacent people stay
soft -- which leaves parts of a person "unclaimed" and lets a second query grab
them (the fragment-slot problem). Sampling a small set of points instead, biased
toward UNCERTAIN locations near the mask boundary for the loss (and uniform for
matching), concentrates the signal where it matters: masks sharpen, the main
query covers the whole person, and fragments stop firing. This is the method the
paper inherits from Mask2Former.

Placed at: instancedepth/models/instance/point_sampling.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def point_sample(input: torch.Tensor, point_coords: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample a feature/mask map at arbitrary points.

    input        (M, C, H, W)
    point_coords (M, P, 2) in [0, 1]  (x, y)
    returns      (M, C, P)
    Because coords are normalized, pred and GT can be sampled at the SAME points
    even when their resolutions differ (no resize needed).
    """
    grid = 2.0 * point_coords - 1.0                       # [0,1] -> [-1,1]
    out = F.grid_sample(input, grid.unsqueeze(2), align_corners=False)  # (M,C,P,1)
    return out.squeeze(-1)


@torch.no_grad()
def _uncertainty(logits: torch.Tensor) -> torch.Tensor:
    """High where the mask logit is near 0 (i.e. near the decision boundary)."""
    return -logits.abs()


@torch.no_grad()
def uncertain_point_coords(
    logits: torch.Tensor,            # (M, 1, H, W) mask logits
    num_points: int,
    oversample_ratio: float = 3.0,
    importance_ratio: float = 0.75,
) -> torch.Tensor:
    """Boundary-biased point coords, (M, num_points, 2) in [0, 1].

    Oversample uniformly, keep the most-uncertain `importance_ratio` fraction, and
    top up with uniform points (Mask2Former's get_uncertain_point_coords).
    """
    m = logits.shape[0]
    over = max(int(num_points * oversample_ratio), num_points)
    coords = torch.rand(m, over, 2, device=logits.device)
    vals = point_sample(logits, coords)[:, 0]             # (M, over)
    unc = _uncertainty(vals)                              # (M, over)
    k = int(importance_ratio * num_points)
    idx = unc.topk(k, dim=1)[1]                           # (M, k) most uncertain
    chosen = torch.gather(coords, 1, idx.unsqueeze(-1).expand(-1, -1, 2))
    rest = num_points - k
    if rest > 0:                                          # top up with uniform points
        chosen = torch.cat([chosen, torch.rand(m, rest, 2, device=logits.device)], dim=1)
    return chosen
