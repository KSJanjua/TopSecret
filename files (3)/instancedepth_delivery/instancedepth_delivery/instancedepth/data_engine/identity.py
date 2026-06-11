"""Tracking-identity post-processing (paper Sec. 3, "Tracking Identity
Generation").

The paper generates initial identity masks with an off-the-shelf tracker
(DEVA) and then matches masks "to prior masks using the highest Intersection
over Union (IoU) to maintain identity consistency". We reproduce that exact
IoU-matching repair on top of SAM3's per-concept tracks, plus the merging that
becomes necessary because SAM3 runs one concept per session:

  1. assign_global_ids : (concept, local_id) -> stable global track id.
  2. dedup_cross_concept : on each frame, masks from different concepts with
     IoU > threshold are duplicates; the higher detection score survives.
  3. repair_identities : highest-IoU matching between a dying track's last mask
     and a newborn track's first mask (within a small frame gap) re-links
     fragmented identities — the paper's IoU consistency step.
  4. filter_short_tracks : drop tracks shorter than `min_track_length`.

All steps operate on the normalized structure

    tracks: Dict[int, Track]      # global_id -> Track
    Track.masks: Dict[int, np.ndarray(bool)]   # frame_index -> mask
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .config import IdentityConfig
from .sam3_engine import FramesOut

log = logging.getLogger("data_engine.identity")


@dataclass
class Track:
    gid: int
    category: str
    category_id: int
    masks: Dict[int, np.ndarray] = field(default_factory=dict)   # frame -> bool (H,W)
    scores: Dict[int, float] = field(default_factory=dict)

    @property
    def first_frame(self) -> int:
        return min(self.masks)

    @property
    def last_frame(self) -> int:
        return max(self.masks)

    def __len__(self) -> int:
        return len(self.masks)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum(dtype=np.int64)
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum(dtype=np.int64)
    return float(inter) / float(union)


# --------------------------------------------------------------------------- #
def assign_global_ids(
    per_concept: Dict[str, FramesOut], category_ids: Dict[str, int]
) -> Dict[int, Track]:
    """Flatten {prompt -> frames -> local_id -> obs} into global tracks."""
    tracks: Dict[int, Track] = {}
    next_gid = 1
    for prompt, frames_out in per_concept.items():
        local_to_gid: Dict[int, int] = {}
        for fi in sorted(frames_out):
            for lid, obs in frames_out[fi].items():
                if lid not in local_to_gid:
                    local_to_gid[lid] = next_gid
                    tracks[next_gid] = Track(
                        gid=next_gid, category=prompt,
                        category_id=category_ids[prompt],
                    )
                    next_gid += 1
                t = tracks[local_to_gid[lid]]
                t.masks[fi] = obs.mask
                t.scores[fi] = obs.score
    return tracks


def dedup_cross_concept(tracks: Dict[int, Track], cfg: IdentityConfig) -> Dict[int, Track]:
    """Remove per-frame duplicate masks produced by different concept prompts."""
    if len({t.category for t in tracks.values()}) <= 1:
        return tracks
    frames = sorted({fi for t in tracks.values() for fi in t.masks})
    for fi in frames:
        present: List[Tuple[int, np.ndarray, float, str]] = [
            (gid, t.masks[fi], t.scores[fi], t.category)
            for gid, t in tracks.items() if fi in t.masks
        ]
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                gi, mi, si, ci = present[i]
                gj, mj, sj, cj = present[j]
                if ci == cj:
                    continue
                if mask_iou(mi, mj) > cfg.cross_concept_dedup_iou:
                    loser = gi if si < sj else gj
                    tracks[loser].masks.pop(fi, None)
                    tracks[loser].scores.pop(fi, None)
    return {g: t for g, t in tracks.items() if len(t)}


def repair_identities(tracks: Dict[int, Track], cfg: IdentityConfig) -> Dict[int, Track]:
    """Highest-IoU re-linking of fragmented tracks (paper's IoU matching)."""
    merged = True
    while merged:
        merged = False
        gids = sorted(tracks, key=lambda g: tracks[g].first_frame)
        for ga in gids:
            if ga not in tracks:
                continue
            a = tracks[ga]
            # candidate newborn tracks of the same category, starting just
            # after `a` ends, ranked by boundary-mask IoU.
            best_g, best_iou = None, cfg.reid_iou
            for gb in gids:
                if gb == ga or gb not in tracks:
                    continue
                b = tracks[gb]
                if b.category_id != a.category_id:
                    continue
                gap = b.first_frame - a.last_frame
                if not (0 < gap <= cfg.max_gap):
                    continue
                iou = mask_iou(a.masks[a.last_frame], b.masks[b.first_frame])
                if iou > best_iou:
                    best_g, best_iou = gb, iou
            if best_g is not None:
                b = tracks.pop(best_g)
                a.masks.update(b.masks)
                a.scores.update(b.scores)
                log.info("re-linked track %d -> %d (IoU=%.2f)", best_g, ga, best_iou)
                merged = True
    return tracks


def filter_short_tracks(tracks: Dict[int, Track], cfg: IdentityConfig) -> Dict[int, Track]:
    kept = {g: t for g, t in tracks.items() if len(t) >= cfg.min_track_length}
    if len(kept) < len(tracks):
        log.info("dropped %d short tracks (<%d frames)",
                 len(tracks) - len(kept), cfg.min_track_length)
    return kept


def renumber(tracks: Dict[int, Track]) -> Dict[int, Track]:
    """Compact ids to 1..K (uint16 mask encoding requires K < 65536)."""
    out: Dict[int, Track] = {}
    for new_gid, old in enumerate(sorted(tracks), start=1):
        t = tracks[old]
        t.gid = new_gid
        out[new_gid] = t
    return out


def build_tracks(
    per_concept: Dict[str, FramesOut],
    category_ids: Dict[str, int],
    cfg: IdentityConfig,
) -> Dict[int, Track]:
    tracks = assign_global_ids(per_concept, category_ids)
    tracks = dedup_cross_concept(tracks, cfg)
    tracks = repair_identities(tracks, cfg)
    tracks = filter_short_tracks(tracks, cfg)
    return renumber(tracks)
