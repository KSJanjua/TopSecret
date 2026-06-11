"""Per-sequence GID-style annotation generation.

For one sequence (one timestamp folder) this module produces everything the
GID dataset provides per frame (paper Sec. 3 + Fig. 2):

  * object_masks/<frame>.png   uint16, pixel value = tracking identity
                               (0 = background) — instance masks + consistent
                               identities in one file.
  * ground_masks/<frame>.png   uint8 binary ground mask.
  * annotations.json           per-frame instances: track_id, category,
                               bbox (xyxy), area, GT instance depth layer
                               (= mean valid GT depth inside the mask,
                               Sec. 4.2.1), and valid-depth pixel count.

Overlap handling: GID's "Object Mask" images give each pixel a unique
identity color, i.e. the stored object mask is a single-label map. When SAM3
masks overlap (occlusions), the pixel is assigned to the instance with the
SMALLER depth layer (the occluder is, by definition, in front). The per-track
binary masks used for depth-layer statistics are computed BEFORE this
flattening so the depth layer of a partially hidden object is not biased by
the visibility resolution.  [Reasonable Assumption]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import DataEngineConfig
from .depth_io import detect_unit_scale, load_depth_meters
from .discover import SequenceRecord
from .identity import Track, build_tracks
from .sam3_engine import FramesOut, VideoSegmenter, build_segmenter

log = logging.getLogger("data_engine.annotate")


# --------------------------------------------------------------------------- #
def _bbox_from_mask(mask: np.ndarray) -> Optional[List[int]]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _depth_layer(mask: np.ndarray, depth_m: np.ndarray) -> Tuple[float, int]:
    """GT instance depth layer = mean of VALID metric depth inside the mask.

    [Strongly Inferred] Sec. 4.2.1 defines the depth layer as "the average
    depth of the instance"; invalid sensor pixels (0) are excluded.
    """
    vals = depth_m[mask]
    vals = vals[vals > 0]
    if vals.size == 0:
        return 0.0, 0
    return float(vals.mean()), int(vals.size)


def _flatten_id_map(
    tracks: Dict[int, Track], fi: int, hw: Tuple[int, int],
    layer_by_gid: Dict[int, float],
) -> np.ndarray:
    """Single-label uint16 identity map; ties resolved nearest-depth-wins."""
    id_map = np.zeros(hw, dtype=np.uint16)
    depth_buf = np.full(hw, np.inf, dtype=np.float32)
    for gid, t in tracks.items():
        m = t.masks.get(fi)
        if m is None:
            continue
        d = layer_by_gid.get(gid, np.inf)
        win = m & (d < depth_buf)
        id_map[win] = gid
        depth_buf[win] = d
    return id_map


def _ground_union(per_concept: Dict[str, FramesOut], fi: int, hw) -> np.ndarray:
    g = np.zeros(hw, dtype=bool)
    for frames_out in per_concept.values():
        for obs in frames_out.get(fi, {}).values():
            g |= obs.mask
    return g


def _preview(rgb_path: Path, id_map: np.ndarray, ground: np.ndarray, out: Path) -> None:
    img = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    rng = np.random.default_rng(0)
    palette = rng.integers(40, 255, size=(int(id_map.max()) + 1, 3), dtype=np.uint8)
    palette[0] = 0
    overlay = palette[id_map]
    img = cv2.addWeighted(img, 0.55, overlay, 0.45, 0)
    img[ground] = (0.7 * img[ground] + 0.3 * np.array([0, 0, 255])).astype(np.uint8)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)


# --------------------------------------------------------------------------- #
def annotate_sequence(
    seq: SequenceRecord,
    cfg: DataEngineConfig,
    obj_segmenter: Optional[VideoSegmenter] = None,
    ground_segmenter: Optional[VideoSegmenter] = None,
) -> Dict:
    """Generate all annotations for one sequence. Returns the manifest dict."""
    rgb_paths = [f.rgb for f in seq.frames]
    probe = cv2.imread(str(rgb_paths[0]), cv2.IMREAD_COLOR)
    hw = probe.shape[:2]

    own_seg = obj_segmenter is None
    obj_segmenter = obj_segmenter or build_segmenter(cfg.sam3)
    ground_segmenter = ground_segmenter or obj_segmenter

    # ---- 1) SAM3: one video session per concept prompt --------------------
    per_concept_obj: Dict[str, FramesOut] = {}
    for prompt in cfg.sam3.object_prompts:
        log.info("[%s] tracking concept '%s' ...", seq.seq_id, prompt)
        per_concept_obj[prompt] = obj_segmenter.track_concept(rgb_paths, hw, prompt)

    per_concept_ground: Dict[str, FramesOut] = {}
    for prompt in cfg.sam3.ground_prompts:
        log.info("[%s] segmenting ground concept '%s' ...", seq.seq_id, prompt)
        per_concept_ground[prompt] = ground_segmenter.track_concept(rgb_paths, hw, prompt)

    # ---- 2) identity merging + IoU repair (paper Sec. 3) ------------------
    tracks = build_tracks(per_concept_obj, cfg.category_ids, cfg.identity)
    log.info("[%s] %d tracks after identity post-processing.", seq.seq_id, len(tracks))

    # ---- 3) per-frame depth layers, masks, annotations --------------------
    depth_scale = detect_unit_scale(seq, cfg.depth)
    out_seq = Path(cfg.output.out_root) / seq.batch / seq.name
    d_obj = out_seq / cfg.output.object_mask_dir
    d_gnd = out_seq / cfg.output.ground_mask_dir
    d_obj.mkdir(parents=True, exist_ok=True)
    d_gnd.mkdir(parents=True, exist_ok=True)

    frames_manifest: Dict[str, Dict] = {}
    for fi, frec in enumerate(seq.frames):
        depth_m = load_depth_meters(frec, cfg.depth, depth_scale)
        if depth_m.shape != hw:
            depth_m = cv2.resize(depth_m, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)

        instances = []
        layer_by_gid: Dict[int, float] = {}
        for gid, t in tracks.items():
            m = t.masks.get(fi)
            if m is None:
                continue
            bbox = _bbox_from_mask(m)
            if bbox is None:
                continue
            layer, n_valid = _depth_layer(m, depth_m)
            layer_by_gid[gid] = layer if layer > 0 else np.inf
            instances.append(dict(
                track_id=gid,
                category=t.category,
                category_id=t.category_id,
                bbox_xyxy=bbox,
                area=int(m.sum()),
                depth_layer_m=round(layer, 4),
                depth_valid_px=n_valid,
                score=round(float(t.scores.get(fi, 1.0)), 4),
            ))

        id_map = _flatten_id_map(tracks, fi, hw, layer_by_gid)
        ground = _ground_union(per_concept_ground, fi, hw)
        ground &= id_map == 0                      # objects always win over ground

        cv2.imwrite(str(d_obj / f"{frec.name}.png"), id_map)
        cv2.imwrite(str(d_gnd / f"{frec.name}.png"), ground.astype(np.uint8) * 255)

        frames_manifest[frec.name] = dict(
            rgb=str(frec.rgb),
            depth_npy=str(frec.depth_npy) if frec.depth_npy else None,
            depth_png=str(frec.depth_png) if frec.depth_png else None,
            object_mask=str(d_obj / f"{frec.name}.png"),
            ground_mask=str(d_gnd / f"{frec.name}.png"),
            instances=instances,
        )

        if cfg.output.preview_dir and fi % cfg.output.preview_every == 0:
            _preview(frec.rgb, id_map, ground,
                     out_seq / cfg.output.preview_dir / f"{frec.name}.jpg")

    if own_seg:
        obj_segmenter.close()

    manifest = dict(
        sequence=seq.seq_id,
        batch=seq.batch,
        name=seq.name,
        image_hw=list(hw),
        num_frames=len(seq.frames),
        num_tracks=len(tracks),
        depth_scale_to_m=depth_scale,
        max_depth_m=cfg.depth.max_depth_m,
        categories={p: cid for p, cid in cfg.category_ids.items()},
        frames=frames_manifest,
    )
    with open(out_seq / cfg.output.annotation_file, "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest
