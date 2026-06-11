"""GID-style PyTorch dataset for InstanceDepth training/eval.

Each sample feeds every phase of the uploaded model code without adaptation:

    image      (3, H, W) float32, ImageNet-normalized, H,W divisible by 14
               (DINOv2DPTBackbone asserts divisibility by patch_size=14).
    depth      (1, H, W) float32 metric meters; 0 = invalid
               -> SigLogLoss / range_segmentation_loss (losses/hdi_losses.py).
    ground     (1, H, W) float32 {0,1} ground mask (auxiliary supervision /
               eval masking; the GID paper annotates it, usage is optional).
    targets    dict consumed verbatim by HungarianMatcher + InstanceSetCriterion:
                 labels (G,)  long       category ids
                 masks  (G, H, W) float  binary instance masks
                 depths (G,)  float      GT instance depth layers (meters)
    meta       sequence / frame bookkeeping.

[Strongly Inferred] Image resolution: the paper never states the training
crop. DINOv2/14 requires multiples of 14; Depth-Anything-V2's standard
fine-tuning resolution is 518x518, and the ablation baseline is a DA-V2
encoder, so 518x518 is the most defensible default (configurable).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class GIDDatasetConfig:
    annotations_root: str                  # output.out_root of the data engine
    split: str = "train"                   # "train" | "test"
    image_size: Tuple[int, int] = (518, 518)   # (H, W), both divisible by 14
    max_depth: float = 10.0
    min_instance_px: int = 64              # drop instances smaller than this AFTER resize
    require_valid_depth_layer: bool = True # drop instances with no valid GT depth
    hflip_prob: float = 0.5                # train-time augmentation
    color_jitter: float = 0.0              # 0 disables; e.g. 0.2 for mild jitter

    def __post_init__(self) -> None:
        h, w = self.image_size
        assert h % 14 == 0 and w % 14 == 0, "image_size must be divisible by 14 (ViT-L/14)"


class GIDInstanceDepthDataset(Dataset):
    """Frame-level dataset over the generated GID-style annotations."""

    def __init__(self, cfg: GIDDatasetConfig) -> None:
        self.cfg = cfg
        root = Path(cfg.annotations_root)
        split_file = root / f"{cfg.split}.txt"
        seq_ids = [s for s in split_file.read_text().splitlines() if s.strip()]

        with open(root / "meta.json") as f:
            self.meta = json.load(f)

        self.index: List[Tuple[Dict, str]] = []     # (sequence manifest, frame key)
        self._manifests: Dict[str, Dict] = {}
        for sid in seq_ids:
            man_path = root / sid / "annotations.json"
            with open(man_path) as f:
                man = json.load(f)
            self._manifests[sid] = man
            for fkey in sorted(man["frames"]):
                self.index.append((man, fkey))

    def __len__(self) -> int:
        return len(self.index)

    # ------------------------------------------------------------------ io
    @staticmethod
    def _load_rgb(path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"failed to read rgb {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _load_depth(frame: Dict, scale: float) -> np.ndarray:
        if frame.get("depth_npy"):
            d = np.load(frame["depth_npy"]).astype(np.float32)
        else:
            d = cv2.imread(frame["depth_png"], cv2.IMREAD_UNCHANGED).astype(np.float32)
        return d * scale

    # ------------------------------------------------------------- sample
    def __getitem__(self, i: int) -> Dict[str, object]:
        cfg = self.cfg
        man, fkey = self.index[i]
        frame = man["frames"][fkey]
        H, W = cfg.image_size

        rgb = self._load_rgb(frame["rgb"])
        depth = self._load_depth(frame, man["depth_scale_to_m"])
        depth[(depth < 0) | (depth > cfg.max_depth) | ~np.isfinite(depth)] = 0.0
        id_map = cv2.imread(frame["object_mask"], cv2.IMREAD_UNCHANGED)
        ground = cv2.imread(frame["ground_mask"], cv2.IMREAD_GRAYSCALE)

        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
        id_map = cv2.resize(id_map, (W, H), interpolation=cv2.INTER_NEAREST)
        ground = cv2.resize(ground, (W, H), interpolation=cv2.INTER_NEAREST)

        flip = cfg.split == "train" and random.random() < cfg.hflip_prob
        if flip:
            rgb, depth = rgb[:, ::-1], depth[:, ::-1]
            id_map, ground = id_map[:, ::-1], ground[:, ::-1]

        if cfg.split == "train" and cfg.color_jitter > 0:
            j = 1.0 + np.random.uniform(-cfg.color_jitter, cfg.color_jitter, size=3)
            rgb = np.clip(rgb.astype(np.float32) * j[None, None, :], 0, 255)

        img = rgb.astype(np.float32) / 255.0
        img = (img - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
        img_t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))

        # ---- instance targets ------------------------------------------------
        labels: List[int] = []
        masks: List[np.ndarray] = []
        depths: List[float] = []
        track_ids: List[int] = []
        for inst in frame["instances"]:
            gid = inst["track_id"]
            m = id_map == gid
            if m.sum() < cfg.min_instance_px:
                continue
            layer = float(inst["depth_layer_m"])
            if cfg.require_valid_depth_layer and not (0.0 < layer <= cfg.max_depth):
                continue
            labels.append(int(inst["category_id"]))
            masks.append(m)
            depths.append(layer)
            track_ids.append(gid)

        if masks:
            masks_t = torch.from_numpy(np.stack(masks)).float()
        else:
            masks_t = torch.zeros(0, H, W)
        targets = dict(
            labels=torch.tensor(labels, dtype=torch.long),
            masks=masks_t,
            depths=torch.tensor(depths, dtype=torch.float32),
            track_ids=torch.tensor(track_ids, dtype=torch.long),
        )
        return dict(
            image=img_t,
            depth=torch.from_numpy(np.ascontiguousarray(depth)).unsqueeze(0),
            ground=torch.from_numpy(np.ascontiguousarray(ground > 0)).float().unsqueeze(0),
            targets=targets,
            meta=dict(sequence=man["sequence"], frame=fkey, flipped=flip),
        )


def collate_gid(batch: List[Dict]) -> Dict[str, object]:
    """Stack dense tensors; keep variable-length targets as a list (the
    HungarianMatcher / InstanceSetCriterion signature)."""
    return dict(
        image=torch.stack([b["image"] for b in batch]),
        depth=torch.stack([b["depth"] for b in batch]),
        ground=torch.stack([b["ground"] for b in batch]),
        targets=[b["targets"] for b in batch],
        meta=[b["meta"] for b in batch],
    )


# --------------------------------------------------------------------------- #
@torch.no_grad()
def build_refine_targets(
    pair_query_idx: torch.Tensor,        # (P, 2) query indices from OcclusionAwareRefinement
    batch_index: torch.Tensor,           # (P,)
    indices: List[Tuple[torch.Tensor, torch.Tensor]],  # HungarianMatcher output
    targets: List[Dict[str, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map refined (main, guest) query pairs to GT depth layers DT (Eq. 10-11).

    A query participates in the refinement loss only if the matcher assigned it
    a GT instance; pairs with any unmatched member are dropped (mask=False).

    Returns
    -------
    dt    (P, 2) GT depth layers (0 where invalid)
    valid (P,)   bool — both members matched to GT
    """
    P = pair_query_idx.shape[0]
    dt = pair_query_idx.new_zeros((P, 2), dtype=torch.float32)
    valid = torch.zeros(P, dtype=torch.bool, device=pair_query_idx.device)
    # per-image query -> gt depth lookup
    lut: List[Dict[int, float]] = []
    for (pi, gi), tgt in zip(indices, targets):
        lut.append({int(p): float(tgt["depths"][g]) for p, g in zip(pi.tolist(), gi.tolist())})
    for p in range(P):
        b = int(batch_index[p])
        q_main, q_guest = int(pair_query_idx[p, 0]), int(pair_query_idx[p, 1])
        if q_main in lut[b] and q_guest in lut[b]:
            dt[p, 0] = lut[b][q_main]
            dt[p, 1] = lut[b][q_guest]
            valid[p] = True
    return dt, valid
