"""Configuration for the GID-style data engine.

Reproduces the GID dataset construction pipeline (Sec. 3 of the paper) on a
custom RGB-D dataset, replacing the paper's SAM[27]+DEVA[13] two-stage pipeline
with SAM3 promptable-concept video segmentation + tracking.

Faithfulness notes
-------------------
[Paper Specified]   Annotations per frame: instance masks, bounding boxes,
                    consistent tracking identities, ground masks, depth from a
                    depth sensor; IoU-based identity matching; ~20% test split
                    chosen at video level; depth visual range 0.01-10.0 m.
[Strongly Inferred] Instance "depth layer" GT = average GT depth inside the
                    instance mask (Sec. 4.2.1: "instance depth layer Dep_i,
                    representing the average depth of the instance").
[Reasonable Assumption]
                    SAM3 replaces SAM (mask extraction) + DEVA (identity
                    tracking): SAM3's concept-prompted video mode produces masks
                    AND temporally consistent IDs in one pass. Ground masks come
                    from SAM3 text prompts ("floor"/"ground") instead of the
                    paper's manual dot prompts, since no human annotator is in
                    the loop. All thresholds below are engineering defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SequenceLayout:
    """Folder names inside each <Batch>/<timestamp>/ sequence directory."""

    rgb_dir: str = "left_rgb"
    depth_png_dir: str = "left_filled"
    depth_npy_dir: str = "left_filled_np"
    rgb_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png")
    depth_png_exts: Tuple[str, ...] = (".png",)
    depth_npy_exts: Tuple[str, ...] = (".npy", ".npz")


@dataclass
class DepthConfig:
    """How raw depth files are decoded into metric meters."""

    # "auto" | "m" | "mm" | "cm". auto: median positive value > 80 -> mm.
    unit: str = "auto"
    # Multiplier applied AFTER unit conversion (e.g. sensor-specific scale).
    extra_scale: float = 1.0
    # Metric clamp range; matches the paper's GID visual range (0.01-10.0 m)
    # and MAX_d = 10.0 used throughout the model configs.
    min_depth_m: float = 0.01
    max_depth_m: float = 10.0
    # Prefer .npy (lossless float) over 16-bit PNG when both exist.
    prefer_npy: bool = True


@dataclass
class SAM3Config:
    """SAM3 backend settings."""

    # "native" = github.com/facebookresearch/sam3 session API
    # "hf"     = HuggingFace transformers Sam3Video* API
    # "mock"   = synthetic segmenter (CI / dry-run, no GPU needed)
    backend: str = "native"
    checkpoint_path: Optional[str] = None      # native backend .pt path (None -> default)
    hf_model_id: str = "facebook/sam3"
    device: str = "cuda"
    # One SAM3 video session tracks ONE text concept; categories therefore run
    # as separate sessions whose outputs are merged (see identity.py).
    object_prompts: Tuple[str, ...] = ("person",)
    ground_prompts: Tuple[str, ...] = ("floor", "ground")
    # Per-object detection score floor (SAM3 obj_id_to_score).
    min_object_score: float = 0.5
    # Drop masks smaller than this fraction of the image area (specks).
    min_mask_area_frac: float = 1e-4
    # Frame index where the text prompt is injected.
    prompt_frame_index: int = 0


@dataclass
class IdentityConfig:
    """Cross-concept merging + IoU-based temporal identity repair (Sec. 3)."""

    # Two masks from DIFFERENT concept sessions overlapping above this IoU on
    # the same frame are duplicates; the higher-score one wins.
    cross_concept_dedup_iou: float = 0.75
    # Paper: masks "matched to prior masks using the highest IoU to maintain
    # identity consistency". A track that dies and a track that is born within
    # `max_gap` frames are merged when their boundary masks exceed this IoU.
    reid_iou: float = 0.5
    max_gap: int = 10
    # Minimum track length (frames); shorter tracks are discarded as noise.
    min_track_length: int = 5


@dataclass
class OutputConfig:
    out_root: str = "gid_custom"
    # uint16 PNG: pixel value == track_id (0 = background).
    object_mask_dir: str = "object_masks"
    ground_mask_dir: str = "ground_masks"
    annotation_file: str = "annotations.json"
    preview_dir: Optional[str] = None          # set e.g. "preview" to dump overlays
    preview_every: int = 60


@dataclass
class SplitConfig:
    # Paper Sec. 3: "we assign a larger proportion (20%) to the test set",
    # chosen at the video level.
    test_fraction: float = 0.20
    seed: int = 2026
    stratify_by_batch: bool = True


@dataclass
class DataEngineConfig:
    dataset_root: str = "Dataset"              # contains Batch 1 .. Batch 10
    layout: SequenceLayout = field(default_factory=SequenceLayout)
    depth: DepthConfig = field(default_factory=DepthConfig)
    sam3: SAM3Config = field(default_factory=SAM3Config)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    # Map prompt text -> category id used in annotations (and in the model's
    # `num_classes`). Filled automatically from sam3.object_prompts if empty.
    category_ids: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.category_ids:
            self.category_ids = {p: i for i, p in enumerate(self.sam3.object_prompts)}

    # ------------------------------------------------------------------ io
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataEngineConfig":
        def sub(klass, key):
            raw = dict(d.get(key, {}))
            for k, v in raw.items():
                if isinstance(v, list):
                    raw[k] = tuple(v)
            return klass(**raw)

        return cls(
            dataset_root=d.get("dataset_root", "Dataset"),
            layout=sub(SequenceLayout, "layout"),
            depth=sub(DepthConfig, "depth"),
            sam3=sub(SAM3Config, "sam3"),
            identity=sub(IdentityConfig, "identity"),
            output=sub(OutputConfig, "output"),
            split=sub(SplitConfig, "split"),
            category_ids=dict(d.get("category_ids", {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DataEngineConfig":
        import yaml

        with open(path, "r") as f:
            return cls.from_dict(yaml.safe_load(f) or {})
