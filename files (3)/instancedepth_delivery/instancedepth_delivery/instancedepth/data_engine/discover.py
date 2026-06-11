"""Discover sequences in the custom dataset tree.

Expected layout (user's dataset):

    Dataset/
      Batch 1/ ... Batch 10/
        20260105_012545/                # one video sequence per timestamp dir
          left_rgb/        frame_*.jpg
          left_filled/     frame_*.png  (16-bit depth)
          left_filled_np/  frame_*.npy  (float depth)

Pairing strategy
----------------
RGB and depth frames are paired by file stem when stems match; otherwise by
sorted (natural-order) index with a warning. Sequences with zero pairable
frames are skipped.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import DataEngineConfig, SequenceLayout

log = logging.getLogger("data_engine.discover")

_NUM_RE = re.compile(r"(\d+)")


def natural_key(p: Path) -> Tuple:
    """Sort 'frame_2' before 'frame_10'."""
    return tuple(int(t) if t.isdigit() else t for t in _NUM_RE.split(p.stem))


@dataclass
class FrameRecord:
    name: str                       # canonical frame name (rgb stem)
    rgb: Path
    depth_npy: Optional[Path] = None
    depth_png: Optional[Path] = None

    @property
    def has_depth(self) -> bool:
        return self.depth_npy is not None or self.depth_png is not None


@dataclass
class SequenceRecord:
    batch: str                      # e.g. "Batch 1"
    name: str                       # e.g. "20260105_012545"
    root: Path
    frames: List[FrameRecord] = field(default_factory=list)

    @property
    def seq_id(self) -> str:
        return f"{self.batch}/{self.name}"

    def __len__(self) -> int:
        return len(self.frames)


def _list_files(d: Path, exts: Tuple[str, ...]) -> List[Path]:
    if not d.is_dir():
        return []
    files = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files, key=natural_key)


def _pair_by_stem_or_index(
    rgb: List[Path], dep: List[Path], kind: str, seq_id: str
) -> Dict[str, Path]:
    """Return mapping rgb-stem -> depth path."""
    if not dep:
        return {}
    dep_by_stem = {p.stem: p for p in dep}
    # Depth stems are often "<rgbstem>_depth" or identical; try both.
    out: Dict[str, Path] = {}
    for r in rgb:
        for cand in (r.stem, f"{r.stem}_depth", r.stem.replace("_rgb", "")):
            if cand in dep_by_stem:
                out[r.stem] = dep_by_stem[cand]
                break
    if len(out) >= min(len(rgb), len(dep)):
        return out
    # Fallback: positional pairing.
    if len(rgb) != len(dep):
        log.warning(
            "[%s] %s: stem match failed and counts differ (rgb=%d, depth=%d); "
            "pairing the first %d by sorted index.",
            seq_id, kind, len(rgb), len(dep), min(len(rgb), len(dep)),
        )
    return {r.stem: d for r, d in zip(rgb, dep)}


def discover_sequence(seq_dir: Path, batch: str, layout: SequenceLayout) -> Optional[SequenceRecord]:
    rgb = _list_files(seq_dir / layout.rgb_dir, layout.rgb_exts)
    if not rgb:
        return None
    npy = _list_files(seq_dir / layout.depth_npy_dir, layout.depth_npy_exts)
    png = _list_files(seq_dir / layout.depth_png_dir, layout.depth_png_exts)

    seq = SequenceRecord(batch=batch, name=seq_dir.name, root=seq_dir)
    npy_map = _pair_by_stem_or_index(rgb, npy, "depth_npy", seq.seq_id)
    png_map = _pair_by_stem_or_index(rgb, png, "depth_png", seq.seq_id)

    for r in rgb:
        rec = FrameRecord(name=r.stem, rgb=r,
                          depth_npy=npy_map.get(r.stem), depth_png=png_map.get(r.stem))
        if rec.has_depth:
            seq.frames.append(rec)
        else:
            log.warning("[%s] frame %s has no depth; dropped.", seq.seq_id, r.stem)
    return seq if seq.frames else None


def discover_dataset(cfg: DataEngineConfig) -> List[SequenceRecord]:
    """Walk Dataset/Batch*/<timestamp>/ and return all usable sequences."""
    root = Path(cfg.dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset_root not found: {root}")

    batch_dirs = sorted(
        [d for d in root.iterdir() if d.is_dir() and d.name.lower().startswith("batch")],
        key=natural_key,
    )
    if not batch_dirs:                       # tolerate a flat layout too
        batch_dirs = [root]

    sequences: List[SequenceRecord] = []
    for b in batch_dirs:
        for seq_dir in sorted([d for d in b.iterdir() if d.is_dir()], key=natural_key):
            seq = discover_sequence(seq_dir, batch=b.name, layout=cfg.layout)
            if seq is not None:
                sequences.append(seq)
    log.info("Discovered %d sequences, %d frames total.",
             len(sequences), sum(len(s) for s in sequences))
    return sequences
