"""Split + statistics + CLI for the GID-style data engine.

split_sequences : video-level split with a 20% test fraction (paper Sec. 3
                  assigns "a larger proportion (20%) to the test set"),
                  stratified by Batch so every recording condition appears in
                  both splits.  [Paper Specified: fraction & video-level;
                  Reasonable Assumption: batch stratification.]
compute_statistics : Fig. 3 analogs — object counts per depth-range bucket
                  (3a) and per-video (avg objects, num frames) scatter data (3b).
main            : `python -m instancedepth.data_engine.run_generate --config cfg.yaml`
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from .annotate import annotate_sequence
from .config import DataEngineConfig
from .discover import SequenceRecord, discover_dataset
from .sam3_engine import build_segmenter

log = logging.getLogger("data_engine.run")


# --------------------------------------------------------------------------- #
def split_sequences(
    sequences: List[SequenceRecord], cfg: DataEngineConfig
) -> Tuple[List[str], List[str]]:
    rng = random.Random(cfg.split.seed)
    train_ids: List[str] = []
    test_ids: List[str] = []

    if cfg.split.stratify_by_batch:
        by_batch: Dict[str, List[SequenceRecord]] = {}
        for s in sequences:
            by_batch.setdefault(s.batch, []).append(s)
        groups = list(by_batch.values())
    else:
        groups = [list(sequences)]

    for group in groups:
        ids = sorted(s.seq_id for s in group)
        rng.shuffle(ids)
        n_test = max(1, round(cfg.split.test_fraction * len(ids))) if len(ids) > 1 else 0
        test_ids += ids[:n_test]
        train_ids += ids[n_test:]
    return sorted(train_ids), sorted(test_ids)


# --------------------------------------------------------------------------- #
def compute_statistics(manifests: List[Dict], bucket_m: float = 2.0) -> Dict:
    """Depth-range population (Fig. 3a) and per-video object/frame stats (Fig. 3b)."""
    depth_buckets: Counter = Counter()
    per_video = []
    for man in manifests:
        counts = []
        for fr in man["frames"].values():
            counts.append(len(fr["instances"]))
            for inst in fr["instances"]:
                d = inst["depth_layer_m"]
                if d > 0:
                    lo = int(d // bucket_m) * bucket_m
                    depth_buckets[f"{lo:.0f}-{lo + bucket_m:.0f}m"] += 1
        per_video.append(dict(
            sequence=man["sequence"],
            num_frames=man["num_frames"],
            num_tracks=man["num_tracks"],
            avg_objects_per_frame=round(sum(counts) / max(len(counts), 1), 2),
        ))
    return dict(
        depth_range_object_counts=dict(sorted(depth_buckets.items())),
        per_video=per_video,
        total_frames=sum(m["num_frames"] for m in manifests),
        total_videos=len(manifests),
        avg_video_len=round(
            sum(m["num_frames"] for m in manifests) / max(len(manifests), 1), 1
        ),
        avg_objects=round(
            sum(v["avg_objects_per_frame"] for v in per_video) / max(len(per_video), 1), 2
        ),
    )


# --------------------------------------------------------------------------- #
def _load_existing_manifests(out_root: Path) -> List[Dict]:
    """Scan out_root for every already-generated annotations.json."""
    manifests = []
    for p in sorted(out_root.glob("*/*/annotations.json")):
        with open(p) as f:
            manifests.append(json.load(f))
    return manifests


def _split_ids(items: List[Tuple[str, str]], cfg: DataEngineConfig
               ) -> Tuple[List[str], List[str]]:
    """items: (batch, seq_id). Video-level 20% split, stratified by batch."""
    rng = random.Random(cfg.split.seed)
    groups: Dict[str, List[str]] = {}
    for batch, sid in items:
        key = batch if cfg.split.stratify_by_batch else "_all_"
        groups.setdefault(key, []).append(sid)
    train_ids: List[str] = []
    test_ids: List[str] = []
    for ids in groups.values():
        ids = sorted(ids)
        rng.shuffle(ids)
        n_test = max(1, round(cfg.split.test_fraction * len(ids))) if len(ids) > 1 else 0
        test_ids += ids[:n_test]
        train_ids += ids[n_test:]
    return sorted(train_ids), sorted(test_ids)


def finalize(cfg: DataEngineConfig) -> Dict:
    """Build train/test split + statistics from ALL sequences annotated so
    far (scans out_root). Run this ONCE, after every sequence is processed —
    partial annotation runs intentionally do not touch the split files."""
    out_root = Path(cfg.output.out_root)
    manifests = _load_existing_manifests(out_root)
    if not manifests:
        raise FileNotFoundError(f"no annotations.json found under {out_root}")
    train_ids, test_ids = _split_ids(
        [(m["batch"], m["sequence"]) for m in manifests], cfg)
    (out_root / "train.txt").write_text("\n".join(train_ids) + "\n")
    (out_root / "test.txt").write_text("\n".join(test_ids) + "\n")
    stats = compute_statistics(manifests)
    meta = dict(
        categories={p: cid for p, cid in cfg.category_ids.items()},
        max_depth_m=cfg.depth.max_depth_m,
        num_train=len(train_ids),
        num_test=len(test_ids),
        statistics=stats,
    )
    with open(out_root / "meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    log.info("Finalized: %d videos, %d frames, train/test = %d/%d.",
             stats["total_videos"], stats["total_frames"],
             len(train_ids), len(test_ids))
    return meta


def generate(
    cfg: DataEngineConfig,
    limit: int | None = None,
    only: List[str] | None = None,
    skip_existing: bool = False,
    do_finalize: bool = True,
) -> Dict | None:
    sequences = discover_dataset(cfg)

    if only:
        sequences = [s for s in sequences
                     if any(pat in s.seq_id for pat in only)]
        if not sequences:
            raise ValueError(f"--only {only} matched no sequences")
    if skip_existing:
        out_root = Path(cfg.output.out_root)
        before = len(sequences)
        sequences = [
            s for s in sequences
            if not (out_root / s.batch / s.name / cfg.output.annotation_file).exists()
        ]
        log.info("skip-existing: %d already done, %d remaining.",
                 before - len(sequences), len(sequences))
    if limit:
        sequences = sequences[:limit]

    if sequences:
        segmenter = build_segmenter(cfg.sam3)
        try:
            for i, seq in enumerate(sequences):
                log.info("=== [%d/%d] %s (%d frames) ===", i + 1, len(sequences),
                         seq.seq_id, len(seq))
                annotate_sequence(seq, cfg, segmenter, segmenter)
        finally:
            segmenter.close()
    else:
        log.info("nothing to annotate.")

    if do_finalize:
        return finalize(cfg)
    log.info("Partial run complete. Split/meta NOT updated; run with "
             "--finalize after all sequences are annotated.")
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate GID-style annotations with SAM3.")
    ap.add_argument("--config", required=True, help="YAML DataEngineConfig")
    ap.add_argument("--limit", type=int, default=None,
                    help="annotate at most N (pending) sequences this run")
    ap.add_argument("--only", action="append", default=None, metavar="SUBSTR",
                    help="annotate only sequences whose 'Batch/timestamp' id "
                         "contains SUBSTR (repeatable), e.g. "
                         "--only 'Batch_1/20260120_094343' or --only Batch_3")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip sequences that already have annotations.json")
    ap.add_argument("--finalize", action="store_true",
                    help="only (re)build train/test split + meta.json from all "
                         "annotations generated so far; no SAM3 runs")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = DataEngineConfig.from_yaml(args.config)

    if args.finalize:
        finalize(cfg)
        return
    partial = bool(args.limit or args.only or args.skip_existing)
    generate(cfg, limit=args.limit, only=args.only,
             skip_existing=args.skip_existing, do_finalize=not partial)


if __name__ == "__main__":
    main()
